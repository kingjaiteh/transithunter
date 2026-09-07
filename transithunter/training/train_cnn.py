"""Phase 2 CNN: AstroNet-style two-branch network on global and local views.

Run: uv run python -m transithunter.training.train_cnn [--epochs 60] [--aux]
Runs on the local GPU when torch sees one, else CPU. Early stopping on
validation PR-AUC; the checkpoint with the best validation PR-AUC is what
gets evaluated on test and exported. Also runs unchanged on Kaggle.
"""

from __future__ import annotations

import argparse
import copy
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn

from transithunter import config
from transithunter.eval.metrics import (
    choose_threshold, evaluate, prevalence, save_plots, save_predictions, write_json,
)
from transithunter.features import ALLOWED_FEATURES
from transithunter.models.cnn import CNNConfig, TransitCNN
from transithunter.training.data import Split, describe, load_dataset
from transithunter.training.tracking import export_dir, log_files, log_split_metrics, start_run


class AuxScaler:
    """Standardise the scalar features on train; blanks become 0 after scaling."""

    def __init__(self, train_features: np.ndarray):
        self.mean = np.nanmean(train_features, axis=0)
        self.std = np.nanstd(train_features, axis=0) + 1e-6

    def __call__(self, features: np.ndarray) -> np.ndarray:
        z = (features - self.mean) / self.std
        return np.nan_to_num(z, nan=0.0).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "features": list(ALLOWED_FEATURES)}


def tensors(split: Split, scaler: AuxScaler | None, device: torch.device) -> tuple:
    g = torch.from_numpy(split.global_view).to(device)
    l = torch.from_numpy(split.local_view).to(device)
    y = torch.from_numpy(split.labels.astype(np.float32)).to(device)
    a = torch.from_numpy(scaler(split.features)).to(device) if scaler else None
    return g, l, a, y


def augment(g: torch.Tensor, l: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A transit is symmetric in phase, so reflecting the views is label-preserving."""
    flip = torch.rand(g.shape[0], device=g.device) < 0.5
    g = torch.where(flip[:, None], g.flip(1), g)
    l = torch.where(flip[:, None], l.flip(1), l)
    return g, l


@torch.no_grad()
def predict(model: nn.Module, g: torch.Tensor, l: torch.Tensor, a: torch.Tensor | None,
            batch: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(g), batch):
        aux = None if a is None else a[i:i + batch]
        out.append(torch.sigmoid(model(g[i:i + batch], l[i:i + batch], aux)).cpu().numpy())
    return np.concatenate(out)


def train(splits: dict[str, Split], cfg: CNNConfig, epochs: int, batch_size: int, lr: float,
          weight_decay: float, patience: int, seed: int, device: torch.device,
          log_every_epoch: bool = True) -> tuple[TransitCNN, AuxScaler | None, list[dict]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = AuxScaler(splits["train"].features) if cfg.n_aux else None
    g_tr, l_tr, a_tr, y_tr = tensors(splits["train"], scaler, device)
    g_va, l_va, a_va, _ = tensors(splits["val"], scaler, device)

    model = TransitCNN(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    best_state, best_auc, best_epoch, since_best = None, -1.0, 0, 0
    history = []
    n = len(y_tr)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            g, l = augment(g_tr[idx], l_tr[idx])
            aux = None if a_tr is None else a_tr[idx]
            loss = loss_fn(model(g, l, aux), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        sched.step()

        val_score = predict(model, g_va, l_va, a_va)
        val_auc = evaluate(splits["val"].labels, val_score, 0.5)["pr_auc"]
        row = {"epoch": epoch, "train_loss": total / n, "val_pr_auc": val_auc}
        history.append(row)
        if log_every_epoch:
            mlflow.log_metrics(row, step=epoch)
        if val_auc > best_auc:
            best_auc, best_epoch, since_best = val_auc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since_best += 1
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  loss {row['train_loss']:.4f}  val PR-AUC {val_auc:.4f}  "
                  f"best {best_auc:.4f} @ {best_epoch}", flush=True)
        if since_best >= patience:
            print(f"early stop at epoch {epoch}, best epoch {best_epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    return model, scaler, history


def main(args: argparse.Namespace) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_dataset(args.version)
    print(describe(splits))
    gpu = f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    print(f"device: {device}{gpu}")
    cfg = CNNConfig(n_aux=len(ALLOWED_FEATURES) if args.aux else 0, dropout=args.dropout)
    te = splits["test"]

    variant = "views_aux" if args.aux else "views"
    run_name = f"cnn_{variant}_{args.version}"
    with start_run(run_name, tags={"model": "astronet_cnn", "dataset": args.version}):
        params = {**cfg.to_dict(), "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                  "weight_decay": args.weight_decay, "patience": args.patience, "seed": args.seed,
                  "device": device.type, "n_train": len(splits["train"]), "n_val": len(splits["val"]),
                  "n_test": len(te)}
        mlflow.log_params({k: str(v) if isinstance(v, tuple) else v for k, v in params.items()})

        t0 = time.time()
        model, scaler, history = train(splits, cfg, args.epochs, args.batch_size, args.lr,
                                       args.weight_decay, args.patience, args.seed, device)
        mlflow.log_metric("train_seconds", time.time() - t0)
        mlflow.log_metric("n_parameters", model.n_parameters())

        g_va, l_va, a_va, _ = tensors(splits["val"], scaler, device)
        g_te, l_te, a_te, _ = tensors(te, scaler, device)
        val_score = predict(model, g_va, l_va, a_va)
        threshold = choose_threshold(splits["val"].labels, val_score)
        val_metrics = evaluate(splits["val"].labels, val_score, threshold)
        test_score = predict(model, g_te, l_te, a_te)
        test_metrics = evaluate(te.labels, test_score, threshold)
        log_split_metrics("val", val_metrics)
        log_split_metrics("test", test_metrics)
        mlflow.log_metric("test_prevalence", prevalence(te.labels))

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            files = save_plots(te.labels, test_score, threshold, tmp, "CNN, test")
            files.append(save_predictions(te.names, te.labels, test_score, tmp / "test_predictions.csv"))
            files.append(write_json({"val": val_metrics, "test": test_metrics, "history": history},
                                    tmp / "metrics.json"))
            log_files(files)

        # Plain state dict plus a card, not mlflow.pytorch.log_model: the pt2
        # flavour traces the graph and serving only needs the weights anyway.
        out = export_dir(run_name)
        torch.save(model.cpu().state_dict(), out / "weights.pt")
        write_json({"config": cfg.to_dict(), "threshold": threshold,
                    "scaler": scaler.to_dict() if scaler else None,
                    "test": test_metrics, "mlflow_run_id": mlflow.active_run().info.run_id},
                   out / "model_card.json")
        log_files([out / "weights.pt", out / "model_card.json"], artifact_path="model")

    print(f"{model.n_parameters():,} parameters, {len(history)} epochs")
    print(f"val  PR-AUC {val_metrics['pr_auc']:.3f}  threshold {threshold:.3f}")
    print(f"test PR-AUC {test_metrics['pr_auc']:.3f}  ROC-AUC {test_metrics['roc_auc']:.3f}  "
          f"P {test_metrics['precision']:.3f}  R {test_metrics['recall']:.3f}  "
          f"(random PR-AUC {prevalence(te.labels):.3f})")
    print(f"exported to {out}")
    return test_metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--aux", action="store_true",
                    help="append allowlisted catalogue features to the head")
    main(ap.parse_args())
