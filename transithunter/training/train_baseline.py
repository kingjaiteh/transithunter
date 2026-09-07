"""Phase 2 baseline: gradient boosting on allowlisted catalogue features.

Run: uv run python -m transithunter.training.train_baseline [--version v1]
Logs params, metrics, PR curve, confusion matrix and predictions to MLflow.
Exports the fitted model and threshold to ARTIFACTS_DIR/models/baseline_<version>/.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import joblib
import mlflow

from transithunter import config
from transithunter.eval.metrics import (
    choose_threshold, evaluate, prevalence, save_plots, save_predictions, write_json,
)
from transithunter.features import ALLOWED_FEATURES
from transithunter.models.baseline import DEFAULT_PARAMS, feature_importance, make_baseline
from transithunter.training.data import describe, load_dataset
from transithunter.training.tracking import export_dir, log_files, log_split_metrics, start_run


def main(version: str = "v1", seed: int = config.RANDOM_SEED) -> dict[str, float]:
    splits = load_dataset(version)
    print(describe(splits))
    tr, va, te = splits["train"], splits["val"], splits["test"]

    with start_run(f"baseline_gbm_{version}", tags={"model": "hgb", "dataset": version}):
        mlflow.log_params({**DEFAULT_PARAMS, "seed": seed, "features": ",".join(ALLOWED_FEATURES),
                           "n_train": len(tr), "n_val": len(va), "n_test": len(te)})
        model = make_baseline(seed).fit(tr.features, tr.labels)

        val_score = model.predict_proba(va.features)[:, 1]
        threshold = choose_threshold(va.labels, val_score)
        val_metrics = evaluate(va.labels, val_score, threshold)
        test_score = model.predict_proba(te.features)[:, 1]
        test_metrics = evaluate(te.labels, test_score, threshold)
        log_split_metrics("val", val_metrics)
        log_split_metrics("test", test_metrics)
        mlflow.log_metric("test_prevalence", prevalence(te.labels))

        importance = feature_importance(model, va.features, va.labels, seed)
        mlflow.log_metrics({f"importance_{k}": v for k, v in importance.items()})

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            files = save_plots(te.labels, test_score, threshold, tmp, "baseline GBM, test")
            files.append(save_predictions(te.names, te.labels, test_score, tmp / "test_predictions.csv"))
            files.append(write_json({"val": val_metrics, "test": test_metrics,
                                     "importance": importance}, tmp / "metrics.json"))
            log_files(files)
        mlflow.sklearn.log_model(model, name="model")

        out = export_dir(f"baseline_{version}")
        joblib.dump(model, out / "model.joblib")
        write_json({"threshold": threshold, "features": list(ALLOWED_FEATURES),
                    "test": test_metrics, "mlflow_run_id": mlflow.active_run().info.run_id},
                   out / "model_card.json")

    print(f"val  PR-AUC {val_metrics['pr_auc']:.3f}  threshold {threshold:.3f}")
    print(f"test PR-AUC {test_metrics['pr_auc']:.3f}  ROC-AUC {test_metrics['roc_auc']:.3f}  "
          f"P {test_metrics['precision']:.3f}  R {test_metrics['recall']:.3f}  "
          f"(random PR-AUC {prevalence(te.labels):.3f})")
    top = sorted(importance.items(), key=lambda kv: -kv[1])[:5]
    print("top features:", ", ".join(f"{k} {v:+.3f}" for k, v in top))
    print(f"exported to {out}")
    return test_metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    a = ap.parse_args()
    main(a.version, a.seed)
