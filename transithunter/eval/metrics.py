"""Metrics and plots shared by every model.

PR-AUC is the headline number (PLAN.md section 8). The decision threshold is
chosen on the validation set only, then applied unchanged to test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def choose_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Threshold on the validation PR curve that maximises F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def evaluate(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((tp + tn) / len(y_true)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def prevalence(y_true: np.ndarray) -> float:
    """PR-AUC of a random classifier, the floor every model must clear."""
    return float(np.mean(y_true))


def save_plots(y_true: np.ndarray, y_score: np.ndarray, threshold: float,
               out_dir: Path, title: str) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, lw=1.5)
    ax.axhline(prevalence(y_true), ls="--", color="grey", lw=1, label="random")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_ylim(0, 1.02); ax.set_xlim(0, 1)
    ax.set_title(f"{title}: PR-AUC {average_precision_score(y_true, y_score):.3f}")
    ax.legend(loc="lower left")
    fig.tight_layout(); p = out_dir / "pr_curve.png"; fig.savefig(p, dpi=120); plt.close(fig)
    paths.append(p)

    cm = confusion_matrix(y_true, (y_score >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="black" if v < cm.max() / 2 else "white")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["FP", "PLANET"]); ax.set_yticklabels(["FP", "PLANET"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    ax.set_title(f"{title} at threshold {threshold:.2f}")
    fig.tight_layout(); p = out_dir / "confusion_matrix.png"; fig.savefig(p, dpi=120); plt.close(fig)
    paths.append(p)
    return paths


def save_predictions(names: np.ndarray, y_true: np.ndarray, y_score: np.ndarray, path: Path) -> Path:
    import pandas as pd

    pd.DataFrame({"kepoi_name": names, "label": y_true, "score": y_score}).to_csv(path, index=False)
    return path


def write_json(obj: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    return path
