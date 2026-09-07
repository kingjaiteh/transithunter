"""One place that knows where MLflow lives and what every run logs."""

from __future__ import annotations

from pathlib import Path

import mlflow

from transithunter import config

EXPERIMENT = "transithunter"


def start_run(run_name: str, tags: dict[str, str] | None = None):
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=config.MLFLOW_ARTIFACT_ROOT)
    mlflow.set_experiment(EXPERIMENT)
    return mlflow.start_run(run_name=run_name, tags=tags)


def log_split_metrics(prefix: str, metrics: dict[str, float], step: int | None = None) -> None:
    mlflow.log_metrics({f"{prefix}_{k}": v for k, v in metrics.items()}, step=step)


def log_files(paths: list[Path], artifact_path: str | None = None) -> None:
    for p in paths:
        mlflow.log_artifact(str(p), artifact_path=artifact_path)


def export_dir(model_name: str) -> Path:
    d = config.ARTIFACTS_DIR / "models" / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d
