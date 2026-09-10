"""One place that knows where MLflow lives and what every run logs."""

from __future__ import annotations

from pathlib import Path

import mlflow

from transithunter import config

EXPERIMENT = "transithunter"


def start_run(run_name: str, tags: dict[str, str] | None = None):
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=config.MLFLOW_ARTIFACT_ROOT)
    else:
        _assert_artifact_root(experiment.artifact_location)
    mlflow.set_experiment(EXPERIMENT)
    return mlflow.start_run(run_name=run_name, tags=tags)


def _assert_artifact_root(recorded: str) -> None:
    """Fail loudly when the experiment writes artifacts somewhere else.

    MLflow stores artifact_location per experiment when the experiment is
    created and never reads MLFLOW_ARTIFACT_ROOT again, so moving the store
    in config.py silently does nothing to an experiment that already exists.
    That is how a run's weights and plots ended up back on C: once.
    """
    local = Path(recorded.removeprefix("file:///").removeprefix("file:"))
    if local.resolve() == config.MLFLOW_ARTIFACT_DIR.resolve():
        return
    raise RuntimeError(
        f"experiment {EXPERIMENT!r} writes artifacts to {recorded}, not "
        f"{config.MLFLOW_ARTIFACT_ROOT}. MLflow cannot change this after the "
        f"experiment exists. Move the files, then rewrite artifact_location in "
        f"experiments and artifact_uri in runs inside {config.MLFLOW_DB_PATH}."
    )


def log_split_metrics(prefix: str, metrics: dict[str, float], step: int | None = None) -> None:
    mlflow.log_metrics({f"{prefix}_{k}": v for k, v in metrics.items()}, step=step)


def log_files(paths: list[Path], artifact_path: str | None = None) -> None:
    for p in paths:
        mlflow.log_artifact(str(p), artifact_path=artifact_path)


def export_dir(model_name: str) -> Path:
    d = config.ARTIFACTS_DIR / "models" / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d
