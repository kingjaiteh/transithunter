"""Paths, endpoints, and the few numbers the whole pipeline agrees on."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Light curves and built datasets go on the secondary drive. C: had 22 GB free
# on 2026-09-02; D: had 164 GB. Override with TRANSITHUNTER_DATA_DIR.
DATA_DIR = Path(os.environ.get("TRANSITHUNTER_DATA_DIR", r"D:\transithunter-data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_PATH = DATA_DIR / "koi_labels.parquet"

# lightkurve 2.6 ignores LIGHTKURVE_CACHE_DIR. Every module that imports
# lightkurve must call configure_lightkurve() first, or files land in
# ~/.cache/lightkurve on C:. transithunter.data.fetch does this on import.
LIGHTKURVE_CACHE_DIR = RAW_DIR


def configure_lightkurve() -> None:
    import lightkurve as lk

    LIGHTKURVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lk.conf.cache_dir = str(LIGHTKURVE_CACHE_DIR)

# Exported model weights. MLflow holds the versioned copies under mlruns/.
ARTIFACTS_DIR = DATA_DIR / "artifacts"
MLFLOW_TRACKING_URI = (PROJECT_ROOT / "mlruns").as_uri()

# NASA Exoplanet Archive Table Access Protocol. No key needed.
TAP_SYNC_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
KOI_TABLE = "cumulative"

# AstroNet view sizes (Shallue & Vanderburg 2018)
GLOBAL_BINS = 2001
LOCAL_BINS = 201

# Labels used for training. CANDIDATE is held aside for Phase 4.
POSITIVE_LABEL = "CONFIRMED"
NEGATIVE_LABEL = "FALSE POSITIVE"
HELDOUT_LABEL = "CANDIDATE"

RANDOM_SEED = 42
