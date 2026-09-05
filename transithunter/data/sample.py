"""Choose which KOIs make up dataset v1.

Option 3 from the Phase 0 sizing decision (2026-09-05): a balanced sample of
about 3,000 KOIs first, widening later if the model wants more. Sampling is by
KOI, but downloads and splits are by star, so once a star is in, every
training KOI on that star comes with it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from transithunter import config
from transithunter.data.labels import load_labels, training_rows

SAMPLE_PATH = config.DATA_DIR / "sample_v1.parquet"
PER_CLASS = 1500


def usable(rows: pd.DataFrame) -> pd.DataFrame:
    """Rows we can actually fold: need a period, epoch and duration."""
    need = ["koi_period", "koi_time0bk", "koi_duration", "koi_depth"]
    return rows[rows[need].notna().all(axis=1) & (rows["koi_period"] > 0)]


def choose_sample(rows: pd.DataFrame, per_class: int = PER_CLASS,
                  seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    rows = usable(rows)
    rng = np.random.default_rng(seed)
    chosen_stars: set[int] = set()
    for label in (1, 0):
        pool = rows[rows["label"] == label]
        picks = pool.iloc[rng.choice(len(pool), size=per_class, replace=False)]
        chosen_stars |= set(picks["kepid"])
    out = rows[rows["kepid"].isin(chosen_stars)].copy()
    return out.sort_values(["kepid", "kepoi_name"]).reset_index(drop=True)


def load_sample(path: Path = SAMPLE_PATH, refresh: bool = False) -> pd.DataFrame:
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    sample = choose_sample(training_rows(load_labels()))
    sample.to_parquet(path, index=False)
    return sample


if __name__ == "__main__":
    s = load_sample(refresh=True)
    print(f"{len(s)} KOIs on {s['kepid'].nunique()} stars -> {SAMPLE_PATH}")
    print(s["koi_disposition"].value_counts().to_string())
    est_gb = s["kepid"].nunique() * 6.9 / 1000
    print(f"estimated download: {est_gb:.1f} GB, {s['kepid'].nunique() * 10 / 3600:.1f} h")
