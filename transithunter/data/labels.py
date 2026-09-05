"""Fetch the Kepler KOI cumulative table from the Exoplanet Archive and cache it.

The parquet on disk keeps everything useful for analysis, including vetting
columns that must never become model inputs. `feature_frame` is the single
choke point that turns the table into model inputs, and it runs the
`features.assert_clean` check every time.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

from transithunter import config
from transithunter.features import ALLOWED_FEATURES, assert_clean

IDENTIFIERS = ("kepid", "kepoi_name", "kepler_name")
LABELS = ("koi_disposition", "koi_pdisposition")
# Needed to phase-fold each KOI. Not model inputs.
EPHEMERIS = ("koi_time0bk", "koi_tce_plnt_num")
# Kept only for error analysis in Phase 4. Forbidden as features.
VETTING = ("koi_score", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec")

COLUMNS = IDENTIFIERS + LABELS + EPHEMERIS + VETTING + ALLOWED_FEATURES


def build_query(columns: tuple[str, ...] = COLUMNS) -> str:
    return f"select {', '.join(columns)} from {config.KOI_TABLE}"


def fetch_koi_table(timeout: float = 120) -> pd.DataFrame:
    """One synchronous TAP call. The table is about 9,600 rows, a few MB."""
    response = requests.get(
        config.TAP_SYNC_URL,
        params={"query": build_query(), "format": "csv"},
        timeout=timeout,
    )
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    df["kepid"] = df["kepid"].astype("int64")
    return df


def load_labels(path: Path = config.LABELS_PATH, refresh: bool = False) -> pd.DataFrame:
    """Return the cached table, fetching it on first use or when refresh is set."""
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = fetch_koi_table()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """CONFIRMED and FALSE POSITIVE only, with a binary `label` column."""
    keep = df["koi_disposition"].isin([config.POSITIVE_LABEL, config.NEGATIVE_LABEL])
    out = df.loc[keep].copy()
    out["label"] = (out["koi_disposition"] == config.POSITIVE_LABEL).astype("int8")
    return out


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Model inputs only. Raises if a forbidden column is requested."""
    columns = list(ALLOWED_FEATURES)
    assert_clean(columns)
    return df[columns]


if __name__ == "__main__":
    table = load_labels(refresh=True)
    print(f"saved {len(table)} rows x {table.shape[1]} cols to {config.LABELS_PATH}")
    print(table["koi_disposition"].value_counts().to_string())
    print(f"distinct stars: {table['kepid'].nunique()}")
