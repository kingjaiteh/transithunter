"""Load the assembled dataset once and hand each split to a model.

Both the baseline and the CNN read the same parquet and npz, so the split
boundary and the feature allowlist are enforced in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from transithunter import config
from transithunter.features import ALLOWED_FEATURES, assert_clean

SPLITS = ("train", "val", "test")


@dataclass
class Split:
    names: np.ndarray        # kepoi_name, (N,)
    global_view: np.ndarray  # (N, GLOBAL_BINS) float32
    local_view: np.ndarray   # (N, LOCAL_BINS) float32
    features: np.ndarray     # (N, len(ALLOWED_FEATURES)) float32, NaN where the catalogue is blank
    labels: np.ndarray       # (N,) int8, 1 = CONFIRMED
    table: pd.DataFrame      # the rows themselves, for error analysis

    def __len__(self) -> int:
        return len(self.labels)


def dataset_paths(version: str = "v1") -> tuple[Path, Path]:
    return (config.PROCESSED_DIR / f"dataset_{version}.parquet",
            config.PROCESSED_DIR / f"views_{version}.npz")


def load_dataset(version: str = "v1") -> dict[str, Split]:
    table_path, stack_path = dataset_paths(version)
    table = pd.read_parquet(table_path)
    z = np.load(stack_path)
    if list(z["kepoi_name"]) != list(table["kepoi_name"]):
        raise ValueError("views and table are out of order; rerun scripts/build_dataset.py --assemble-only")
    assert_clean(list(ALLOWED_FEATURES))

    out: dict[str, Split] = {}
    for split in SPLITS:
        mask = (table["split"] == split).to_numpy()
        rows = table.loc[mask]
        out[split] = Split(
            names=rows["kepoi_name"].to_numpy(),
            global_view=z["global_view"][mask].astype(np.float32),
            local_view=z["local_view"][mask].astype(np.float32),
            features=rows[list(ALLOWED_FEATURES)].to_numpy(dtype=np.float32),
            labels=rows["label"].to_numpy(dtype=np.int8),
            table=rows.reset_index(drop=True),
        )
    return out


def describe(splits: dict[str, Split]) -> str:
    lines = []
    for name, s in splits.items():
        pos = int(s.labels.sum())
        lines.append(f"{name}: {len(s)} KOIs, {pos} planets, {len(s) - pos} false positives")
    return "\n".join(lines)
