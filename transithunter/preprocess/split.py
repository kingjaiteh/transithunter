"""Train / validation / test split by star.

One star can host several KOIs, and every KOI on a star is cut from the same
light curve. Splitting by KOI would put near-identical inputs on both sides of
the test boundary, so the unit of assignment is `kepid`. Stratified on each
star's majority label so class balance is similar across the three sets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from transithunter import config

FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def assign_splits(table: pd.DataFrame, seed: int = config.RANDOM_SEED) -> pd.Series:
    """Return a `split` value per row, identical for rows sharing a `kepid`."""
    star_label = table.groupby("kepid")["label"].mean().round().astype(int)
    rng = np.random.default_rng(seed)
    star_split: dict[int, str] = {}
    for label in sorted(star_label.unique()):
        stars = star_label.index[star_label == label].to_numpy()
        rng.shuffle(stars)
        n = len(stars)
        n_train = int(round(FRACTIONS["train"] * n))
        n_val = int(round(FRACTIONS["val"] * n))
        for k in stars[:n_train]:
            star_split[k] = "train"
        for k in stars[n_train:n_train + n_val]:
            star_split[k] = "val"
        for k in stars[n_train + n_val:]:
            star_split[k] = "test"
    return table["kepid"].map(star_split)


def assert_no_star_leakage(table: pd.DataFrame) -> None:
    per_star = table.groupby("kepid")["split"].nunique()
    leaked = per_star[per_star > 1]
    if len(leaked):
        raise AssertionError(f"{len(leaked)} stars appear in more than one split: {leaked.index[:5].tolist()}")
