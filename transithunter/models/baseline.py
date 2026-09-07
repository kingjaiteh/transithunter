"""Gradient boosting on catalogue features: the number the CNN has to beat.

Inputs are the eleven allowlisted transit and stellar columns from the KOI
table. Histogram gradient boosting handles the blanks in the stellar columns
natively, so there is no imputation step to keep in sync at serving time.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from transithunter import config
from transithunter.features import ALLOWED_FEATURES

DEFAULT_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 400,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": False,  # validation is handled by the caller
}


def make_baseline(seed: int = config.RANDOM_SEED, **overrides) -> HistGradientBoostingClassifier:
    params = {**DEFAULT_PARAMS, **overrides}
    return HistGradientBoostingClassifier(random_state=seed, **params)


def feature_importance(model: HistGradientBoostingClassifier, X_val: np.ndarray,
                       y_val: np.ndarray, seed: int = config.RANDOM_SEED) -> dict[str, float]:
    """Permutation importance in PR-AUC, on the validation set."""
    from sklearn.inspection import permutation_importance

    r = permutation_importance(model, X_val, y_val, scoring="average_precision",
                               n_repeats=10, random_state=seed)
    return {name: float(v) for name, v in zip(ALLOWED_FEATURES, r.importances_mean)}
