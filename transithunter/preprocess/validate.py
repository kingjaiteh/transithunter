"""Great Expectations suite for the assembled training table, plus array checks.

Run: uv run python -m transithunter.preprocess.validate
Exit code is non-zero when anything fails, so it can gate a training run.
"""

from __future__ import annotations

import sys

import great_expectations as gx
import numpy as np
import pandas as pd

from transithunter import config
from transithunter.features import ALLOWED_FEATURES, assert_clean

DATASET_PATH = config.PROCESSED_DIR / "dataset_v1.parquet"
STACK_PATH = config.PROCESSED_DIR / "views_v1.npz"

E = gx.expectations


def table_suite() -> gx.ExpectationSuite:
    suite = gx.ExpectationSuite(name="transithunter_dataset_v1")
    for exp in [
        E.ExpectColumnValuesToBeUnique(column="kepoi_name"),
        E.ExpectColumnValuesToNotBeNull(column="kepid"),
        E.ExpectColumnValuesToBeInSet(column="label", value_set=[0, 1]),
        E.ExpectColumnValuesToBeInSet(column="split", value_set=["train", "val", "test"]),
        E.ExpectColumnValuesToBeInSet(
            column="koi_disposition", value_set=[config.POSITIVE_LABEL, config.NEGATIVE_LABEL]),
        # PLAN.md: label balance. Sampling is 1500 per class by KOI, siblings widen it a bit.
        E.ExpectColumnMeanToBeBetween(column="label", min_value=0.35, max_value=0.65),
        E.ExpectColumnValuesToBeBetween(column="koi_period", min_value=0.1, max_value=2000),
        E.ExpectColumnValuesToBeBetween(column="koi_duration", min_value=0.1, max_value=200),
        E.ExpectColumnValuesToBeBetween(column="koi_depth", min_value=0),
        E.ExpectColumnValuesToBeBetween(column="n_points", min_value=1000),
        E.ExpectColumnValuesToBeBetween(column="n_transits_seen", min_value=1),
    ]:
        suite.add_expectation(exp)
    # Stellar features may be missing for a few stars, but not for most.
    for col in ("koi_steff", "koi_slogg", "koi_srad", "koi_kepmag", "koi_model_snr"):
        suite.add_expectation(E.ExpectColumnValuesToNotBeNull(column=col, mostly=0.95))
    return suite


def validate_table(df: pd.DataFrame) -> tuple[bool, list[str]]:
    context = gx.get_context(mode="ephemeral")
    batch = (
        context.data_sources.add_pandas("pandas")
        .add_dataframe_asset("dataset")
        .add_batch_definition_whole_dataframe("all")
        .get_batch(batch_parameters={"dataframe": df})
    )
    result = batch.validate(table_suite())
    failures = [
        f"{r.expectation_config.type} {r.expectation_config.kwargs.get('column', '')}: "
        f"{r.result.get('unexpected_percent', r.result)}"
        for r in result.results if not r.success
    ]
    return bool(result.success), failures


def validate_views(df: pd.DataFrame) -> list[str]:
    """Checks GE cannot express: the tensors themselves."""
    z = np.load(STACK_PATH)
    problems = []
    g, l, names = z["global_view"], z["local_view"], z["kepoi_name"]
    if list(names) != list(df["kepoi_name"]):
        problems.append("view order does not match the table")
    if g.shape[1:] != (config.GLOBAL_BINS,) or l.shape[1:] != (config.LOCAL_BINS,):
        problems.append(f"wrong view shapes {g.shape} {l.shape}")
    for name, arr in (("global", g), ("local", l)):
        if not np.isfinite(arr).all():
            problems.append(f"{name} views contain NaN or inf")
        mins = arr.min(axis=1)
        if not np.allclose(mins, -1.0, atol=1e-5):
            problems.append(f"{(~np.isclose(mins, -1.0, atol=1e-5)).sum()} {name} views not normalised to -1")
        flat = arr.std(axis=1) < 1e-6
        if flat.any():
            problems.append(f"{flat.sum()} {name} views are flat")
    return problems


def main() -> int:
    df = pd.read_parquet(DATASET_PATH)
    assert_clean(list(ALLOWED_FEATURES))
    ok, failures = validate_table(df)
    problems = validate_views(df)
    print(f"table: {len(df)} rows, GE {'PASSED' if ok else 'FAILED'}")
    for f in failures:
        print("  table:", f)
    for p in problems:
        print("  views:", p)
    if not problems:
        print("views: PASSED")
    return 0 if ok and not problems else 1


if __name__ == "__main__":
    sys.exit(main())
