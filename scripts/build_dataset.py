r"""Phase 1: turn every cached star in the sample into global/local views.

Resumable: one .npz per KOI under processed/views_v1/. Re-running only
processes KOIs that have no file yet. Then assembles the feature table with
split labels into processed/dataset_v1.parquet and the stacked views into
processed/views_v1.npz.

Run: uv run python scripts/build_dataset.py [--assemble-only]
"""

from __future__ import annotations

import sys
import time
import traceback

import numpy as np
import pandas as pd

from transithunter import config
from transithunter.data.download import done_stars
from transithunter.data.sample import load_sample
from transithunter.features import ALLOWED_FEATURES, assert_clean
from transithunter.preprocess.split import assert_no_star_leakage, assign_splits
from transithunter.preprocess.views import load_cached, make_views, siblings_of

VIEWS_DIR = config.PROCESSED_DIR / "views_v1"
DATASET_PATH = config.PROCESSED_DIR / "dataset_v1.parquet"
STACK_PATH = config.PROCESSED_DIR / "views_v1.npz"
ERRORS_PATH = config.PROCESSED_DIR / "views_v1_errors.csv"


def process_pending(sample: pd.DataFrame) -> None:
    VIEWS_DIR.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in VIEWS_DIR.glob("*.npz")}
    cached = done_stars()
    todo = sample[sample["kepid"].isin(cached) & ~sample["kepoi_name"].isin(have)]
    print(f"{len(sample)} KOIs in sample, {len(have)} built, {len(todo)} buildable now", flush=True)

    errors = []
    t0 = time.time()
    for i, (kepid, group) in enumerate(todo.groupby("kepid"), 1):
        try:
            lc = load_cached(int(kepid))
        except Exception as exc:  # noqa: BLE001
            for name in group["kepoi_name"]:
                errors.append({"kepoi_name": name, "stage": "load", "error": repr(exc)})
            continue
        for _, koi in group.iterrows():
            try:
                v = make_views(lc, koi, siblings_of(koi, sample))
                np.savez_compressed(
                    VIEWS_DIR / f"{koi['kepoi_name']}.npz",
                    global_view=v.global_view.astype(np.float32),
                    local_view=v.local_view.astype(np.float32),
                    n_points=v.n_points, n_transits_seen=v.n_transits_seen,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append({"kepoi_name": koi["kepoi_name"], "stage": "views",
                               "error": repr(exc), "trace": traceback.format_exc(limit=2)})
        if i % 25 == 0:
            rate = (time.time() - t0) / i
            print(f"  {i} stars, {rate:.1f} s/star, {len(errors)} errors", flush=True)
    if errors:
        pd.DataFrame(errors).to_csv(ERRORS_PATH, index=False)
        print(f"{len(errors)} errors -> {ERRORS_PATH}", flush=True)


def assemble(sample: pd.DataFrame) -> None:
    files = sorted(VIEWS_DIR.glob("*.npz"))
    names = [p.stem for p in files]
    rows = sample.set_index("kepoi_name").loc[names].reset_index()
    rows["split"] = assign_splits(rows)
    assert_no_star_leakage(rows)
    assert_clean(list(ALLOWED_FEATURES))

    globals_, locals_, n_points, n_transits = [], [], [], []
    for p in files:
        z = np.load(p)
        globals_.append(z["global_view"]); locals_.append(z["local_view"])
        n_points.append(int(z["n_points"])); n_transits.append(int(z["n_transits_seen"]))
    rows["n_points"] = n_points
    rows["n_transits_seen"] = n_transits
    rows.to_parquet(DATASET_PATH, index=False)
    np.savez_compressed(STACK_PATH, kepoi_name=np.array(names),
                        global_view=np.stack(globals_), local_view=np.stack(locals_))
    print(f"assembled {len(rows)} KOIs -> {DATASET_PATH} and {STACK_PATH}")
    print(rows.groupby(["split", "koi_disposition"]).size().unstack(fill_value=0).to_string())
    print(f"stars per split: {rows.groupby('split')['kepid'].nunique().to_dict()}")


if __name__ == "__main__":
    sample = load_sample()
    if "--assemble-only" not in sys.argv:
        process_pending(sample)
    assemble(sample)
