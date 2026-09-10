"""Turn one KOI into AstroNet's two inputs: a global view and a local view.

Global view: the whole folded orbit in 2001 bins. Local view: a zoom on the
transit, four transit durations wide, in 201 bins. Both are normalised so the
out-of-transit level is 0 and the deepest point is -1 (Shallue & Vanderburg
2018). Reads FITS straight from the lightkurve cache, so no network is needed
once the downloader has run.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightkurve as lk
import numpy as np
import pandas as pd

from transithunter import config
from transithunter.data.fetch import MAST_DIR
from transithunter.preprocess.fold import flatten, transit_mask

LOCAL_WIDTH_DURATIONS = 4.0  # total width of the local view, in transit durations


@dataclass
class Views:
    global_view: np.ndarray  # (GLOBAL_BINS,)
    local_view: np.ndarray   # (LOCAL_BINS,)
    n_points: int            # cadences that survived cleaning
    n_transits_seen: int     # transits with at least one in-window point


def load_cached(kepid: int) -> lk.LightCurve:
    """Stitch every cached quarter for a star without touching MAST."""
    files = sorted(f for d in MAST_DIR.glob(f"kplr{kepid:09d}*") for f in d.rglob("*.fits"))
    if not files:
        raise FileNotFoundError(f"KIC {kepid} is not in the cache at {MAST_DIR}")
    curves = [lk.read(str(f), flux_column="pdcsap_flux", quality_bitmask="default") for f in files]
    return lk.LightCurveCollection(curves).stitch().remove_nans()


def bin_median(phase: np.ndarray, flux: np.ndarray, lo: float, hi: float, bins: int) -> np.ndarray:
    """Median flux per equal-width phase bin. Empty bins are linearly filled."""
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.digitize(phase, edges) - 1, 0, bins - 1)
    inside = (phase >= lo) & (phase < hi)
    view = pd.Series(flux[inside]).groupby(idx[inside]).median().reindex(range(bins)).to_numpy()
    empty = np.isnan(view)
    if empty.all():
        raise ValueError("no points inside the view window")
    if empty.any():
        centres = np.arange(bins)
        view[empty] = np.interp(centres[empty], centres[~empty], view[~empty])
    return view


def normalise(view: np.ndarray) -> np.ndarray:
    """Median to 0, minimum to -1. Falls back to unit scale on a flat view."""
    view = view - np.median(view)
    depth = -view.min()
    return view / depth if depth > 0 else view


def make_views(lc: lk.LightCurve, koi: pd.Series, siblings: pd.DataFrame | None = None) -> Views:
    """`koi` is one row of the label table; `siblings` are the other KOIs on the same star."""
    period, t0, duration = float(koi["koi_period"]), float(koi["koi_time0bk"]), float(koi["koi_duration"])

    # Flatten around every known transit on the star so none of them is eroded.
    flat = flatten(lc, period, t0, duration)
    if siblings is not None and len(siblings):
        keep = np.ones(len(flat), dtype=bool)
        for _, sib in siblings.iterrows():
            keep &= ~transit_mask(flat, float(sib["koi_period"]), float(sib["koi_time0bk"]),
                                  float(sib["koi_duration"]))
        flat = flat[keep]

    # Fold by hand so phase, flux and time stay in the same order.
    time_days = flat.time.value.astype(float)
    cycles = (time_days - t0) / period
    phase = (cycles + 0.5) % 1.0 - 0.5
    flux = flat.flux.value.astype(float)

    global_view = normalise(bin_median(phase, flux, -0.5, 0.5, config.GLOBAL_BINS))
    half = LOCAL_WIDTH_DURATIONS / 2 * duration / 24 / period
    half = min(half, 0.5)
    local_view = normalise(bin_median(phase, flux, -half, half, config.LOCAL_BINS))

    in_window = np.abs(phase) < half
    n_transits = int(np.unique(np.round(cycles[in_window])).size)
    return Views(global_view, local_view, n_points=len(flat), n_transits_seen=n_transits)


def siblings_of(koi: pd.Series, table: pd.DataFrame) -> pd.DataFrame:
    same_star = table["kepid"] == koi["kepid"]
    return table[same_star & (table["kepoi_name"] != koi["kepoi_name"])]
