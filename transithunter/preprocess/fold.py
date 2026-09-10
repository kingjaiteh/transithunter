"""Detrend a stitched light curve and phase-fold it at a known ephemeris.

Phase 0 needs flatten and fold. The global/local binning (AstroNet views)
comes in Phase 1 and will build on `fold_at`.
"""

from __future__ import annotations

import astropy.units as u
import lightkurve as lk
import numpy as np

# Kepler long cadence is 29.4 minutes. Keep the flattening window well wider
# than the transit so the dip itself is not smoothed away.
CADENCE_DAYS = 29.4 / 60 / 24
WINDOW_MULTIPLIER = 3.0

# Eclipsing binaries can eclipse for hours on a sub-day orbit, and three times
# that duration is then wider than the orbit itself. Masking every cadence
# leaves the trend fit nothing to fit, so never mask more than half a cycle.
MAX_MASK_DUTY = 0.5


def mask_width_days(period: float, duration_hours: float) -> float:
    """Width of the protected window around each transit, capped at half a cycle."""
    return min(WINDOW_MULTIPLIER * duration_hours / 24, MAX_MASK_DUTY * period)


def transit_mask(lc: lk.LightCurve, period: float, t0: float, duration_hours: float) -> np.ndarray:
    """True where a point falls inside a transit, with a small margin."""
    width = mask_width_days(period, duration_hours)
    return lc.create_transit_mask(period=period, transit_time=t0, duration=width * u.day)


def flatten(lc: lk.LightCurve, period: float, t0: float, duration_hours: float) -> lk.LightCurve:
    """Remove stellar variability with a Savitzky-Golay filter, protecting transits.

    Without the mask the filter would partly fit the dip and shrink it.
    """
    window_days = max(3 * WINDOW_MULTIPLIER * duration_hours / 24, 0.75)
    window_length = int(window_days / CADENCE_DAYS)
    if window_length % 2 == 0:
        window_length += 1
    mask = transit_mask(lc, period, t0, duration_hours)
    return lc.flatten(window_length=window_length, mask=mask).remove_outliers(sigma_upper=3, sigma_lower=20)


def fold_at(lc: lk.LightCurve, period: float, t0: float) -> lk.FoldedLightCurve:
    """Stack every orbit on top of each other. Transit sits at phase 0."""
    return lc.fold(period=period, epoch_time=t0, normalize_phase=True)


def prepare(lc: lk.LightCurve, period: float, t0: float, duration_hours: float) -> lk.FoldedLightCurve:
    return fold_at(flatten(lc, period, t0, duration_hours), period, t0)
