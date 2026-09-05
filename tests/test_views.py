import lightkurve as lk
import numpy as np
import pandas as pd

from transithunter import config
from transithunter.preprocess.views import bin_median, make_views, normalise

CADENCE = 29.4 / 60 / 24


def box_transit(time, period, t0, duration_hours, depth):
    phase = ((time - t0) / period + 0.5) % 1 - 0.5
    return np.where(np.abs(phase * period) < duration_hours / 24 / 2, 1 - depth, 1.0)


def synthetic_lc(days=400, period=10.0, t0=3.0, duration_hours=4.0, depth=0.01, seed=1):
    rng = np.random.default_rng(seed)
    time = np.arange(0, days, CADENCE)
    flux = box_transit(time, period, t0, duration_hours, depth)
    flux *= 1 + 0.002 * np.sin(2 * np.pi * time / 37)   # slow stellar variability
    flux += rng.normal(0, 0.0005, len(time))
    return lk.LightCurve(time=time, flux=flux, flux_err=np.full(len(time), 0.0005))


def koi_row(period=10.0, t0=3.0, duration_hours=4.0, name="K99999.01"):
    return pd.Series({"kepid": 99999, "kepoi_name": name, "koi_period": period,
                      "koi_time0bk": t0, "koi_duration": duration_hours})


def test_views_have_expected_shape_and_centre():
    v = make_views(synthetic_lc(), koi_row())
    assert v.global_view.shape == (config.GLOBAL_BINS,)
    assert v.local_view.shape == (config.LOCAL_BINS,)
    # the box bottom is flat, so the minimum can sit anywhere inside the transit
    half_width_global = 4.0 / 24 / 10.0 * config.GLOBAL_BINS / 2
    assert abs(v.global_view.argmin() - config.GLOBAL_BINS // 2) <= half_width_global + 2
    assert abs(v.local_view.argmin() - config.LOCAL_BINS // 2) <= config.LOCAL_BINS / 8 + 2
    assert np.isclose(v.local_view.min(), -1.0)
    assert abs(np.median(v.local_view)) < 0.05


def test_transit_count_matches_injected_signal():
    v = make_views(synthetic_lc(days=400, period=10.0, t0=3.0), koi_row())
    assert v.n_transits_seen == 40


def test_sibling_transits_are_masked_out():
    lc = synthetic_lc(depth=0.01)
    # Half the main period, so the sibling lands on the same two phases in
    # every orbit and survives median binning. Median binning alone already
    # removes siblings whose periods do not line up.
    sib_period, sib_t0, sib_dur = 5.0, 5.5, 3.0
    lc.flux = lc.flux * box_transit(lc.time.value, sib_period, sib_t0, sib_dur, 0.05)
    koi = koi_row()
    siblings = pd.DataFrame([koi_row(sib_period, sib_t0, sib_dur, "K99999.02")])

    with_mask = make_views(lc, koi, siblings)
    without = make_views(lc, koi, None)
    # the sibling's deeper dips pollute the unmasked global view away from centre
    centre = config.GLOBAL_BINS // 2
    off_centre = np.r_[0:centre - 100, centre + 100:config.GLOBAL_BINS]
    assert without.global_view[off_centre].min() < -0.5
    assert with_mask.global_view[off_centre].min() > -0.5


def test_bin_median_fills_empty_bins():
    phase = np.array([-0.4, -0.3, 0.3, 0.4])
    flux = np.array([1.0, 1.0, 3.0, 3.0])
    view = bin_median(phase, flux, -0.5, 0.5, 10)
    assert not np.isnan(view).any()
    assert view[0] == 1.0 and view[-1] == 3.0 and 1.0 < view[5] < 3.0


def test_normalise_flat_view_does_not_divide_by_zero():
    out = normalise(np.ones(50))
    assert np.all(out == 0)
