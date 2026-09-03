"""Explicit allowlist of KOI-table columns a model may see.

PLAN.md section 8: never train on anything derived from the disposition.
`assert_clean` runs at dataset-build time and in tests, so a new column cannot
reach the feature table without being named here first.
"""

from __future__ import annotations

# Measured from the light curve or the stellar catalogue, not from vetting.
ALLOWED_FEATURES: tuple[str, ...] = (
    "koi_period",       # orbital period, days
    "koi_duration",     # transit duration, hours
    "koi_depth",        # transit depth, ppm
    "koi_prad",         # planet radius, Earth radii
    "koi_impact",       # impact parameter
    "koi_model_snr",    # transit signal-to-noise
    "koi_num_transits",
    "koi_steff",        # stellar effective temperature, K
    "koi_slogg",        # stellar surface gravity
    "koi_srad",         # stellar radius, solar radii
    "koi_kepmag",       # Kepler-band magnitude
)

# Anything that encodes the vetting verdict or was produced by the vetter.
FORBIDDEN_EXACT: frozenset[str] = frozenset({
    "koi_disposition",
    "koi_pdisposition",
    "koi_score",
    "koi_vet_stat",
    "koi_vet_date",
    "koi_comment",
    "kepler_name",      # only confirmed planets get a Kepler name
    "koi_tce_delivname",
})
FORBIDDEN_PREFIXES: tuple[str, ...] = ("koi_fpflag_",)


def is_forbidden(column: str) -> bool:
    return column in FORBIDDEN_EXACT or column.startswith(FORBIDDEN_PREFIXES)


def assert_clean(columns: list[str] | tuple[str, ...]) -> None:
    leaked = sorted(c for c in columns if is_forbidden(c))
    if leaked:
        raise ValueError(f"label-leaking columns in feature table: {leaked}")
