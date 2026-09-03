import pytest

from transithunter.features import ALLOWED_FEATURES, assert_clean, is_forbidden


def test_allowlist_contains_no_forbidden_columns():
    assert_clean(ALLOWED_FEATURES)


@pytest.mark.parametrize(
    "col", ["koi_score", "koi_disposition", "koi_fpflag_nt", "koi_fpflag_ss", "kepler_name"]
)
def test_known_leaks_are_forbidden(col):
    assert is_forbidden(col)


def test_assert_clean_raises_on_leak():
    with pytest.raises(ValueError):
        assert_clean(["koi_period", "koi_score"])
