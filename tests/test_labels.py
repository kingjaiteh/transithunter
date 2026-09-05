import pytest

from transithunter import config
from transithunter.data.labels import COLUMNS, build_query, feature_frame, load_labels, training_rows
from transithunter.features import ALLOWED_FEATURES, is_forbidden

pytestmark = pytest.mark.skipif(not config.LABELS_PATH.exists(), reason="KOI table not fetched yet")


def test_query_names_every_column():
    q = build_query()
    assert all(c in q for c in COLUMNS)


def test_training_rows_are_binary_and_exclude_candidates():
    rows = training_rows(load_labels())
    assert set(rows["label"].unique()) == {0, 1}
    assert config.HELDOUT_LABEL not in set(rows["koi_disposition"])


def test_feature_frame_only_has_allowed_columns():
    feats = feature_frame(load_labels())
    assert tuple(feats.columns) == ALLOWED_FEATURES
    assert not any(is_forbidden(c) for c in feats.columns)
