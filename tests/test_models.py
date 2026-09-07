import numpy as np
import pytest

from transithunter import config
from transithunter.eval.metrics import choose_threshold, evaluate
from transithunter.models.baseline import make_baseline


def test_choose_threshold_separates_clean_scores():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = choose_threshold(y, s)
    assert 0.3 < t <= 0.7
    m = evaluate(y, s, t)
    assert m["pr_auc"] == 1.0 and m["precision"] == 1.0 and m["recall"] == 1.0
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (3, 3, 0, 0)


def test_baseline_learns_with_nans():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 11)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    X[rng.random(X.shape) < 0.1] = np.nan
    model = make_baseline(max_iter=50).fit(X[:300], y[:300])
    score = model.predict_proba(X[300:])[:, 1]
    assert evaluate(y[300:], score, 0.5)["roc_auc"] > 0.85


def test_cnn_forward_shapes():
    torch = pytest.importorskip("torch")
    from transithunter.models.cnn import CNNConfig, TransitCNN

    model = TransitCNN(CNNConfig())
    g = torch.zeros(4, config.GLOBAL_BINS)
    l = torch.zeros(4, config.LOCAL_BINS)
    assert model(g, l).shape == (4,)
    assert 0 < model.n_parameters() < 5_000_000

    aux_model = TransitCNN(CNNConfig(n_aux=11))
    assert aux_model(g, l, torch.zeros(4, 11)).shape == (4,)


def test_cnn_augment_is_a_reflection():
    torch = pytest.importorskip("torch")
    from transithunter.training.train_cnn import augment

    torch.manual_seed(0)
    g = torch.arange(20.0).repeat(8, 1)
    l = torch.arange(5.0).repeat(8, 1)
    g2, l2 = augment(g, l)
    for row_g, row_l, orig_g, orig_l in zip(g2, l2, g, l):
        flipped = torch.equal(row_g, orig_g.flip(0))
        assert flipped or torch.equal(row_g, orig_g)
        assert torch.equal(row_l, orig_l.flip(0) if flipped else orig_l)
