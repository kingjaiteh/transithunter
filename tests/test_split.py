import numpy as np
import pandas as pd

from transithunter.preprocess.split import FRACTIONS, assert_no_star_leakage, assign_splits


def fake_table(n_stars=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for kepid in range(n_stars):
        n_koi = rng.choice([1, 1, 1, 2, 3])
        label = int(rng.random() < 0.4)
        for j in range(n_koi):
            rows.append({"kepid": kepid, "kepoi_name": f"K{kepid:05d}.{j + 1:02d}", "label": label})
    return pd.DataFrame(rows)


def test_every_koi_on_a_star_shares_a_split():
    table = fake_table()
    table["split"] = assign_splits(table)
    assert table["split"].notna().all()
    assert_no_star_leakage(table)


def test_split_fractions_by_star_are_close():
    table = fake_table()
    table["split"] = assign_splits(table)
    per_star = table.drop_duplicates("kepid")["split"].value_counts(normalize=True)
    for name, frac in FRACTIONS.items():
        assert abs(per_star[name] - frac) < 0.03


def test_leakage_check_fires():
    table = fake_table(n_stars=5)
    table["split"] = "train"
    table.loc[table.index[-1], "split"] = "test"  # last KOI of last star
    if (table["kepid"] == table["kepid"].iloc[-1]).sum() == 1:
        table.loc[table.index[-2], "kepid"] = table["kepid"].iloc[-1]
    try:
        assert_no_star_leakage(table)
    except AssertionError:
        return
    raise AssertionError("leak not detected")


def test_split_is_deterministic():
    a = assign_splits(fake_table())
    b = assign_splits(fake_table())
    assert a.equals(b)
