r"""Phase 0 acceptance: 20 downloads, measured size per target, two folded plots.

Run: uv run python scripts/phase0_spike.py
Writes to D:\transithunter-data\artifacts\phase0\ (or $TRANSITHUNTER_DATA_DIR).
"""

from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from transithunter import config
from transithunter.data.fetch import fetch_stitched
from transithunter.data.labels import load_labels, training_rows
from transithunter.preprocess.fold import prepare

OUT = config.ARTIFACTS_DIR / "phase0"
OUT.mkdir(parents=True, exist_ok=True)
N_PER_CLASS = 10
KEPLER_10 = 11904151
KEPLER_10_B = "K00072.01"


def pick_targets(labels: pd.DataFrame) -> pd.DataFrame:
    rows = training_rows(labels)
    rows = rows[rows["koi_period"].notna() & rows["koi_time0bk"].notna() & rows["koi_duration"].notna()]
    rng = np.random.default_rng(config.RANDOM_SEED)

    # Hero examples for the plots: Kepler-10 b, and the loudest eclipsing-binary
    # false positive (stellar eclipse flag set, not-transit-like flag clear).
    planet = rows[rows["kepoi_name"] == KEPLER_10_B]
    fps = rows[(rows["label"] == 0) & (rows["koi_fpflag_ss"] == 1) & (rows["koi_fpflag_nt"] == 0)
               & (rows["koi_period"] < 20)]
    eclipsing_binary = fps.sort_values("koi_model_snr", ascending=False).head(1)

    def sample(label: int, n: int, exclude: set[int]) -> pd.DataFrame:
        pool = rows[(rows["label"] == label) & ~rows["kepid"].isin(exclude)]
        return pool.iloc[rng.choice(len(pool), size=n, replace=False)]

    chosen = pd.concat([
        planet, eclipsing_binary,
        sample(1, N_PER_CLASS - 1, {KEPLER_10}),
        sample(0, N_PER_CLASS - 1, set(eclipsing_binary["kepid"])),
    ])
    return chosen.drop_duplicates("kepid").reset_index(drop=True)


def plot_folded(folded, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(folded.phase.value, folded.flux.value, s=1, alpha=0.3, color="0.4", label="cadences")
    binned = folded.bin(bins=200)
    ax.plot(binned.phase.value, binned.flux.value, color="C3", lw=1.5, label="binned")
    ax.set_xlim(-0.25, 0.25)
    lo = np.nanpercentile(folded.flux.value, 0.5)
    ax.set_ylim(lo - 0.2 * (1 - lo), 1 + 0.5 * (1 - lo) + 0.002)
    ax.set_xlabel("phase (fraction of orbit)")
    ax.set_ylabel("relative flux")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    labels = load_labels()
    targets = pick_targets(labels)
    print(f"{len(targets)} targets: {targets['label'].sum()} planets, {(targets['label'] == 0).sum()} false positives")

    records = []
    for i, row in targets.iterrows():
        try:
            r = fetch_stitched(int(row["kepid"]))
        except Exception as exc:  # noqa: BLE001  - record and keep going
            print(f"  {row['kepoi_name']}: FAILED {exc}")
            records.append({"kepoi_name": row["kepoi_name"], "kepid": int(row["kepid"]), "error": str(exc)})
            continue
        rec = {
            "kepoi_name": row["kepoi_name"], "kepid": r.kepid, "label": int(row["label"]),
            "disposition": row["koi_disposition"], "quarters": r.n_files, "points": r.n_points,
            "megabytes": round(r.megabytes, 2), "seconds": round(r.seconds, 1),
        }
        records.append(rec)
        print(f"  {row['kepoi_name']:<12} {row['koi_disposition']:<15} {r.n_files:>2} q  {r.megabytes:5.1f} MB  {r.seconds:4.0f} s")

        if i < 2:  # the two hero examples
            folded = prepare(r.lc, row["koi_period"], row["koi_time0bk"], row["koi_duration"])
            name = row["kepler_name"] if isinstance(row["kepler_name"], str) else row["kepoi_name"]
            title = f"{name} ({row['koi_disposition']}), P = {row['koi_period']:.3f} d, depth {row['koi_depth']:.0f} ppm"
            fname = OUT / f"folded_{'planet' if row['label'] == 1 else 'false_positive'}.png"
            plot_folded(folded, title, fname)
            print(f"    saved {fname}")
        time.sleep(1)  # courteous to MAST

    df = pd.DataFrame(records)
    ok = df[df["megabytes"].notna()] if "megabytes" in df else df.iloc[0:0]
    summary = {
        "n_targets": int(len(df)),
        "n_failed": int(len(df) - len(ok)),
        "mb_per_target_mean": round(float(ok["megabytes"].mean()), 2),
        "mb_per_target_max": round(float(ok["megabytes"].max()), 2),
        "seconds_per_target_mean": round(float(ok["seconds"].mean()), 1),
        "quarters_mean": round(float(ok["quarters"].mean()), 1),
        "raw_cache_total_mb": round(sum(f.stat().st_size for f in config.RAW_DIR.rglob("*") if f.is_file()) / 1e6, 1),
    }
    df.to_csv(OUT / "downloads.csv", index=False)
    (OUT / "sizing.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
