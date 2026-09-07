"""Run the whole pipeline for one star on demand: fetch, detrend, fold, classify.

This is what the API calls. It reuses the exact preprocessing the training set
went through (`preprocess.views.make_views`) so serving and training cannot
drift apart. Models are loaded from the export directory that the training
scripts write; MLflow is not needed at serving time.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd

from transithunter import config
from transithunter.data.fetch import fetch_stitched
from transithunter.data.labels import load_labels
from transithunter.features import ALLOWED_FEATURES
from transithunter.preprocess.views import load_cached, make_views, siblings_of

MODELS_DIR = config.ARTIFACTS_DIR / "models"
STAGES = ("ephemeris", "fetch", "fold", "classify")


@dataclass
class Ephemeris:
    period: float          # days
    t0: float              # BKJD
    duration_hours: float
    source: str            # "catalogue" or "bls"
    kepoi_name: str | None = None
    depth_ppm: float | None = None


@dataclass
class VetResult:
    kepid: int
    ephemeris: Ephemeris
    verdict: str                          # "PLANET" or "FALSE POSITIVE"
    probability: float                    # CNN probability of PLANET
    threshold: float
    baseline_probability: float | None    # GBM on catalogue features, catalogue KOIs only
    catalogue_disposition: str | None     # ground truth for display, never a model input
    global_view: list[float]
    local_view: list[float]
    n_points: int
    n_transits_seen: int
    seconds: dict[str, float] = field(default_factory=dict)
    cached_light_curve: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class Vetter:
    def __init__(self, models_dir: Path = MODELS_DIR, cnn_name: str = "cnn_views_v1",
                 baseline_name: str = "baseline_v1", device: str | None = None):
        import torch

        from transithunter.models.cnn import CNNConfig, TransitCNN

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        card = json.loads((models_dir / cnn_name / "model_card.json").read_text())
        cfg = CNNConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in card["config"].items()})
        self.cnn = TransitCNN(cfg)
        self.cnn.load_state_dict(torch.load(models_dir / cnn_name / "weights.pt", map_location="cpu"))
        self.cnn.to(self.device).eval()
        self.cnn_card = card
        self.threshold = float(card["threshold"])
        self.scaler = card.get("scaler")

        baseline_dir = models_dir / baseline_name
        self.baseline = joblib.load(baseline_dir / "model.joblib") if baseline_dir.exists() else None
        self.baseline_card = (json.loads((baseline_dir / "model_card.json").read_text())
                              if self.baseline is not None else None)
        self.labels = load_labels()

    # -- ephemerides ---------------------------------------------------------

    def kois_on(self, kepid: int) -> pd.DataFrame:
        return self.labels[self.labels["kepid"] == kepid].sort_values("kepoi_name")

    def catalogue_ephemeris(self, kepid: int, kepoi_name: str | None) -> tuple[Ephemeris, pd.Series] | None:
        kois = self.kois_on(kepid)
        kois = kois[kois[["koi_period", "koi_time0bk", "koi_duration"]].notna().all(axis=1)]
        if kepoi_name is not None:
            kois = kois[kois["kepoi_name"] == kepoi_name]
        if kois.empty:
            return None
        row = kois.iloc[0]
        eph = Ephemeris(float(row["koi_period"]), float(row["koi_time0bk"]), float(row["koi_duration"]),
                        "catalogue", str(row["kepoi_name"]),
                        None if pd.isna(row["koi_depth"]) else float(row["koi_depth"]))
        return eph, row

    @staticmethod
    def bls_ephemeris(lc, min_period: float = 0.5, max_period: float = 50.0, n_periods: int = 20000) -> Ephemeris:
        """Box Least Squares search on a flattened curve for stars with no catalogue entry."""
        import astropy.units as u

        flat = lc.flatten(window_length=901).remove_outliers(sigma_upper=3, sigma_lower=20)
        periods = np.geomspace(min_period, max_period, n_periods)
        pg = flat.to_periodogram(method="bls", period=periods,
                                 duration=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4])
        return Ephemeris(float(pg.period_at_max_power.to(u.day).value),
                         float(pg.transit_time_at_max_power.value),
                         float(pg.duration_at_max_power.to(u.hour).value),
                         "bls", None, float(pg.depth_at_max_power) * 1e6)

    # -- pipeline ------------------------------------------------------------

    def vet(self, kepid: int, kepoi_name: str | None = None,
            progress: Callable[[str], None] | None = None) -> VetResult:
        import torch

        seconds: dict[str, float] = {}
        tick = lambda stage: progress(stage) if progress else None  # noqa: E731

        tick("ephemeris")
        t = time.time()
        found = self.catalogue_ephemeris(kepid, kepoi_name)
        seconds["ephemeris"] = time.time() - t

        tick("fetch")
        t = time.time()
        # The local FITS cache first: lightkurve's search alone costs about
        # 10 s of MAST round trips even when every quarter is already on disk.
        try:
            lc, cached = load_cached(kepid), True
        except FileNotFoundError:
            lc, cached = fetch_stitched(kepid).lc, False
        seconds["fetch"] = time.time() - t

        if found is None:
            if kepoi_name is not None:
                raise LookupError(f"{kepoi_name} is not a KOI on KIC {kepid}")
            t = time.time()
            eph, row = self.bls_ephemeris(lc), None
            seconds["ephemeris"] += time.time() - t
            siblings = None
        else:
            eph, row = found
            siblings = siblings_of(row, self.labels)

        tick("fold")
        t = time.time()
        koi = pd.Series({"koi_period": eph.period, "koi_time0bk": eph.t0, "koi_duration": eph.duration_hours})
        views = make_views(lc, koi, siblings)
        seconds["fold"] = time.time() - t

        tick("classify")
        t = time.time()
        g = torch.from_numpy(views.global_view.astype(np.float32))[None].to(self.device)
        l = torch.from_numpy(views.local_view.astype(np.float32))[None].to(self.device)
        aux = None
        if self.scaler and row is not None:
            x = row[list(ALLOWED_FEATURES)].to_numpy(dtype=np.float32)
            z = (x - np.array(self.scaler["mean"])) / np.array(self.scaler["std"])
            aux = torch.from_numpy(np.nan_to_num(z, nan=0.0).astype(np.float32))[None].to(self.device)
        with torch.no_grad():
            prob = float(torch.sigmoid(self.cnn(g, l, aux)).item())
        baseline_prob = None
        if self.baseline is not None and row is not None:
            x = row[list(ALLOWED_FEATURES)].to_numpy(dtype=np.float32)[None]
            baseline_prob = float(self.baseline.predict_proba(x)[0, 1])
        seconds["classify"] = time.time() - t

        return VetResult(
            kepid=kepid, ephemeris=eph,
            verdict="PLANET" if prob >= self.threshold else "FALSE POSITIVE",
            probability=prob, threshold=self.threshold, baseline_probability=baseline_prob,
            catalogue_disposition=None if row is None else str(row["koi_disposition"]),
            global_view=views.global_view.astype(float).tolist(),
            local_view=views.local_view.astype(float).tolist(),
            n_points=views.n_points, n_transits_seen=views.n_transits_seen,
            seconds=seconds, cached_light_curve=cached,
        )


if __name__ == "__main__":
    import sys

    kepid = int(sys.argv[1]) if len(sys.argv) > 1 else 11904151  # Kepler-10
    r = Vetter().vet(kepid, progress=lambda s: print(f"  {s}...", flush=True))
    print(f"KIC {kepid} {r.ephemeris.kepoi_name or 'BLS'}: {r.verdict} p={r.probability:.3f} "
          f"(threshold {r.threshold:.2f}, baseline {r.baseline_probability}, "
          f"catalogue {r.catalogue_disposition})")
    print({k: round(v, 1) for k, v in r.seconds.items()})
