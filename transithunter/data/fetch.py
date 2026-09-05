"""Download Kepler long-cadence light curves through lightkurve.

Importing this module points lightkurve's cache at the data drive. Import it
before calling any lightkurve function elsewhere.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import lightkurve as lk

from transithunter import config

config.configure_lightkurve()
MAST_DIR = config.LIGHTKURVE_CACHE_DIR / "mastDownload" / "Kepler"


@dataclass
class FetchResult:
    kepid: int
    n_files: int
    n_points: int
    bytes_on_disk: int
    seconds: float
    lc: lk.LightCurve

    @property
    def megabytes(self) -> float:
        return self.bytes_on_disk / 1e6


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def fetch_stitched(kepid: int, quality_bitmask: str = "default") -> FetchResult:
    """All Kepler quarters for one star, PDCSAP flux, stitched into one curve.

    lightkurve caches each quarter's FITS file, so a second call is disk only.
    """
    t0 = time.time()
    search = lk.search_lightcurve(
        f"KIC {kepid}", mission="Kepler", cadence="long", author="Kepler"
    )
    if len(search) == 0:
        raise LookupError(f"no Kepler long-cadence light curves for KIC {kepid}")
    collection = search.download_all(
        flux_column="pdcsap_flux", quality_bitmask=quality_bitmask
    )
    lc = collection.stitch().remove_nans()
    seconds = time.time() - t0

    # lightkurve keeps one directory per target and quarter set under MAST_DIR.
    on_disk = sum(dir_size(p) for p in MAST_DIR.glob(f"kplr{kepid:09d}*"))
    return FetchResult(
        kepid=kepid,
        n_files=len(collection),
        n_points=len(lc),
        bytes_on_disk=on_disk,
        seconds=seconds,
        lc=lc,
    )


if __name__ == "__main__":
    import sys

    kepid = int(sys.argv[1]) if len(sys.argv) > 1 else 11904151  # Kepler-10
    r = fetch_stitched(kepid)
    print(f"KIC {kepid}: {r.n_files} quarters, {r.n_points} points, "
          f"{r.megabytes:.1f} MB on disk, {r.seconds:.0f} s")
    print(f"time span {r.lc.time.min().value:.1f} to {r.lc.time.max().value:.1f} BKJD")
