# TransitHunter

Finds planet-like dips in Kepler light curves and classifies each candidate as
PLANET or FALSE POSITIVE, trained against NASA's Kepler Objects of Interest
labels. Baseline gradient boosting on transit and stellar features, then an
AstroNet-style 1D CNN over phase-folded views, tracked in MLflow.

Status: Phase 0 done. The KOI table is fetched and cached, 20 light curves
downloaded through lightkurve, and Kepler-10 b plus an eclipsing binary
phase-folded and plotted. Each star costs about 7 MB and 10 seconds. Next is
the dataset build: a resumable downloader, global and local views, and a
train/validation/test split by star.

## What is here

- `transithunter/data/labels.py` fetches the Kepler Objects of Interest table
  from the Exoplanet Archive and turns it into training rows. Model inputs
  pass through an explicit allowlist so vetting columns cannot leak.
- `transithunter/data/fetch.py` downloads and stitches all Kepler quarters for
  one star, caching on the data drive.
- `transithunter/preprocess/fold.py` detrends around the transit and folds at
  the catalogue period.
- `scripts/phase0_spike.py` reproduces the sizing measurement and the two plots.

## Setup

```powershell
cd transithunter
uv sync
uv run pytest
```

Light curves are cached under the directory named by `TRANSITHUNTER_DATA_DIR`
(default `D:\transithunter-data`).
