# TransitHunter

Finds planet-like dips in Kepler light curves and classifies each candidate as
PLANET or FALSE POSITIVE, trained against NASA's Kepler Objects of Interest
labels. Baseline gradient boosting on transit and stellar features, then an
AstroNet-style 1D CNN over phase-folded views, tracked in MLflow.

Status: scaffold only. Phase 0 (pull the KOI table, download 20 light curves,
fold one planet and one false positive, measure per-target size) is next. See
PLAN.md for scope and sequencing.

## Setup

```powershell
cd transithunter
uv sync
uv run pytest
```

Light curves are cached under the directory named by `TRANSITHUNTER_DATA_DIR`
(default `D:\transithunter-data`).
