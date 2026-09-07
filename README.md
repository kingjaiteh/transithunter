# TransitHunter

Finds planet-like dips in Kepler light curves and classifies each candidate as
PLANET or FALSE POSITIVE, trained against NASA's Kepler Objects of Interest
labels. Baseline gradient boosting on transit and stellar features, then an
AstroNet-style 1D CNN over phase-folded views, tracked in MLflow, served
through FastAPI with a React front end.

Status: Phase 2 in progress. The dataset build (3,453 KOIs on 2,693 stars,
about 19 GB of light curves) is still downloading. Both trainers, the
inference path, the API and the UI run end to end on the partial dataset.
Next is training on the full dataset and publishing the metrics table here.

## How it works

1. `transithunter/data/labels.py` pulls the KOI table from the Exoplanet
   Archive over TAP. Model inputs pass through an explicit allowlist in
   `transithunter/features.py`, so vetting columns cannot leak into training.
2. `transithunter/data/download.py` fetches every Kepler quarter for each
   sampled star through lightkurve. It keeps a manifest, so it can be killed
   and restarted at any point.
3. `transithunter/preprocess/views.py` detrends around the transit, masks
   sibling planets on the same star, folds at the catalogue period and bins
   into a 2001-bin global view and a 201-bin local view.
4. `scripts/build_dataset.py` writes the views and a feature table, split
   70/15/15 by star so no light curve appears on both sides of the test
   boundary. `transithunter/preprocess/validate.py` runs a Great Expectations
   suite plus tensor checks and exits non-zero if anything fails.
5. `transithunter/training/train_baseline.py` fits histogram gradient boosting
   on the eleven allowlisted catalogue features.
   `transithunter/training/train_cnn.py` trains the two-branch CNN on the
   local GPU with early stopping on validation PR-AUC. Both log to MLflow and
   export weights plus a model card.
6. `transithunter/inference.py` runs fetch, detrend, fold and classify for one
   star on demand, using the same preprocessing code as training. Stars with
   no catalogue entry get a Box Least Squares period search.
7. `api/main.py` exposes that as a job: POST `/api/vet/{kepid}` returns a job
   id, GET `/api/jobs/{id}` reports the stage and result. Results are cached
   on disk, so repeat requests return at once. The built React app in `web/`
   is served from the same process.

## Setup

```powershell
cd transithunter
uv sync --extra train --extra validate --extra api
uv run pytest
```

Light curves and built datasets go under `TRANSITHUNTER_DATA_DIR` (default
`D:\transithunter-data`). On Windows, uv installs the CUDA build of torch from
the PyTorch index declared in `pyproject.toml`.

Build the dataset, then train:

```powershell
uv run python -m transithunter.data.download
uv run python scripts/build_dataset.py
uv run python -m transithunter.preprocess.validate
uv run python -m transithunter.training.train_baseline
uv run python -m transithunter.training.train_cnn
```

Run the demo:

```powershell
cd web; npm install; npm run build; cd ..
uv run uvicorn api.main:app --port 8000
```

Then open http://localhost:8000. For UI development, `npm run dev` in `web/`
proxies `/api` to port 8000.

## Design decisions

- Split by star, not by KOI. One star can host several candidates cut from
  the same light curve, so a per-KOI split would leak.
- PR-AUC is the headline metric. The classes are 55/45 in the sample and far
  more unbalanced in the full catalogue, so accuracy would flatter a model.
- The threshold is chosen once on the validation PR curve and frozen. Test
  numbers are reported at that threshold.
- The CNN's input views are built by the same function at training and
  serving time. There is no second preprocessing path to drift.
- Models are exported as plain weights plus a JSON card. Serving does not
  import MLflow.
