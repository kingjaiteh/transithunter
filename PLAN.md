# TransitHunter — Detecting and Vetting Exoplanet Transits in Kepler Light Curves

## PLAN.md — drop this in the repo root. Claude Code should treat this as the source of truth for scope and sequencing.

---

## 1. Elevator pitch

Given a star's brightness-over-time data (a light curve), find planet-like dips and classify each candidate as PLANET or FALSE POSITIVE, trained against NASA's ground-truth Kepler labels. Real supervised ML with measurable precision/recall, served through FastAPI + React with interactive phase-folded plots.

## 2. Goals

- End-to-end pipeline: fetch light curve -> detrend -> BLS period search -> phase-fold -> classify -> explain.
- A trained classifier that beats a stated baseline, with honest metrics (PR-AUC, confusion matrix) on a held-out set.
- Interactive demo: enter a KIC ID (Kepler star identifier), watch the pipeline run, see the folded transit and the verdict.
- Experiment tracking with MLflow so every model version is reproducible.

## 3. Non-goals (v1)

- No claim of discovering new planets. This reproduces a known vetting task on labeled data.
- No TESS in v1 (stretch: TESS TOIs generalization test).
- No LLM anywhere in the core pipeline. Optional stretch only.

## 4. Tech stack (each item earns its place)

- **lightkurve** — Python package for Kepler/K2/TESS light curves; wraps astroquery.mast, handles download, caching, detrending, folding.
- **astroquery / Exoplanet Archive TAP** — labels come from the Kepler cumulative KOI table via IPAC's Table Access Protocol (ADQL, an SQL dialect for astronomy). Free, no key.
- **Box Least Squares (BLS, via astropy.timeseries)** — slides a box-shaped dip template across many trial periods to find repeating transits; produces period, depth, duration.
- **scikit-learn** — baseline model (gradient boosting on BLS + stellar features). Always beat a baseline before deep learning.
- **PyTorch 1D CNN (AstroNet-style)** — two-branch convolutional net over "global view" (whole folded curve, ~2001 bins) and "local view" (zoom on transit, ~201 bins), per Shallue & Vanderburg 2018. Trains in minutes-to-hours on a Kaggle T4.
- **Hugging Face (stretch)** — fine-tune a pretrained light-curve transformer (Astromer 2) or use its embeddings as features; compare against the CNN. This is the "current research" talking point.
- **MLflow** — experiment tracking: params, metrics, artifacts per run. Natural fit given your Databricks background.
- **Great Expectations** — validation on the assembled training table.
- **FastAPI + React (Vite, TypeScript)** — serving and UI; plotly or visx for interactive folded light curves.
- **Kaggle Notebooks** — free 30 GPU hrs/week (T4/P100) for training; export model weights as artifacts. Local CPU for everything else.
- **Docker + HF Spaces** — demo hosting (CPU inference is fine; the CNN is tiny).

## 5. Data

- **Labels**: KOI cumulative table (`cumulative`) via Exoplanet Archive TAP. ~9,500 Kepler Objects of Interest with `koi_disposition` in {CONFIRMED, CANDIDATE, FALSE POSITIVE}. Training uses CONFIRMED vs FALSE POSITIVE; CANDIDATE held aside.
- **Light curves**: Kepler long-cadence PDCSAP flux from MAST via lightkurve. PDCSAP = pipeline-detrended flux (instrument systematics removed by NASA's pipeline).
- **Sizing rule**: Phase 0 measures real per-target download size, then pick a stratified sample (target ~2,000-3,000 KOIs, balanced classes) that fits a ~20-30 GB local disk budget. Prefer stitched quarters; consider DV time series files (smaller, pre-stitched) if raw quarters are too heavy.
- All data is US Government work / public domain. MAST asks for courteous request rates; lightkurve caching handles this.

## 6. Architecture

```
[TAP query] -> koi_labels.parquet
[lightkurve fetch] -> raw .fits cache -> [preprocess: flatten, fold at koi_period,
    bin to global(2001)/local(201) views] -> tensors + feature table -> [GE checks]

[Train] (Kaggle GPU): baseline GBM -> CNN -> (stretch) Astromer embeddings
    all runs logged to MLflow -> best model exported to models/

[Serve] FastAPI: /vet/{kic_id} -> fetch -> preprocess -> BLS (if no known period)
    -> model -> {verdict, probability, period, depth, folded curve arrays}
[React] : input KIC ID, pipeline status steps, interactive folded plot, metrics page
```

## 7. Phases and acceptance criteria

**Phase 0 — Spike (2-3 days)**
- Pull the KOI table via TAP; download 20 light curves with lightkurve; fold one known planet (e.g., Kepler-10 b) and one known false positive; eyeball the difference.
- Accept: two clean folded plots saved; measured MB/target recorded; sample size chosen.

**Phase 1 — Dataset build (weeks 1-2)**
- Resumable downloader with cache; preprocessing to global/local views; GE suite (no NaN floods, label balance, period > 0); stratified train/val/test split BY STAR (no leakage of the same star across splits); dataset card in repo documenting choices.
- Accept: tensors + feature table for full sample; GE green; split leakage test passes.

**Phase 2 — Baseline then CNN (weeks 2-3)**
- GBM baseline on BLS/stellar features, logged to MLflow. Then AstroNet-style CNN trained on Kaggle; early stopping; threshold chosen on validation PR curve.
- Accept: CNN beats baseline PR-AUC on test set; confusion matrix + PR curve artifacts in MLflow; README metrics table with the baseline shown honestly.

**Phase 3 — Serving + UI (weeks 4-5)**
- FastAPI /vet endpoint running full pipeline on demand (cache aggressively; first-time fetch for an unseen star can take ~a minute — show pipeline progress states in UI). React app: KIC input, staged progress, interactive folded curve, probability gauge, model card page.
- Accept: known planet and known FP both classified correctly end-to-end from the UI; p95 warm-request < 2s for cached stars.

**Phase 4 — Eval hardening + deploy (week 5-6)**
- Evaluate on the held-out CANDIDATE set and discuss (no ground truth — report score distribution); error analysis notebook on top-20 worst mistakes; Dockerize; deploy to HF Spaces.
- Accept: public demo live; error analysis written up; limitations section in README.

**Stretch goals**
- Astromer 2 fine-tune / embedding comparison vs CNN (the Hugging Face training story).
- Embed light curves and index in Pinecone: "find stars with similar transit signals" (shares your Pinecone project with MarsLens).
- TESS generalization test on TOI table.

## 8. Risks and honest notes

- Label noise: KOI dispositions come from an automated vetter plus humans; do not chase the last 2% of accuracy, and say so.
- Data volume is the biggest schedule risk; the Phase 0 sizing spike exists to kill it early.
- Class imbalance: FALSE POSITIVE outnumbers CONFIRMED; use stratified sampling and PR-AUC (not plain accuracy) everywhere.
- Do not train on features that leak the label (e.g., koi_score or disposition-derived columns). Keep an explicit allowlist of input features.

## 9. Repo layout

```
transithunter/
  data/          # TAP client, downloader, cache
  preprocess/    # flatten/fold/bin, GE suite, splits
  models/        # baseline, cnn, (stretch) astromer
  training/      # Kaggle-runnable training scripts, MLflow config
  api/           # FastAPI app
  web/           # React (Vite) app
  eval/          # metrics, error analysis notebooks
  docker/
  PLAN.md
```

## 10. Glossary (new tech explained)

- **Light curve**: brightness of a star measured over time.
- **Transit**: periodic small dip in brightness when a planet crosses its star.
- **Phase folding**: overlaying all orbits on top of each other at the detected period so the repeated dip stacks into one clear signal.
- **PDCSAP flux**: NASA-pipeline-cleaned brightness values (systematics removed).
- **BLS**: Box Least Squares, the classic transit-search algorithm.
- **Global/local view**: AstroNet's two inputs — the whole folded curve plus a zoom on the dip.
- **PR-AUC**: area under the precision-recall curve; the right headline metric when classes are imbalanced.
- **MLflow run**: one logged training attempt with its params, metrics, and saved model.
