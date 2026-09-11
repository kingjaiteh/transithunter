"""FastAPI service around transithunter.inference.Vetter.

Run: uv run uvicorn api.main:app --reload --port 8000
Endpoints live under /api; the built React app in web/dist is served at /.

A first-time star takes about 10 to 60 seconds to fetch from MAST, so vetting
is a job: POST /vet/{kepid} returns a job id at once, GET /jobs/{id} reports
the current stage and, when done, the result. Finished results are cached in
memory and on disk so a repeat request for the same star returns immediately.
"""

from __future__ import annotations

import json
import threading
import traceback
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from transithunter import config
from transithunter.inference import STAGES, Vetter

CACHE_DIR = config.ARTIFACTS_DIR / "vet_cache"
EXAMPLES = [
    {"kepid": 11904151, "name": "Kepler-10", "note": "confirmed planet, 190 ppm transit"},
    {"kepid": 10666592, "name": "Kepler-2 (HAT-P-7)", "note": "confirmed hot Jupiter"},
    {"kepid": 7970760, "name": "KOI-6943", "note": "eclipsing binary false positive, 24 percent deep"},
    {"kepid": 6603043, "name": "KOI-368", "note": "false positive, 110 day period"},
]

WEB_DIST = config.PROJECT_ROOT / "web" / "dist"

# Finished jobs stay in memory so the UI can poll them after the fact. A long
# running demo would grow this forever, so keep only the most recent ones; the
# results themselves are on disk in CACHE_DIR either way.
MAX_JOBS = 512


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the models in the background so the first vet is not the one that pays."""
    threading.Thread(target=vetter, daemon=True).start()
    yield


app = FastAPI(title="TransitHunter", version="0.1", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])
api = APIRouter(prefix="/api")


@dataclass
class Job:
    id: str
    kepid: int
    kepoi_name: str | None
    stage: str = "queued"
    stages: list[str] = field(default_factory=lambda: list(STAGES))
    result: dict | None = None
    error: str | None = None


_jobs: OrderedDict[str, Job] = OrderedDict()
_lock = threading.Lock()
_vetter: Vetter | None = None


def remember(job: Job) -> None:
    with _lock:
        _jobs[job.id] = job
        _jobs.move_to_end(job.id)
        while len(_jobs) > MAX_JOBS:
            _jobs.popitem(last=False)


def vetter() -> Vetter:
    global _vetter
    if _vetter is None:
        _vetter = Vetter()
    return _vetter


def cache_key(kepid: int, kepoi_name: str | None) -> str:
    return f"{kepid}_{kepoi_name or 'auto'}"


def cached_result(key: str) -> dict | None:
    p = CACHE_DIR / f"{key}.json"
    return json.loads(p.read_text()) if p.exists() else None


def run_job(job: Job) -> None:
    def progress(stage: str) -> None:
        job.stage = stage

    try:
        result = vetter().vet(job.kepid, job.kepoi_name, progress).to_dict()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (CACHE_DIR / f"{cache_key(job.kepid, job.kepoi_name)}.json").write_text(json.dumps(result))
        job.result, job.stage = result, "done"
    except Exception as exc:  # noqa: BLE001
        job.error, job.stage = f"{type(exc).__name__}: {exc}", "error"
        traceback.print_exc()


def _finish_app() -> None:
    """Routes go under /api; the built React app, when present, is served at /."""
    app.include_router(api)
    if WEB_DIST.exists():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")


@api.get("/health")
def health() -> dict:
    return {"ok": True, "models_loaded": _vetter is not None, "device": str(_vetter.device) if _vetter else None}


@api.get("/examples")
def examples() -> list[dict]:
    return EXAMPLES


@api.get("/models")
def models() -> dict:
    v = vetter()
    return {"cnn": v.cnn_card, "baseline": v.baseline_card}


@api.get("/kois/{kepid}")
def kois(kepid: int) -> list[dict]:
    rows = vetter().kois_on(kepid)
    cols = ["kepoi_name", "kepler_name", "koi_disposition", "koi_period", "koi_depth", "koi_duration"]
    return json.loads(rows[cols].to_json(orient="records"))


@api.post("/vet/{kepid}")
def vet(kepid: int, koi: str | None = None, refresh: bool = False) -> dict:
    key = cache_key(kepid, koi)
    if not refresh and (hit := cached_result(key)) is not None:
        job = Job(id=key, kepid=kepid, kepoi_name=koi, stage="done", result=hit)
        remember(job)
        return {"job_id": job.id, "stage": "done", "cached": True}
    job = Job(id=uuid.uuid4().hex[:12], kepid=kepid, kepoi_name=koi)
    remember(job)
    threading.Thread(target=run_job, args=(job,), daemon=True).start()
    return {"job_id": job.id, "stage": job.stage, "cached": False}


@api.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return asdict(job)


_finish_app()
