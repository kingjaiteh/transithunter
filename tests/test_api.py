"""Contract tests for the job API.

The real Vetter needs trained weights and a FITS cache, neither of which
belongs in a test run, so these swap in a stub that returns a fixed
`VetResult`. What is under test is the serving layer: job lifecycle, staged
progress, the result cache, and error handling.
"""

from __future__ import annotations

import json
import threading
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from transithunter.inference import Ephemeris, VetResult

KEPID = 11904151  # Kepler-10
KOI = "K00072.01"


def a_result(kepid: int = KEPID, verdict: str = "PLANET") -> VetResult:
    return VetResult(
        kepid=kepid,
        ephemeris=Ephemeris(0.8375, 133.0, 1.8, "catalogue", KOI, 152.0),
        verdict=verdict,
        probability=0.824,
        threshold=0.5,
        baseline_probability=0.71,
        catalogue_disposition="CONFIRMED",
        global_view=[0.0] * 2001,
        local_view=[0.0] * 201,
        n_points=60000,
        n_transits_seen=1700,
        seconds={"fetch": 2.0, "fold": 1.0},
        cached_light_curve=True,
    )


class StubVetter:
    """Stands in for Vetter. `gate`, when cleared, holds vet() mid-pipeline."""

    def __init__(self) -> None:
        self.device = "cpu"
        self.cnn_card = {"threshold": 0.5, "test": {"pr_auc": 0.975}, "config": {"dropout": 0.3}}
        self.baseline_card = {"threshold": 0.5, "test": {"pr_auc": 0.968}, "features": ["koi_prad"]}
        self.calls: list[tuple[int, str | None]] = []
        self.gate = threading.Event()
        self.gate.set()
        self.raises: Exception | None = None

    def kois_on(self, kepid: int) -> pd.DataFrame:
        return pd.DataFrame([{
            "kepoi_name": KOI, "kepler_name": "Kepler-10 b", "koi_disposition": "CONFIRMED",
            "koi_period": 0.8375, "koi_depth": 152.0, "koi_duration": 1.8, "kepid": kepid,
        }])

    def vet(self, kepid, kepoi_name=None, progress=None):
        self.calls.append((kepid, kepoi_name))
        if progress:
            progress("fetch")
        self.gate.wait(timeout=5)
        if self.raises is not None:
            raise self.raises
        return a_result(kepid)


@pytest.fixture
def stub(tmp_path, monkeypatch):
    from api import main

    vetter = StubVetter()
    monkeypatch.setattr(main, "_vetter", vetter)
    monkeypatch.setattr(main, "CACHE_DIR", tmp_path / "vet_cache")
    monkeypatch.setattr(main, "_jobs", main.OrderedDict())
    return vetter


@pytest.fixture
def client(stub):
    from api.main import app

    with TestClient(app) as c:  # the `with` runs the lifespan model warm-up
        yield c


def wait_for(client: TestClient, job_id: str, stage: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["stage"] == stage:
            return job
        time.sleep(0.02)
    pytest.fail(f"job {job_id} never reached {stage}, last stage {job['stage']}")


def test_health_and_examples(client):
    health = client.get("/api/health").json()
    assert health["ok"] and health["models_loaded"]

    examples = client.get("/api/examples").json()
    assert len(examples) >= 2
    assert {"kepid", "name", "note"} <= set(examples[0])
    # The demo needs one of each so a reviewer can see both verdicts.
    notes = " ".join(e["note"] for e in examples)
    assert "confirmed" in notes and "false positive" in notes


def test_models_endpoint_serves_both_cards(client):
    cards = client.get("/api/models").json()
    assert cards["cnn"]["test"]["pr_auc"] > 0
    assert cards["baseline"]["features"] == ["koi_prad"]


def test_kois_endpoint_returns_display_columns(client):
    rows = client.get(f"/api/kois/{KEPID}").json()
    assert rows[0]["kepoi_name"] == KOI
    # kepid would be redundant in the dropdown, and the frontend type omits it.
    assert "kepid" not in rows[0]


def test_vet_runs_a_job_then_serves_it_from_cache(client, stub, tmp_path):
    started = client.post(f"/api/vet/{KEPID}?koi={KOI}").json()
    assert started["cached"] is False

    job = wait_for(client, started["job_id"], "done")
    assert job["result"]["verdict"] == "PLANET"
    assert job["result"]["kepid"] == KEPID
    assert len(job["result"]["global_view"]) == 2001
    assert job["error"] is None

    cached = tmp_path / "vet_cache" / f"{KEPID}_{KOI}.json"
    assert json.loads(cached.read_text())["probability"] == pytest.approx(0.824)

    again = client.post(f"/api/vet/{KEPID}?koi={KOI}").json()
    assert again["cached"] is True and again["stage"] == "done"
    assert client.get(f"/api/jobs/{again['job_id']}").json()["result"]["verdict"] == "PLANET"
    assert len(stub.calls) == 1, "a cache hit must not re-run the pipeline"


def test_refresh_reruns_a_cached_star(client, stub):
    started = client.post(f"/api/vet/{KEPID}?koi={KOI}").json()
    wait_for(client, started["job_id"], "done")

    again = client.post(f"/api/vet/{KEPID}?koi={KOI}&refresh=true").json()
    assert again["cached"] is False
    wait_for(client, again["job_id"], "done")
    assert len(stub.calls) == 2


def test_job_reports_the_stage_it_is_on(client, stub):
    stub.gate.clear()
    started = client.post(f"/api/vet/{KEPID}").json()
    job = wait_for(client, started["job_id"], "fetch")
    assert job["stages"][:2] == ["ephemeris", "fetch"]
    assert job["result"] is None

    stub.gate.set()
    wait_for(client, started["job_id"], "done")


def test_a_failed_job_reports_the_error(client, stub):
    stub.raises = LookupError("K99999.01 is not a KOI on KIC 1")
    started = client.post(f"/api/vet/{KEPID}").json()

    job = wait_for(client, started["job_id"], "error")
    assert "LookupError" in job["error"] and "K99999.01" in job["error"]
    assert job["result"] is None


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/nosuchjob").status_code == 404


def test_the_job_store_stays_bounded(client, stub, monkeypatch):
    from api import main

    monkeypatch.setattr(main, "MAX_JOBS", 4)
    for kepid in range(100, 112):
        client.post(f"/api/vet/{kepid}")
    assert len(main._jobs) == 4
