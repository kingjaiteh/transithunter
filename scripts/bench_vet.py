"""Measure the API against the Phase 3 acceptance bar and check its verdicts.

PLAN.md asks for p95 under 2 seconds on a warm request for a cached star. The
API caches a finished result on disk, so there are two timings worth having and
this reports both:

  cold  POST /api/vet plus polling until the job finishes. The light curve is
        already in the FITS cache, so this is detrend, fold and classify, not a
        MAST download.
  warm  POST /api/vet plus one GET /api/jobs/{id} once the result is cached.
        This is what a visitor re-opening a star actually waits for, and the
        number the acceptance criterion is about.

The sampled KOIs come from the held-out test split, so the run doubles as an
end-to-end check that the served model still agrees with the catalogue on stars
it never trained on.

Start the server first, then:

    uv run python scripts/bench_vet.py --stars 20 --repeats 10
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from transithunter import config

OUT_PATH = config.ARTIFACTS_DIR / "phase3" / "latency.json"
TARGET_P95_SECONDS = 2.0


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile. Avoids interpolating over a handful of samples."""
    ordered = sorted(values)
    rank = min(len(ordered), max(1, math.ceil(q * len(ordered))))
    return ordered[rank - 1]


def pick_kois(n: int, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """A class-balanced sample of test-split KOIs, one per star."""
    table = pd.read_parquet(config.PROCESSED_DIR / "dataset_v1.parquet")
    test = table[table["split"] == "test"].drop_duplicates("kepid")
    per_class = max(1, n // 2)
    picked = pd.concat([
        test[test["label"] == label].sample(min(per_class, (test["label"] == label).sum()),
                                            random_state=seed)
        for label in (1, 0)
    ])
    return picked[["kepid", "kepoi_name", "koi_disposition", "label"]].reset_index(drop=True)


def run_job(base: str, kepid: int, koi: str, refresh: bool = False,
            poll: float = 0.2) -> tuple[float, dict]:
    """POST a vet job, poll to completion, return the elapsed seconds and the result."""
    t0 = time.perf_counter()
    started = requests.post(f"{base}/api/vet/{kepid}",
                            params={"koi": koi, "refresh": str(refresh).lower()},
                            timeout=30).json()
    job = requests.get(f"{base}/api/jobs/{started['job_id']}", timeout=30).json()
    while job["stage"] not in ("done", "error"):
        time.sleep(poll)
        job = requests.get(f"{base}/api/jobs/{started['job_id']}", timeout=30).json()
    elapsed = time.perf_counter() - t0
    if job["stage"] == "error":
        raise RuntimeError(f"KIC {kepid} {koi}: {job['error']}")
    return elapsed, job["result"]


def warm_request(base: str, kepid: int, koi: str) -> tuple[float, dict]:
    """The two calls the UI makes for an already-vetted star, timed together."""
    t0 = time.perf_counter()
    started = requests.post(f"{base}/api/vet/{kepid}", params={"koi": koi}, timeout=30).json()
    assert started["cached"], f"KIC {kepid} {koi} was not cached"
    job = requests.get(f"{base}/api/jobs/{started['job_id']}", timeout=30).json()
    return time.perf_counter() - t0, job["result"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--stars", type=int, default=20, help="test-split KOIs to vet")
    ap.add_argument("--repeats", type=int, default=10, help="warm requests per star")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    health = requests.get(f"{base}/api/health", timeout=10).json()
    print(f"server up, models_loaded={health['models_loaded']}, device={health['device']}")

    kois = pick_kois(args.stars)
    print(f"{len(kois)} test-split KOIs, {int(kois['label'].sum())} planets\n")

    cold: list[float] = []
    rows: list[dict] = []
    for i, koi in kois.iterrows():
        # refresh=True so the timing is the pipeline, not a leftover cache entry.
        try:
            seconds, result = run_job(base, int(koi["kepid"]), koi["kepoi_name"], refresh=True)
        except RuntimeError as exc:
            print(f"  [{i + 1}/{len(kois)}] {exc}")
            rows.append({"kepid": int(koi["kepid"]), "kepoi_name": koi["kepoi_name"],
                         "error": str(exc)})
            continue
        cold.append(seconds)
        agrees = (result["verdict"] == "PLANET") == (koi["koi_disposition"] == "CONFIRMED")
        rows.append({
            "kepid": int(koi["kepid"]), "kepoi_name": koi["kepoi_name"],
            "catalogue": koi["koi_disposition"], "verdict": result["verdict"],
            "probability": round(result["probability"], 4), "agrees": agrees,
            "cold_seconds": round(seconds, 2),
            "stage_seconds": {k: round(v, 2) for k, v in result["seconds"].items()},
        })
        print(f"  [{i + 1}/{len(kois)}] KIC {koi['kepid']} {koi['kepoi_name']}: "
              f"{result['verdict']:<14} p={result['probability']:.3f} "
              f"vs {koi['koi_disposition']:<14} {'ok' if agrees else 'MISS'}  {seconds:.1f} s")

    done = [r for r in rows if "error" not in r]
    warm: list[float] = []
    print(f"\nwarm pass: {args.repeats} repeats over {len(done)} cached stars")
    for _ in range(args.repeats):
        for r in done:
            seconds, _ = warm_request(base, r["kepid"], r["kepoi_name"])
            warm.append(seconds)

    p95 = percentile(warm, 0.95)
    summary = {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_url": base, "device": health["device"],
        "stars": len(done), "warm_requests": len(warm),
        "warm_seconds": {"p50": round(percentile(warm, 0.50), 3), "p95": round(p95, 3),
                         "max": round(max(warm), 3), "mean": round(statistics.fmean(warm), 3)},
        "cold_seconds": {"p50": round(percentile(cold, 0.50), 2),
                         "p95": round(percentile(cold, 0.95), 2), "max": round(max(cold), 2)},
        "target_p95_seconds": TARGET_P95_SECONDS,
        "passes": p95 < TARGET_P95_SECONDS,
        "agreement": sum(r["agrees"] for r in done) / len(done),
        "failures": [r for r in rows if "error" in r],
        "results": rows,
    }

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))

    w = summary["warm_seconds"]
    print(f"\nwarm  p50 {w['p50'] * 1000:.0f} ms  p95 {w['p95'] * 1000:.0f} ms  max {w['max'] * 1000:.0f} ms")
    print(f"cold  p50 {summary['cold_seconds']['p50']:.1f} s  p95 {summary['cold_seconds']['p95']:.1f} s")
    print(f"catalogue agreement {summary['agreement']:.1%} on {len(done)} held-out KOIs")
    print(f"p95 target {TARGET_P95_SECONDS:.0f} s: {'PASS' if summary['passes'] else 'FAIL'}")
    print(f"wrote {path}")
    return 0 if summary["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
