"""Resumable bulk downloader for the sampled stars.

Keeps a manifest CSV on the data drive. A star with status "ok" in the
manifest is skipped, so the script can be killed and restarted at any point.
Run: uv run python -m transithunter.data.download
"""

from __future__ import annotations

import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from transithunter import config
from transithunter.data.fetch import fetch_stitched
from transithunter.data.sample import load_sample

MANIFEST = config.DATA_DIR / "download_manifest.csv"
FIELDS = ["kepid", "status", "quarters", "points", "megabytes", "seconds", "error", "finished_at"]
PAUSE_S = 1.0
RETRIES = 3


def read_manifest(path: Path = MANIFEST) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=FIELDS)
    return pd.read_csv(path)


def done_stars(path: Path = MANIFEST) -> set[int]:
    m = read_manifest(path)
    return set(m.loc[m["status"] == "ok", "kepid"].astype(int))


def append(row: dict, path: Path = MANIFEST) -> None:
    new = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def download_one(kepid: int) -> dict:
    last_error = ""
    for attempt in range(1, RETRIES + 1):
        try:
            r = fetch_stitched(kepid)
            return {"kepid": kepid, "status": "ok", "quarters": r.n_files, "points": r.n_points,
                    "megabytes": round(r.megabytes, 2), "seconds": round(r.seconds, 1),
                    "error": "", "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        except LookupError as exc:
            return {"kepid": kepid, "status": "missing", "error": str(exc),
                    "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        except Exception as exc:  # noqa: BLE001  - network flakiness, retry
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(5 * attempt)
    return {"kepid": kepid, "status": "failed", "error": last_error,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def main(limit: int | None = None) -> None:
    stars = sorted(load_sample()["kepid"].unique())
    done = done_stars()
    todo = [k for k in stars if k not in done]
    if limit:
        todo = todo[:limit]
    print(f"{len(stars)} stars in sample, {len(done)} done, {len(todo)} to go", flush=True)

    t0 = time.time()
    for i, kepid in enumerate(todo, 1):
        row = download_one(int(kepid))
        append(row)
        elapsed = time.time() - t0
        rate = elapsed / i
        eta_h = rate * (len(todo) - i) / 3600
        print(f"[{i}/{len(todo)}] KIC {kepid} {row['status']} {row.get('megabytes', '')} MB "
              f"{row.get('seconds', '')} s  eta {eta_h:.1f} h", flush=True)
        time.sleep(PAUSE_S)
    print("done", flush=True)


if __name__ == "__main__":
    main(limit=int(sys.argv[1]) if len(sys.argv) > 1 else None)
