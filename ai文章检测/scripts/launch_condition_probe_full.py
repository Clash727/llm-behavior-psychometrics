#!/usr/bin/env python
"""Durable supervisor for the gated singleton-scored train/dev extraction."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/revy/miniconda3/envs/dl_env/bin/python")
CONFIG = ROOT / "configs/condition_probe_production.json"
OUTPUT = ROOT / "artifacts/condition_probe/full_singleton_v1"
SUPERVISOR = ROOT / "artifacts/condition_probe/supervisor_full_singleton_v1"
STATUS = SUPERVISOR / "supervisor_status.json"
LOG = SUPERVISOR / "console.log"


def process_start_ticks(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        return raw[raw.rfind(")") + 2:].split()[19]
    except (OSError, IndexError):
        return None


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def progress_snapshot() -> dict:
    latest = {}
    for split in ("train", "dev"):
        path = OUTPUT / split / "run_status.json"
        if path.exists():
            try:
                latest[split] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                latest[split] = {"status_read_error": True}
    return latest


def main() -> int:
    SUPERVISOR.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise SystemExit(f"refusing to reuse existing formal output directory: {OUTPUT}")
    if not CONFIG.is_file() or not PYTHON.is_file():
        raise SystemExit("formal config or dl_env Python interpreter is missing")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if cfg.get("scoring_batch_size") != 1 or cfg.get("shard_rows") != 32:
        raise SystemExit("formal config must use scoring_batch_size=1 and shard_rows=32")
    expected_output = (ROOT / cfg["output_dir"]).resolve()
    if expected_output != OUTPUT.resolve():
        raise SystemExit(f"formal config output path mismatch: {expected_output}")

    command = [str(PYTHON), "scripts/run_condition_probe.py", "--config",
               "configs/condition_probe_production.json", "--splits", "train", "dev"]
    started = datetime.now(timezone.utc).isoformat()
    initial = {"state": "starting", "started_utc": started,
               "supervisor_pid": os.getpid(),
               "supervisor_start_ticks": process_start_ticks(os.getpid()),
               "worker_pid": None, "worker_start_ticks": None, "returncode": None,
               "command": command, "config_path": str(CONFIG),
               "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
               "runner_sha256": hashlib.sha256((ROOT / "scripts/run_condition_probe.py").read_bytes()).hexdigest(),
               "worker_log": str(LOG), "formal_output": str(OUTPUT), "splits": ["train", "dev"],
               "scoring_batch_size": 1, "shard_rows": 32}
    atomic_json(STATUS, initial)

    returncode = None
    worker = None
    try:
        with LOG.open("ab", buffering=0) as log:
            log.write((json.dumps({"event": "supervisor_start", **initial}, ensure_ascii=False) + "\n").encode())
            worker = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True, close_fds=True)
            status = {**initial, "state": "running", "worker_pid": worker.pid,
                      "worker_start_ticks": process_start_ticks(worker.pid),
                      "worker_started_utc": datetime.now(timezone.utc).isoformat()}
            atomic_json(STATUS, status)
            log.write((json.dumps({"event": "worker_started", "pid": worker.pid,
                                   "start_ticks": status["worker_start_ticks"],
                                   "command": command}, ensure_ascii=False) + "\n").encode())
            while True:
                returncode = worker.poll()
                status = {**status,
                          "updated_utc": datetime.now(timezone.utc).isoformat(),
                          "elapsed_seconds": time.time() - worker_start_epoch,
                          "split_status": progress_snapshot(), "returncode": returncode,
                          "state": "complete" if returncode == 0 else "failed" if returncode is not None else "running"}
                atomic_json(STATUS, status)
                if returncode is not None:
                    log.write((json.dumps({"event": "worker_exit", "returncode": returncode,
                                           "state": status["state"]}) + "\n").encode())
                    break
                time.sleep(5)
    except BaseException as exc:
        if worker is not None and worker.poll() is None:
            # Keep the worker detached and mark supervisor attention explicitly.
            atomic_json(STATUS, {**initial, "state": "supervisor_error",
                                 "worker_pid": worker.pid,
                                 "worker_start_ticks": process_start_ticks(worker.pid),
                                 "returncode": None, "error": repr(exc),
                                 "split_status": progress_snapshot()})
        raise
    return int(returncode or 0)


if __name__ == "__main__":
    worker_start_epoch = time.time()
    raise SystemExit(main())
