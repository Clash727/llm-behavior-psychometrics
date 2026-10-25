#!/usr/bin/env python3
"""Detached supervisor for a durable token-confidence extraction run."""
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
PYTHON = "/home/revy/miniconda3/envs/dl_env/bin/python"
CONFIG = ROOT / "configs/token_confidence_full.json"
OUT = ROOT / "artifacts/token_confidence/features_full"
LOG = OUT / "run.log"
STATUS = OUT / "run_status.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def start_ticks(pid: int):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        tail = stat[stat.rfind(")") + 2:].split()
        return int(tail[19])  # proc stat field 22 (starttime)
    except (OSError, ValueError, IndexError):
        return None


def atomic_status(payload):
    temp = STATUS.with_suffix(".json.tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, STATUS)
    fd = os.open(OUT, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    command = [PYTHON, str(ROOT / "scripts/run_token_confidence.py"),
               "--config", str(CONFIG), "--batch-size", "2", "--report"]
    config_hash = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    base = {"config": str(CONFIG.relative_to(ROOT)), "config_sha256": config_hash,
            "command": command, "supervisor_pid": os.getpid(),
            "supervisor_start_ticks": start_ticks(os.getpid())}
    started = time.time()
    try:
        with LOG.open("ab", buffering=0) as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, close_fds=True)
            running = {**base, "status": "running", "worker_pid": child.pid,
                       "worker_start_ticks": start_ticks(child.pid), "started_at": utc_now(),
                       "log": str(LOG.relative_to(ROOT))}
            atomic_status(running)
            returncode = child.wait()
        final = {**running, "status": "complete" if returncode == 0 else "failed",
                 "returncode": int(returncode), "finished_at": utc_now(),
                 "elapsed_seconds": round(time.time() - started, 3)}
        atomic_status(final)
        return int(returncode)
    except Exception as exc:
        failed = {**base, "status": "failed", "returncode": None,
                  "error": repr(exc), "finished_at": utc_now(),
                  "elapsed_seconds": round(time.time() - started, 3)}
        atomic_status(failed)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
