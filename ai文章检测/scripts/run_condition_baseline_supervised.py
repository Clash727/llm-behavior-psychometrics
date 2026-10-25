#!/usr/bin/env python3
"""Run the formal CPU analysis under a persistent lock and status record.

This wrapper does not create a release marker. The baseline runner remains the
authority for validating its contents and the GPU extraction manifests.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def proc_start_ticks(pid: int) -> int | None:
    """Return Linux /proc starttime (clock ticks since boot), if available."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        tail = stat[stat.rfind(")") + 2:].split()  # first tail field is proc field 3
        return int(tail[19])  # proc field 22
    except (OSError, ValueError, IndexError):
        return None


def atomic_state(path: Path, record: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--release-marker", type=Path,
                        default=ROOT / "artifacts/condition_baseline/FORMAL_RELEASE.json")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "artifacts/condition_baseline/formal")
    parser.add_argument("--manifest-dir", type=Path,
                        default=ROOT / "artifacts/condition_baseline/manifests")
    args = parser.parse_args()

    out = args.output_dir.resolve()
    release = args.release_marker.resolve()
    probe_root = args.probe_root.resolve()
    manifest_dir = args.manifest_dir.resolve()
    lock_path = out.with_name(out.name + ".lock")
    state_path = out.with_name(out.name + ".run_state.json")
    log_path = out.with_name(out.name + ".supervisor.log")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    if not release.is_file():
        raise SystemExit(f"formal run refused: release marker is absent: {release}")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"formal run refused: nonempty output already exists: {out}")
    if state_path.exists():
        raise SystemExit(f"formal run refused: prior supervisor state exists: {state_path}; inspect before recovery")

    lock_file = lock_path.open("a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"formal run refused: another supervisor holds {lock_path}")

    record = {
        "state": "starting", "started_at_utc": utc_now(), "supervisor_pid": os.getpid(),
        "supervisor_start_ticks": proc_start_ticks(os.getpid()),
        "probe_root": str(probe_root), "release_marker": str(release),
        "release_marker_sha256": sha(release), "output_dir": str(out),
        "lock_path": str(lock_path), "log_path": str(log_path),
    }
    atomic_state(state_path, record)

    command = [sys.executable, str(ROOT / "scripts/run_condition_baseline.py"), "run",
               "--probe-root", str(probe_root), "--release-marker", str(release),
               "--output-dir", str(out), "--manifest-dir", str(manifest_dir)]
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "2"
    with log_path.open("ab", buffering=0) as log:
        log.write((f"[{utc_now()}] launching: {' '.join(command)}\n").encode())
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 pass_fds=(lock_file.fileno(),))
        record.update({"state": "running", "child_pid": child.pid,
                       "child_start_ticks": proc_start_ticks(child.pid), "command": command,
                       "started_at_utc": utc_now()})
        atomic_state(state_path, record)
        return_code = child.wait()
        record.update({"state": "complete" if return_code == 0 else "failed",
                       "return_code": return_code, "finished_at_utc": utc_now()})
        atomic_state(state_path, record)
        log.write((f"[{utc_now()}] finished with exit code {return_code}\n").encode())
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
