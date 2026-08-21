#!/usr/bin/env python3
"""Launch run_exact_pool_mip, signal it during search, and record the event."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from durable_io import atomic_write_json


def run(args):
    progress = Path(args.progress_dir)
    command = [
        sys.executable, "-u", str(Path(__file__).with_name("run_exact_pool_mip.py")),
        "--result", str(args.result),
        "--timelimit", str(int(args.timelimit)),
        "--mipgap", "0", "--threads", "1", "--two-stage",
        "--progress-dir", str(progress), "--out", str(args.out),
    ]
    process = subprocess.Popen(command, cwd=args.cwd)
    latest = progress / "latest.json"
    deadline = time.monotonic() + 30.0
    observed = None
    while time.monotonic() < deadline:
        if latest.is_file():
            try:
                observed = json.loads(latest.read_text())
            except (OSError, ValueError):
                observed = None
            if observed and observed.get("stage") == "fleet":
                break
        if process.poll() is not None:
            break
        time.sleep(0.005)
    time.sleep(args.signal_delay_s)
    signal_sent = process.poll() is None
    if signal_sent:
        process.send_signal(signal.SIGINT)
    returncode = process.wait(timeout=30)
    payload = {
        "command": command,
        "signal": "SIGINT" if signal_sent else None,
        "signal_delay_s": args.signal_delay_s,
        "latest_before_signal": observed,
        "returncode": returncode,
    }
    atomic_write_json(args.audit_out, payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--progress-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--timelimit", type=float, default=120)
    parser.add_argument("--signal-delay-s", type=float, default=0.25)
    args = parser.parse_args(argv)
    payload = run(args)
    print(json.dumps({
        "signal": payload["signal"], "returncode": payload["returncode"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
