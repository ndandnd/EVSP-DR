#!/usr/bin/env python3
"""Reconcile an ambiguous ladder gate release from Slurm accounting."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from launch_scale_ladder import _replace_json


def reconcile(root, expected_plan_sha):
    root = Path(root).resolve()
    plan_raw = (root / "approved-plan.json").read_bytes()
    if hashlib.sha256(plan_raw).hexdigest() != expected_plan_sha:
        raise ValueError("approved plan hash mismatch")
    manifest_path = root / "campaign.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("approval_sha256") != expected_plan_sha
        or manifest.get("gate_state")
        not in {"release_attempting", "held_release_failed"}
        or not str(manifest.get("gate_job_id") or "").isdigit()
    ):
        raise ValueError("campaign is not in a reconcilable state")
    gate = str(manifest["gate_job_id"])
    completed = subprocess.run(
        [
            "sacct", "-X", "-n", "-P", "-j", gate,
            "--format=JobIDRaw,State",
        ],
        text=True, capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("cannot query sacct")
    states = {
        fields[0]: fields[1].split()[0].split("+", 1)[0]
        for line in completed.stdout.splitlines()
        if len(fields := line.split("|")) >= 2
    }
    if states.get(gate) != "COMPLETED":
        raise ValueError("gate is not proven completed")
    manifest["gate_state"] = "released_reconciled"
    manifest["submitted"] = True
    manifest["gate_reconciliation"] = {
        "source": "sacct", "gate_job_id": gate, "state": "COMPLETED",
    }
    _replace_json(manifest_path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--approved-plan-sha256", required=True)
    args = parser.parse_args(argv)
    reconcile(args.campaign_root, args.approved_plan_sha256)
    print("LADDER GATE RECONCILED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
