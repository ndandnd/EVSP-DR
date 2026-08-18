#!/usr/bin/env python3
"""Reconcile an ambiguous ladder gate release from Slurm accounting."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from launch_scale_ladder import _replace_json


def reconcile(root, expected_plan_sha, *, release_held_gate=False):
    root = Path(root).resolve()
    plan_raw = (root / "approved-plan.json").read_bytes()
    if hashlib.sha256(plan_raw).hexdigest() != expected_plan_sha:
        raise ValueError("approved plan hash mismatch")
    manifest_path = root / "campaign.json"
    plan = json.loads(plan_raw)
    sacct = plan.get("sacct") or {}
    sacct_path = Path(str(sacct.get("path") or ""))
    if (
        sacct.get("available") is not True
        or not sacct_path.is_file()
        or hashlib.sha256(sacct_path.read_bytes()).hexdigest()
        != sacct.get("sha256")
    ):
        raise ValueError("approved sacct executable unavailable/changed")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("approval_sha256") != expected_plan_sha
        or manifest.get("gate_state")
        not in {
            "release_attempting", "held_release_failed",
            "release_retry_attempting", "release_retry_requested",
        }
        or not str(manifest.get("gate_job_id") or "").isdigit()
    ):
        raise ValueError("campaign is not in a reconcilable state")
    gate = str(manifest["gate_job_id"])
    completed = subprocess.run(
        [
            str(sacct_path), "-X", "-n", "-P", "-j", gate,
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
    if states.get(gate) == "PENDING" and release_held_gate:
        scontrol = plan.get("scontrol") or {}
        scontrol_path = Path(str(scontrol.get("path") or ""))
        if (
            scontrol.get("available") is not True
            or not scontrol_path.is_file()
            or hashlib.sha256(scontrol_path.read_bytes()).hexdigest()
            != scontrol.get("sha256")
        ):
            raise ValueError("approved scontrol unavailable/changed")
        shown = subprocess.run(
            [str(scontrol_path), "show", "job", gate, "-o"],
            text=True, capture_output=True, check=False,
        )
        if (
            shown.returncode != 0
            or "JobState=PENDING" not in shown.stdout
            or "Reason=JobHeldUser" not in shown.stdout
        ):
            raise ValueError("gate is not proven held by the user")
        manifest["gate_state"] = "release_retry_attempting"
        _replace_json(manifest_path, manifest)
        released = subprocess.run(
            [str(scontrol_path), "release", gate],
            text=True, capture_output=True, check=False,
        )
        if released.returncode != 0:
            manifest["gate_state"] = "held_release_failed"
            manifest["release_error"] = (
                released.stderr or released.stdout
            ).strip()
            _replace_json(manifest_path, manifest)
            raise RuntimeError("gate release retry failed")
        manifest["gate_state"] = "release_retry_requested"
        _replace_json(manifest_path, manifest)
        return manifest
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
    parser.add_argument("--release-held-gate", action="store_true")
    args = parser.parse_args(argv)
    payload = reconcile(
        args.campaign_root,
        args.approved_plan_sha256,
        release_held_gate=args.release_held_gate,
    )
    print(f"LADDER GATE STATE: {payload['gate_state']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
