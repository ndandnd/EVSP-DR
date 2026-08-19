#!/usr/bin/env python3
"""Deterministic primary-grid membership gate for tariff fixed-duty jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_tariff_response_manifest import REPO_ROOT, sha256_file


SCHEMA = "evsp-dr-tariff-fixed-duty-submission-preflight-v1"
MEMBERSHIP_PATH = (
    REPO_ROOT / "data/scale_ladder/known_membership_preflight.json"
)
MEMBERSHIP_SHA256 = (
    "5124534373e8d3aff981c55891b8f7ed321fdf1efe96c8bbfd093d957c1b94c8"
)
PRIMARY_PHYSICS = {
    "g_kwh": 300.0,
    "charge_kw": 300.0,
    "reserve_kwh": 0.0,
    "soc_step": 15.0,
    "block_min": 10,
}


def _canonical(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def build_preflight(plan, membership_path=MEMBERSHIP_PATH):
    path = Path(membership_path).expanduser().resolve()
    if sha256_file(path) != MEMBERSHIP_SHA256:
        raise ValueError("known-membership preflight hash mismatch")
    membership = json.loads(path.read_text())
    if (
        membership.get("schema")
        != "evsp-dr-scale-ladder-membership-preflight-v1"
        or membership.get("membership_schema")
        != "evsp-dr-scale-ladder-known-membership-v1"
    ):
        raise ValueError("known-membership preflight schema mismatch")
    physics = plan.get("physics") or {}
    observed_physics = {
        "g_kwh": float(physics.get("g_kwh", -1)),
        "charge_kw": float(physics.get("charge_kw", -1)),
        "reserve_kwh": float(physics.get("reserve_kwh", -1)),
        "soc_step": float(physics.get("soc_step", -1)),
        "block_min": int(physics.get("block_min", -1)),
    }
    if observed_physics != PRIMARY_PHYSICS:
        raise ValueError("tariff pilot primary physics changed")
    by_instance = {}
    for cell in membership.get("cells") or []:
        instance_sha = (
            (cell.get("trip_identity") or {}).get("instance_file_sha256")
        )
        if not instance_sha or instance_sha in by_instance:
            raise ValueError("membership instances are missing or duplicated")
        by_instance[instance_sha] = cell
    instance_rows = []
    all_blockers = []
    for key in ("k5", "k8", "k40"):
        instance = (plan.get("instances") or {}).get(key) or {}
        instance_sha = instance.get("sha256")
        cell = by_instance.get(instance_sha)
        if cell is None or int(cell.get("scale", -1)) != int(key[1:]):
            raise ValueError(f"{key} has no exact membership preflight")
        duties = cell.get("duties") or []
        if len(duties) != int(instance.get("duty_count", -1)):
            raise ValueError(f"{key} membership duty count mismatch")
        blockers = [{
            "duty_id": str(duty["duty_id"]),
            "known_partition_continuously_feasible":
                duty.get("known_partition_continuously_feasible"),
            "known_partition_in_primary_expanded_space":
                duty.get("known_partition_in_primary_expanded_space"),
            "fixed_sequence_pricing_certified":
                duty.get("fixed_sequence_pricing_certified"),
            "nonrepresentability_reason":
                duty.get("nonrepresentability_reason"),
            "first_feasible_soc_step":
                duty.get("first_feasible_soc_step"),
            "first_feasible_block_min":
                duty.get("first_feasible_block_min"),
        } for duty in duties
            if duty.get("known_partition_in_primary_expanded_space")
            is not True
        ]
        if any(
            blocker["known_partition_continuously_feasible"] is not True
            or not blocker["nonrepresentability_reason"]
            for blocker in blockers
        ):
            raise ValueError(f"{key} blocker evidence is incomplete")
        row = {
            "instance_key": key,
            "cell_id": cell["cell_id"],
            "instance_file_sha256": instance_sha,
            "duty_count": len(duties),
            "primary_grid_representable_duty_count":
                len(duties) - len(blockers),
            "primary_grid_nonrepresentable_duty_count": len(blockers),
            "known_partition_in_primary_expanded_space":
                cell.get("known_partition_in_primary_expanded_space"),
            "blockers": blockers,
        }
        instance_rows.append(row)
        all_blockers.extend({
            "instance_key": key,
            **blocker,
        } for blocker in blockers)
    affected = [
        job["job_key"] for job in plan.get("jobs") or []
        if (
            job["phase"] == "FIXED_FULL"
            or job["phase"] == "SEED"
            or (
                job["treatment"] in {
                    "GIRO-AUGMENTED", "GIRO40-AUGMENTED",
                }
                and job["phase"] in {"CG", "MIP"}
            )
        )
    ]
    payload = {
        "schema": SCHEMA,
        "membership_path": str(path),
        "membership_sha256": MEMBERSHIP_SHA256,
        "physics": PRIMARY_PHYSICS,
        "instances": instance_rows,
        "primary_grid_nonrepresentable_duty_count": len(all_blockers),
        "affected_job_keys": affected,
        "affected_job_count": len(affected),
        "submission_blocked": bool(all_blockers),
        "blocking_policy":
            "fail_closed_no_grid_change_no_duty_skip",
        "blockers": all_blockers,
    }
    payload["preflight_sha256"] = hashlib.sha256(
        _canonical(payload)
    ).hexdigest()
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    payload = build_preflight(plan)
    encoded = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as handle:
            handle.write(encoded)
    print(encoded, end="")
    return 2 if payload["submission_blocked"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
