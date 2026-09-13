#!/usr/bin/env python3
"""Print, or optionally submit, the six frontier and six dependent MIP jobs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from giro_zero_fee_campaign import FEES, TARIFFS, digest, read_plan


SLURM = Path("/usr/local/slurm/slurm-25.05.5/bin")


def sbatch_frontier(plan: dict, cell: dict, worker: Path) -> list[str]:
    root = Path(plan["root"])
    return [
        str(SLURM / "sbatch"), "--parsable", "--partition=default_partition",
        "--exclude=scaglione-compute-01", "--cpus-per-task=1", "--mem=24G",
        "--time=02:00:00", "--no-requeue", "--job-name=gt_" + cell["id"],
        "--output=" + str(root / "logs/%x_%j.out"),
        "--error=" + str(root / "logs/%x_%j.err"), "--chdir=" + str(root),
        str(worker), cell["id"], "frontier",
    ]


def sbatch_mip(plan: dict, cell: dict, worker: Path,
               frontier_job_ids: list[str] | None = None) -> list[str]:
    root = Path(plan["root"])
    command = [
        str(SLURM / "sbatch"), "--parsable", "--partition=default_partition",
        "--exclude=scaglione-compute-01",
        "--cpus-per-task=8", "--mem=48G", "--time=02:00:00", "--no-requeue",
        "--job-name=gm_" + cell["id"],
        "--output=" + str(root / "logs/%x_%j.out"),
        "--error=" + str(root / "logs/%x_%j.err"), "--chdir=" + str(root),
    ]
    if frontier_job_ids:
        command.append("--dependency=afterok:" + ":".join(frontier_job_ids))
    command.extend([str(worker), cell["id"], "mip"])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    plan = read_plan(root)
    worker = root / "worker.sh"
    if not worker.is_file():
        raise FileNotFoundError(worker)
    cells = {cell["id"]: cell for cell in plan["cells"]}
    frontiers = {}
    for tariff in TARIFFS:
        for fee in FEES:
            cell = cells[f"{tariff}_fee{int(fee)}"]
            frontiers[cell["id"]] = sbatch_frontier(plan, cell, worker)
    if not args.submit:
        print(json.dumps({
            "plan_sha256": digest(root / "plan.json"),
            "frontier_commands": list(frontiers.values()),
            "mip_commands": [
                sbatch_mip(
                    plan, cells[f"{tariff}_fee{int(fee)}"], worker,
                    ["<" + tariff + "_fee0_frontier_job>", "<" + tariff + "_fee5_frontier_job>"],
                )
                for tariff in TARIFFS for fee in FEES
            ],
        }, indent=2))
        return
    root.joinpath("logs").mkdir(exist_ok=True)
    ledger_path = root / "jobs.json"
    if ledger_path.exists():
        raise FileExistsError(ledger_path)
    ledger = {"schema": "evsp-dr-terminal-energy-fee-submission-v1", "jobs": [],
              "plan_sha256": digest(root / "plan.json")}
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    for tariff in TARIFFS:
        frontier_ids = []
        for fee in FEES:
            cell = cells[f"{tariff}_fee{int(fee)}"]
            command = frontiers[cell["id"]]
            job = subprocess.check_output(command, text=True).strip().split(";")[0]
            frontier_ids.append(job)
            ledger["jobs"].append({"pair_id": cell["id"], "stage": "frontier", "job_id": job, "command": command})
            ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
        for fee in FEES:
            cell = cells[f"{tariff}_fee{int(fee)}"]
            command = sbatch_mip(plan, cell, worker, frontier_ids)
            job = subprocess.check_output(command, text=True).strip().split(";")[0]
            ledger["jobs"].append({"pair_id": cell["id"], "stage": "mip", "job_id": job, "dependency": frontier_ids, "command": command})
            ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")


if __name__ == "__main__":
    main()
