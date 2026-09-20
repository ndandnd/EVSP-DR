#!/usr/bin/env python
"""Collect the diving-pricing pilot into one comparison table.

Read-only over every result directory; writes only into ``--work/status_*``.

    python collect.py --work /home/nc437/ladder-lite/diving_pricing_20260919
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

ARMS = ("arm_a", "arm_b")
COLUMNS = [
    "case_id", "budget_arm", "graph_build_s", "graph_build_charged_s",
    "control_buses", "control_fleet_bound", "control_fleet_proven",
    "control_status", "control_runtime_s", "control_timelimit_s",
    "control_pool_columns", "control_result_sha256",
    "treatment_buses", "treatment_fleet_bound", "treatment_fleet_proven",
    "treatment_status", "treatment_runtime_s", "treatment_mip_timelimit_s",
    "treatment_pool_columns", "treatment_result_sha256",
    "cache_io_s", "dive_stage_s", "mip_stage_s", "total_stage_s",
    "dive_stop_reason", "dive_nodes", "columns_generated",
    "dive_reached_target", "source_immutable",
    "fresh_cg_result_sha256", "fresh_journal_sha256",
    "augmented_result_sha256", "augmented_journal_sha256",
]


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def latest(directory: Path, name: str):
    if not directory.is_dir():
        return None
    runs = sorted(
        (path for path in directory.iterdir() if (path / name).is_file()),
        key=lambda path: (path / name).stat().st_mtime,
    )
    return runs[-1] if runs else None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args(argv)

    work = args.work.expanduser().resolve()
    manifest = json.loads((work / "manifest.json").read_text())
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    status = work / f"status_{stamp}"
    status.mkdir(parents=True, exist_ok=True)

    rows = []
    for case_id, case in sorted(manifest["cases"].items()):
        for arm in ARMS:
            row = {name: None for name in COLUMNS}
            row.update(
                case_id=case_id, budget_arm=arm,
                graph_build_s=case["budget"]["graph_build_s"],
                graph_build_charged_s=(
                    0.0 if arm == "arm_a"
                    else case["budget"]["graph_build_s"]
                ),
                fresh_cg_result_sha256=case["fresh_cg_result_sha256"],
                fresh_journal_sha256=case.get("fresh_journal_sha256"),
            )
            base = work / "results" / case_id
            control = latest(base / f"control_{arm}", "result.json")
            if control is not None:
                value = json.loads((control / "result.json").read_text())
                row.update(
                    control_buses=value.get("buses"),
                    control_fleet_bound=value.get("fleet_bound"),
                    control_fleet_proven=value.get("fleet_proven"),
                    control_status=value.get("status_name"),
                    control_runtime_s=value.get("runtime_s"),
                    control_pool_columns=value.get("pool_columns"),
                    control_result_sha256=sha256(control / "result.json"),
                )
                execution = control / "execution.json"
                if execution.is_file():
                    row["control_timelimit_s"] = json.loads(
                        execution.read_text()
                    ).get("timelimit_s")
            # One treatment run per case is paired against BOTH control
            # rows: the arms differ only in the control budget and in how
            # the graph build is charged, never in what the treatment does.
            treatment = latest(base / "treatment", "result.json")
            if treatment is not None:
                value = json.loads((treatment / "result.json").read_text())
                row.update(
                    treatment_buses=value.get("buses"),
                    treatment_fleet_bound=value.get("fleet_bound"),
                    treatment_fleet_proven=value.get("fleet_proven"),
                    treatment_status=value.get("status_name"),
                    treatment_runtime_s=value.get("runtime_s"),
                    treatment_pool_columns=value.get("pool_columns"),
                    treatment_result_sha256=sha256(treatment / "result.json"),
                )
                timing_path = treatment / "timing.json"
                if timing_path.is_file():
                    timing = json.loads(timing_path.read_text())
                    integer = timing.get("dive_integer_solution") or {}
                    row.update(
                        cache_io_s=timing.get("cache_io_s"),
                        dive_stage_s=timing.get("dive_stage_s"),
                        mip_stage_s=timing.get("mip_stage_s"),
                        total_stage_s=timing.get("total_stage_s"),
                        treatment_mip_timelimit_s=timing.get(
                            "mip_timelimit_s"
                        ),
                        dive_stop_reason=timing.get("dive_stop_reason"),
                        dive_nodes=timing.get("dive_nodes"),
                        columns_generated=timing.get("columns_generated"),
                        dive_reached_target=integer.get("buses"),
                        source_immutable=timing.get("source_immutable"),
                        augmented_result_sha256=timing.get(
                            "augmented_result_sha256"
                        ),
                        augmented_journal_sha256=timing.get(
                            "augmented_journal_sha256"
                        ),
                    )
            rows.append(row)

    with open(status / "comparison.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "# Diving-with-pricing pilot — collected status",
        "",
        f"Collected {stamp}. Execution commit "
        f"`{manifest['execution_commit']}`; MIP runner "
        f"`{manifest['mip_execution_commit']}`.",
        "",
        "Fleet results are proofs **within the supplied pool**. No dive-node "
        "outcome is a global certificate.",
        "",
        "One treatment run per case is paired against both control rows; "
        "the arms differ in the control budget and in whether the graph "
        "build is charged, not in the treatment itself.",
        "",
        "| case | arm | graph build s | control buses (proven) | "
        "treatment buses (proven) | dive reached | new columns | dive stop | "
        "treatment total s |",
        "|---|---|---:|---|---|---:|---:|---|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['case_id']} | {row['budget_arm']} | "
            f"{row['graph_build_s']:.0f} | "
            f"{row['control_buses']} ({row['control_fleet_proven']}) | "
            f"{row['treatment_buses']} ({row['treatment_fleet_proven']}) | "
            f"{row['dive_reached_target']} | {row['columns_generated']} | "
            f"{row['dive_stop_reason']} | {row['total_stage_s']} |"
        )
    immutable = {row["source_immutable"] for row in rows if
                 row["source_immutable"] is not None}
    report += [
        "",
        f"Source pools immutable across every treatment job: "
        f"{immutable if immutable else 'no treatment job has reported yet'}.",
        "",
        "Per-row provenance (source and augmented hashes, timings, MIP time "
        "limits) is in [comparison.csv](comparison.csv).",
    ]
    (status / "README.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
