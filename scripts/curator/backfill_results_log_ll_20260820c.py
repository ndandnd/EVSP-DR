#!/usr/bin/env python3
"""Backfill records/RESULTS_LOG.csv for campaign ll_20260820c from committed aggregates.

The sanctioned writer, scripts/ladder_lite/record_results.sh, requires the
campaign's normalized outputs (cg_run_summary.csv, mip_run_summary.csv,
cg_iteration_long.csv, mip_checkpoint_long.csv) plus approved-plan.json and
campaign.json under $LL_ROOT. Those files were never committed to any git ref
(verified 2026-08-21 across all refs); durable copies exist only on the
cluster under ~/ladder-lite (STATUS_20260821.md section 8). Executed on this
branch it fails closed with "normalized summaries missing" (exit 1).

This script therefore derives rows from the two committed, reviewed campaign
aggregates instead, labels them curator_backfill_from_committed_aggregates,
and leaves every field the aggregates do not carry empty rather than inferred.
See records/runs/ll_20260820c/README.md for the field-by-field provenance.
"""
import csv
import hashlib
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
RUN_ID = "ll_20260820c"
DATE = "2026-08-21"
COMMIT = "339db0ab917a3db1b47e63e4debcab3066d0af79"  # D0010: local-only campaign commit
LABEL = "curator_backfill_from_committed_aggregates"
MEANING = "combined-cost-master route weight"
LADDER = REPO / "analysis/scale_ladder/ll_20260820c/ladder_summary.csv"
MATRIX = REPO / "analysis/scale_ladder/ll_20260820c/resolution_matrix.csv"
TARGET = REPO / "records/RESULTS_LOG.csv"


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    with TARGET.open(newline="") as h:
        reader = csv.DictReader(h)
        fields = reader.fieldnames
        if any(r["run_id"] == RUN_ID for r in reader):
            print(f"rows for {RUN_ID} already present; refusing to append twice")
            return 1

    ladder_sha, matrix_sha = sha(LADDER), sha(MATRIX)
    rows = []

    def base_row():
        row = {f: "" for f in fields}
        row.update({
            "date_utc": DATE, "run_id": RUN_ID,
            "execution_mode": "ladder_lite_direct_array", "commit": COMMIT,
            "label": LABEL, "route_weight_meaning": MEANING,
        })
        return row

    with LADDER.open(newline="") as h:
        for r in csv.DictReader(h):
            m = re.fullmatch(r"k(\d+)_s(\d+)_c(\d+)", r["cell"])
            certified = r["certified"] == "True"
            row = base_row()
            row.update({
                "group": "CG", "cell_id": r["cell"], "phase": "CG",
                "scale": r["scale"], "sel_rep": m.group(2), "cg_rep": m.group(3),
                "soc_step": "15.0", "block_min": "10",
                "status": "completed" if r["stop_state"] == "certified" else "censored",
                "route_weight": r["route_weight"],
                "min_reduced_cost": r["min_rc"], "certified": r["certified"],
                "artificial_mass": r["artificials"], "n_columns": r["pool_columns"],
                "iters": r["iterations"], "wall_s": str(round(float(r["elapsed_h"]) * 3600)),
                "stop_reason": r["stop_state"],
                "censor_reason": "" if r["stop_state"] == "certified" else r["stop_state"],
                "target_fleet": r["target_fleet"],
                "artifact_path": str(LADDER.relative_to(REPO)), "artifact_sha256": ladder_sha,
                "notes": f"charging_cost_usd={r['charging_cost_usd']}; wall_s from elapsed_h "
                         "(3-decimal hours); primary 15kWh/10min grid; "
                         "see records/runs/ll_20260820c/README.md",
            })
            rows.append(row)

    with MATRIX.open(newline="") as h:
        for r in csv.DictReader(h):
            if r["soc_step_kwh"] == "15.0" and r["block_min"] == "10":
                continue  # same runs as the ladder_summary primary rows above
            m = re.fullmatch(r"k(\d+)_s(\d+)", r["cell"])
            certified = r["certified"] == "True"
            soc = r["soc_step_kwh"].rstrip("0").rstrip(".")
            row = base_row()
            row.update({
                "group": "CG_SENSITIVITY",
                "cell_id": f"{r['cell']}_soc{soc}_blk{r['block_min']}",
                "phase": "CG_SENSITIVITY",
                "scale": r["scale"], "sel_rep": m.group(2),
                "soc_step": r["soc_step_kwh"], "block_min": r["block_min"],
                "status": "completed" if certified else "censored",
                "route_weight": r["route_weight"],
                "min_reduced_cost": r["min_rc"], "certified": r["certified"],
                "n_columns": r["pool_columns"], "iters": r["iterations"],
                "wall_s": str(round(float(r["elapsed_h"]) * 3600)),
                "censor_reason": "" if certified else "uncertified_at_stop",
                "target_fleet": r["target_fleet"],
                "artifact_path": str(MATRIX.relative_to(REPO)), "artifact_sha256": matrix_sha,
                "notes": f"reaches_target={r['reaches_target']}; cg replicate id not "
                         "recorded in committed aggregate; wall_s from elapsed_h "
                         "(2-decimal hours); see records/runs/ll_20260820c/README.md",
            })
            rows.append(row)

    with TARGET.open("a", newline="") as h:
        csv.DictWriter(h, fieldnames=fields, lineterminator="\n").writerows(rows)
    print(f"appended={len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
