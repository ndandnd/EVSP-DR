#!/usr/bin/env python3
"""Generate two plain scientific figures for the EVSP-DR bottlenecks.

The inputs are read from the checked-in evidence records.  The script writes
only the two PNGs, their short captions, and a single provenance JSON next to
this file.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

MONITOR_JSON = PROJECT_ROOT / "outputs/post_meeting_20260910/monitor/20260912T052658Z.json"
OVERNIGHT_REPORT = PROJECT_ROOT / "outputs/overnight_extension_20260912/RESULTS_20260912T052658Z.md"
OVERNIGHT_TIMING_CODE = PROJECT_ROOT / "outputs/overnight_extension_20260912/exact_pricer_bounded.py"
OVERNIGHT_MANIFEST = PROJECT_ROOT / "outputs/overnight_extension_20260912/manifest.json"
CAPACITY_RECORDS = PROJECT_ROOT / "outputs/parallel_research_20260911/capacity_deadline5_completed/records.json"
CAPACITY_README = PROJECT_ROOT / "outputs/parallel_research_20260911/capacity_deadline5_completed/README.md"
CAPACITY_TIMEOUT_EVIDENCE = PROJECT_ROOT / "outputs/research_register/CAPACITY_PILOT_TIMEOUT_EVIDENCE_20260910.json"
EXECUTION_ISSUES = PROJECT_ROOT / "outputs/research_register/EXECUTION_ISSUES_20260910.md"
CAPACITY_PRICER_COMMIT = "253588e9b22d68fcbc67cb56bc3eb30cbb0e16b6"
CAPACITY_PRICER_TREE_PATH = "src/run_capacity_speed_event_cg.py"
CAPACITY_PRICER_BLOB_SHA256 = "86c2b9e797d99a4321157e6ce9dd64874e285e59551748395f28599f2f4d8377"

WARM_CASES = [
    "w1_k07",
    "w2_k09",
    "w3_k11",
    "w3_k12",
    "w4_k10",
    "w5_k11",
    "w6_k11",
]

BLUE = "#2f5d8a"
ORANGE = "#c87532"
GREEN = "#3b7d5b"
PURPLE = "#76558f"
GRID = "#b7c0c9"
TEXT = "#24303a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_record(path: Path, *, note: str | None = None) -> dict:
    result = {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": sha256(path),
    }
    if note is not None:
        result["note"] = note
    return result


def read_warm_records() -> list[dict]:
    snapshot = json.loads(MONITOR_JSON.read_text())
    records = snapshot["campaigns"]["overnight_extension_20260912"]["cg"]
    found: dict[str, dict] = {}
    for record in records:
        if record.get("column_pool_treatment") != "WARM-INHERITED-EVENT":
            continue
        match = re.search(r"/cases/(w\d+_k\d+)/cg\.json$", record.get("path", ""))
        if match is None:
            continue
        case = match.group(1)
        if case not in WARM_CASES:
            continue
        if record.get("stop_reason") != "certified" or record.get("certified_rc_optimal") is not True:
            raise ValueError(f"warm case {case} is not certified in the snapshot")
        audit = record["inherited_event_pool_audit"]
        import_s = float(audit["import_runtime_s"])
        wall_s = float(record["wall_s"])
        if wall_s < import_s:
            raise ValueError(f"wall_s is shorter than import for {case}")
        found[case] = {
            "case": case,
            "import_runtime_s": import_s,
            "import_minutes": import_s / 60.0,
            "wall_s": wall_s,
            "remaining_cg_s": wall_s - import_s,
            "remaining_cg_minutes": (wall_s - import_s) / 60.0,
            "attempt_wall_s": float(record["attempt_wall_s"]),
            "iterations": int(record["attempt_iterations"]),
            "pool_columns": int(record["final_lp"]["pool_columns"]),
            "source_status_sha256": audit["source_status_sha256"],
            "imported_columns": int(audit["accepted_columns"]),
            "import_rejected_columns": int(audit["rejected_columns"]),
            "import_deadline_reached": bool(audit["import_deadline_reached"]),
            "inherited_duals": bool(audit["inherited_duals"]),
            "inherited_basis": bool(audit["inherited_basis"]),
            "inherited_lp_certificate": bool(audit["inherited_lp_certificate"]),
        }
    if list(found) != WARM_CASES:
        raise ValueError(f"expected warm cases {WARM_CASES}, found {list(found)}")
    return [found[case] for case in WARM_CASES]


def read_capacity_record() -> dict:
    records = json.loads(CAPACITY_RECORDS.read_text())
    matches = [
        record
        for record in records
        if record.get("phase") == "cg"
        and record.get("path", "").endswith("/e1_short_k3__capacity/cg.json")
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one k3 capacity CG record, found {len(matches)}")
    record = matches[0]
    result = record["result"]
    iteration = result["last_iteration"]
    if int(iteration["iteration"]) != 3:
        raise ValueError("k3 capacity telemetry is not from iteration 3")
    return {
        "case": "e1_short_k3__capacity",
        "arm": result["arm"],
        "status": result["status"],
        "stop_reason": result["stop_reason"],
        "certified_rc_optimal": bool(result["certified_rc_optimal"]),
        "record_path_on_cluster": record["path"],
        "record_sha256": record["sha256"],
        "execution_commit": result["provenance"]["git_commit"],
        "iteration": int(iteration["iteration"]),
        "lp_solve_s": float(iteration["lp_solve_s"]),
        "pricing_s": float(iteration["pricing_s"]),
        "pricing_hours": float(iteration["pricing_s"]) / 3600.0,
        "pricing_minutes": float(iteration["pricing_s"]) / 60.0,
        "priced_trip_count": int(iteration["priced_trip_count"]),
        "lp_columns": int(iteration["lp_columns"]),
        "lp_rows": int(iteration["lp_rows"]),
        "lp_nonzeros": int(iteration["lp_nonzeros"]),
        "pool_columns": int(iteration["pool_columns"]),
        "nonzero_capacity_duals": int(iteration["nonzero_capacity_duals"]),
        "route_weight": float(iteration["route_weight"]),
        "min_reduced_cost": float(iteration["min_reduced_cost"]),
        "final_weighted_objective": float(result["final"]["objective"]),
        "final_pool_columns": int(result["final"]["pool_columns"]),
    }


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.edgecolor": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def draw_warm_chart(data: list[dict], destination: Path) -> None:
    chain_labels = {
        "w1_k07": "Chain 1, k=7",
        "w2_k09": "Chain 2, k=9",
        "w3_k11": "Chain 3, k=11",
        "w3_k12": "Chain 3, k=12",
        "w4_k10": "Chain 4, k=10",
        "w5_k11": "Chain 5, k=11",
        "w6_k11": "Chain 6, k=11",
    }
    labels = [chain_labels[row["case"]] for row in data]
    import_minutes = np.array([row["import_minutes"] for row in data])
    remaining_minutes = np.array([row["remaining_cg_minutes"] for row in data])
    y = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(8.4, 4.75))
    ax.barh(
        y,
        import_minutes,
        color=BLUE,
        edgecolor="white",
        linewidth=0.7,
        label="Checking saved routes",
    )
    ax.barh(
        y,
        remaining_minutes,
        left=import_minutes,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.7,
        label="Finding new routes and other CG work",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Minutes")
    ax.set_ylabel("Warm case")
    ax.set_xlim(0, float(np.max(import_minutes + remaining_minutes)) * 1.12)
    ax.grid(axis="x", color=GRID, linestyle=":", linewidth=0.8)
    ax.set_axisbelow(True)
    for idx, (left, right) in enumerate(zip(import_minutes, remaining_minutes)):
        ax.text(
            left / 2,
            idx,
            f"{left:.1f}",
            va="center",
            ha="center",
            color="white",
            fontsize=8.5,
        )
        ax.text(
            left + right / 2,
            idx,
            f"{right:.1f}",
            va="center",
            ha="center",
            color="white",
            fontsize=8.5,
        )
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_capacity_chart(data: dict, destination: Path) -> None:
    values = [data["lp_solve_s"], data["pricing_s"]]
    labels = ["LP solve", "Search for a new route"]
    y = np.arange(len(values))

    fig, ax = plt.subplots(figsize=(7.6, 3.55))
    ax.barh(
        y,
        values,
        color=[PURPLE, GREEN],
        edgecolor="white",
        linewidth=0.7,
        height=0.56,
    )
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Seconds (log scale)")
    ax.set_xlim(0.003, 100000)
    ax.grid(axis="x", which="major", color=GRID, linestyle=":", linewidth=0.8)
    ax.grid(axis="x", which="minor", visible=False)
    ax.set_axisbelow(True)
    annotations = ["0.006 s", "7.15 hours"]
    for idx, (value, annotation) in enumerate(zip(values, annotations)):
        ax.text(value * 1.16, idx, annotation, va="center", ha="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_caption(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n")


def build_provenance(warm_data: list[dict], capacity_data: dict) -> dict:
    return {
        "schema": "evsp-dr-bottleneck-figures-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "path": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
            "sha256": sha256(Path(__file__).resolve()),
            "python": sys.executable,
            "matplotlib": matplotlib.__version__,
        },
        "figures": {
            "bottleneck_warm_import_vs_remaining.png": {
                "caption_path": "outputs/document_recovery_20260912/bottleneck_warm_import_vs_remaining.md",
                "sources": [
                    source_record(MONITOR_JSON, note="Seven certified warm CG records and cumulative wall_s."),
                    source_record(OVERNIGHT_REPORT, note="Defines the seven-case certified snapshot and reports the warm import table."),
                    source_record(OVERNIGHT_TIMING_CODE, note="Run clock and cumulative wall_s semantics."),
                    source_record(OVERNIGHT_MANIFEST, note="Campaign case and execution identity."),
                ],
                "data": warm_data,
                "transform": "remaining_cg_minutes = (wall_s - inherited_event_pool_audit.import_runtime_s) / 60",
                "wall_s_semantics": {
                    "includes_import_before_subtraction": True,
                    "basis": "t0 is set before inherited event-pool import; wall_s is _cumulative_elapsed_s().",
                    "residual_scope": "Residual includes network/setup and all later CG work, including route search, LP solves, and finalization.",
                },
                "interpretation_boundary": "These are phase timings. They are not a paired speedup comparison; older full-pool cases differ in treatment and scope.",
            },
            "bottleneck_capacity_k3_lp_vs_pricing.png": {
                "caption_path": "outputs/document_recovery_20260912/bottleneck_capacity_k3_lp_vs_pricing.md",
                "sources": [
                    source_record(CAPACITY_RECORDS, note="Exact iteration-3 LP and pricing telemetry."),
                    source_record(CAPACITY_README, note="Rounded 7.15-hour/0.00563-second interpretation and proof boundary."),
                    source_record(CAPACITY_TIMEOUT_EVIDENCE, note="Scheduler timeout evidence; it contains no per-iteration timing fields used for the plotted values."),
                    source_record(EXECUTION_ISSUES, note="Register wording that identifies the same third iteration."),
                    {
                        "path": f"git:{CAPACITY_PRICER_COMMIT}:{CAPACITY_PRICER_TREE_PATH}",
                        "sha256": CAPACITY_PRICER_BLOB_SHA256,
                        "note": "Immutable pricer source showing pricing_s is timed around one network.min_reduced_cost_route call.",
                    },
                ],
                "data": capacity_data,
                "comparison_semantics": {
                    "same_iteration": True,
                    "iteration": capacity_data["iteration"],
                    "lp_solve_s": "The restricted-master LP solve time recorded for iteration 3.",
                    "pricing_s": "One call to network.min_reduced_cost_route(...) in iteration 3, timed around that call; not an aggregate over the run.",
                    "priced_trip_count": "The selected candidate route covers 12 trips; it is not a count of separately timed pricing calls.",
                    "aggregate_or_onecall": "onecall",
                },
                "interpretation_boundary": "The run stopped at pricing_deadline and has no pricing certificate; the chart explains elapsed work only.",
            },
        },
    }


def main() -> None:
    configure_matplotlib()
    warm_data = read_warm_records()
    capacity_data = read_capacity_record()

    warm_png = OUT_DIR / "bottleneck_warm_import_vs_remaining.png"
    capacity_png = OUT_DIR / "bottleneck_capacity_k3_lp_vs_pricing.png"
    draw_warm_chart(warm_data, warm_png)
    draw_capacity_chart(capacity_data, capacity_png)

    write_caption(
        OUT_DIR / "bottleneck_warm_import_vs_remaining.md",
        """Seven certified warm cases from the 2026-09-12 overnight snapshot. “Checking saved routes” is the measured inherited-event-pool import. The run clock starts before that import, so `wall_s` includes it; the second segment is `wall_s - import_runtime_s` and includes the remaining setup and CG work. These phase timings do not claim a paired speedup because older full-pool cases use a different treatment.""",
    )
    write_caption(
        OUT_DIR / "bottleneck_capacity_k3_lp_vs_pricing.md",
        """Capacity k=3, iteration 3: the same telemetry record reports an LP solve of 0.005633926019072533 seconds and one new-route pricing call of 25722.64842124097 seconds (7.145180117 hours). The pricing number is one `network.min_reduced_cost_route` call, not an aggregate across the run; `priced_trip_count=12` describes the returned candidate route. The run later stopped at a pricing deadline without a pricing certificate.""",
    )

    provenance = build_provenance(warm_data, capacity_data)
    provenance["outputs"] = {
        "bottleneck_warm_import_vs_remaining.png": {
            "sha256": sha256(warm_png),
            "bytes": warm_png.stat().st_size,
        },
        "bottleneck_capacity_k3_lp_vs_pricing.png": {
            "sha256": sha256(capacity_png),
            "bytes": capacity_png.stat().st_size,
        },
    }
    (OUT_DIR / "bottleneck_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=False) + "\n"
    )


if __name__ == "__main__":
    main()
