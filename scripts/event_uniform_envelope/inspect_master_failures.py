#!/usr/bin/env python3
"""Extract exact-CG restricted-master failures and their last valid LPs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


BUS_COST = 100_000.0


def load_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read JSON object: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"JSON artifact is not an object: {path}")
    return value


def master_errors(path: Path) -> list[dict]:
    errors = []
    if not path.is_file():
        return errors
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                # An interrupted append may damage only the final JSONL row.
                if any(rest.strip() for rest in handle):
                    raise SystemExit(
                        f"malformed telemetry before EOF at {path}:{line_number}"
                    )
                break
            if (
                record.get("record_type") == "phase"
                and record.get("phase") == "master_attempt"
                and record.get("outcome") == "error"
            ):
                errors.append(record)
    return errors


def text(value) -> str:
    return "" if value is None else str(value)


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int)
    args = parser.parse_args()

    root = args.resume_root.expanduser().resolve()
    plan = load_object(root / "execution_plan.json")
    matrix_path = root / "matrix.tsv"
    rows = []
    with matrix_path.open(newline="", encoding="utf-8") as handle:
        for item in csv.DictReader(handle, delimiter="\t"):
            status_path = Path(item["resume_status"])
            status = load_object(status_path)
            if status.get("stop_reason") != "master_failed":
                continue
            final = status.get("final") or {}
            final_lp = status.get("final_lp") or {}
            telemetry_path = Path(str(status_path) + ".phase-telemetry.jsonl")
            errors = master_errors(telemetry_path)
            final_iteration = final.get("iter")
            main_errors = [
                record for record in errors
                if (record.get("details") or {}).get("purpose")
                != "final_resolve"
            ]
            final_resolve_errors = [
                record for record in errors
                if (record.get("details") or {}).get("purpose")
                == "final_resolve"
            ]
            maximum_main = max(
                (record.get("iteration") for record in main_errors
                 if isinstance(record.get("iteration"), int)),
                default=None,
            )
            maximum_resolve = max(
                (record.get("iteration") for record in final_resolve_errors
                 if isinstance(record.get("iteration"), int)),
                default=None,
            )
            final_errors = [
                record for record in main_errors
                if record.get("iteration") == maximum_main
            ] + [
                record for record in final_resolve_errors
                if record.get("iteration") == maximum_resolve
            ]
            objective = number(final_lp.get("objective"))
            route_weight = number(final_lp.get("route_weight"))
            charging_component = (
                objective - BUS_COST * route_weight
                if objective is not None and route_weight is not None else None
            )
            details = [record.get("details") or {} for record in final_errors]
            log_out = sorted(root.glob(f"logs/*_{item['local_index']}.out"))
            log_err = sorted(root.glob(f"logs/*_{item['local_index']}.err"))
            rows.append({
                "local_index": item["local_index"],
                "source_panel_index": item["source_panel_index"],
                "target_fleet": item["target_fleet"],
                "cell": item["cell"],
                "representation_id": item["representation_id"],
                "stop_reason": status.get("stop_reason"),
                "cumulative_wall_h": float(status.get("wall_s", 0.0)) / 3600,
                "iterations": status.get("iterations"),
                "columns": status.get("columns"),
                "last_completed_iteration": final_iteration,
                "last_completed_lp_objective": final.get("lp_obj"),
                "last_completed_route_weight": final.get("route_weight"),
                "last_completed_artificials": final.get("artificials"),
                "last_completed_min_rc": final.get("min_rc"),
                "preserved_lp_source": status.get("final_lp_source"),
                "preserved_lp_iteration": final_lp.get("iteration"),
                "preserved_lp_pool_columns": final_lp.get("pool_columns"),
                "preserved_lp_objective": objective,
                "preserved_lp_route_weight": route_weight,
                "preserved_lp_charging_component": charging_component,
                "preserved_lp_artificial_total": final_lp.get("artificial_total"),
                "preserved_lp_master_method": final_lp.get("master_method"),
                "preserved_lp_max_row_violation": final_lp.get("max_row_violation"),
                "preserved_lp_max_bound_violation": final_lp.get("max_bound_violation"),
                "preserved_lp_feasibility_tolerance": final_lp.get(
                    "feasibility_tolerance"
                ),
                "final_failed_attempts": len(final_errors),
                "failed_purposes": " | ".join(
                    text(detail.get("purpose") or "main")
                    for detail in details
                ),
                "failed_methods": " | ".join(text(detail.get("method")) for detail in details),
                "failed_messages": " | ".join(text(detail.get("error")) for detail in details),
                "failed_durations_s": " | ".join(
                    text(record.get("duration_s")) for record in final_errors
                ),
                "failed_time_limits_s": " | ".join(
                    text(detail.get("time_limit_s")) for detail in details
                ),
                "telemetry_path": str(telemetry_path),
                "stdout_logs": " | ".join(str(path) for path in log_out),
                "stderr_logs": " | ".join(str(path) for path in log_err),
            })

    if args.expected is not None and len(rows) != args.expected:
        raise SystemExit(
            f"expected {args.expected} master failures, found {len(rows)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['cell']} (k={row['target_fleet']}): "
            f"wall={row['cumulative_wall_h']:.3f}h, "
            f"iteration={row['last_completed_iteration']}, "
            f"columns={row['columns']}"
        )
        print(
            "  last valid LP: "
            f"objective={row['preserved_lp_objective']}, "
            f"route_weight={row['preserved_lp_route_weight']}, "
            f"charging_component={row['preserved_lp_charging_component']}, "
            f"artificials={row['preserved_lp_artificial_total']}, "
            f"method={row['preserved_lp_master_method']}"
        )
        print(
            "  residuals: "
            f"row={row['preserved_lp_max_row_violation']}, "
            f"bound={row['preserved_lp_max_bound_violation']}, "
            f"audit_tolerance={row['preserved_lp_feasibility_tolerance']}"
        )
        print(f"  last completed pricing min_rc={row['last_completed_min_rc']}")
        for purpose, method, message in zip(
            row["failed_purposes"].split(" | "),
            row["failed_methods"].split(" | "),
            row["failed_messages"].split(" | "),
        ):
            if method or message:
                print(f"  failed {purpose}/{method}: {message}")
    print(f"CSV: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
