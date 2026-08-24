#!/usr/bin/env python3
"""Build Panel B uniform arms from certified Panel A event wall times."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-a", type=Path, required=True)
    parser.add_argument("--panel-b", type=Path, required=True)
    parser.add_argument("--execution-repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    a_root = args.panel_a.resolve()
    b_root = args.panel_b.resolve()
    execution_repo = args.execution_repo.resolve()

    gate = json.loads((a_root / "panel_b_gate.json").read_text())
    if gate.get("eligible") is not True:
        raise SystemExit("Panel A event gate is not satisfied")

    rows_a = list(csv.reader((a_root / "matrix.tsv").open(), delimiter="\t"))
    by_cell = {}
    for fields in rows_a:
        by_cell.setdefault(fields[1], []).append(fields)

    rows_b = []
    event_sources = {}
    index = 0
    for cell in sorted(by_cell):
        event_rows = [fields for fields in by_cell[cell] if fields[5] == "event"]
        if len(event_rows) != 1:
            raise SystemExit(f"{cell}: expected one event row")
        event_rep = event_rows[0][4]
        event_status = a_root / "cg" / f"A__{cell}__{event_rep}.json"
        status = json.loads(event_status.read_text())
        if (
            status.get("certified_rc_optimal") is not True
            or status.get("stop_reason") != "certified"
        ):
            raise SystemExit(f"{cell}: event source is not certified")
        event_wall_s = float(status["wall_s"])
        runner_limit_s = math.ceil(event_wall_s) + 60
        event_sources[cell] = {
            "status": str(event_status),
            "status_sha256": sha256(event_status),
            "journal": status["columns_journal"],
            "wall_s": event_wall_s,
            "peak_rss_mb": status.get("peak_rss_mb"),
            "iterations": status.get("iterations"),
        }
        for fields in by_cell[cell]:
            _, _, target, instance_csv, rep, time_model, soc, block = fields
            if time_model != "uniform":
                continue
            rows_b.append([
                index, cell, target, instance_csv, rep, soc, block,
                repr(event_wall_s), runner_limit_s,
            ])
            index += 1
    if len(rows_b) != 45:
        raise SystemExit(f"expected 45 Panel B arms, found {len(rows_b)}")

    b_root.mkdir(parents=True, exist_ok=False)
    for name in ("cg", "frozen", "mip", "target", "logs"):
        (b_root / name).mkdir()
    with (b_root / "matrix.tsv").open("w", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(rows_b)

    code_paths = (
        "src/exact_pricer_expanded.py",
        "src/event_pricer_network.py",
        "src/freeze_exact_cg_at_wall.py",
    )
    (b_root / "execution_plan.json").write_text(json.dumps({
        "schema": "evsp-dr-event-uniform-panel-b-execution-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "producer_commit": args.commit,
        "panel_a_root": str(a_root),
        "panel_a_event_sources": event_sources,
        "uniform_arms": 45,
        "scientific_budget": "paired certified Panel A event wall_s",
        "runner_limit": "ceil(event_wall_s)+60 seconds",
        "published_pool": "last complete iteration at or before event wall_s",
        "raw_pool_only": True,
        "known_routes_injected": False,
        "code_sha256": {
            path: sha256(execution_repo / path) for path in code_paths
        },
    }, indent=2, sort_keys=True) + "\n")
    print(f"prepared 45 Panel B arms under {b_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
