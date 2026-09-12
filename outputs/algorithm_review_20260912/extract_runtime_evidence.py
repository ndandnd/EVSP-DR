"""Reconcile the review's case metrics against its frozen collector snapshot.

Read-only by default: python extract_runtime_evidence.py SNAPSHOT
The snapshot hash and every selected case metric must match runtime_evidence.json.
"""
import hashlib
import json
from pathlib import Path
import re
import sys


def extract(snapshot):
    rows = []
    for campaign in (
        "covering_complement75", "covering_rerun9", "overnight_extension_20260912"
    ):
        group = snapshot["campaigns"][campaign]
        phases = {
            v["path"].removesuffix(".phase-telemetry.jsonl"): v
            for v in group["phases"]
        }
        for cg in group["cg"]:
            if cg.get("certified_rc_optimal") is not True:
                continue
            phase = phases.get(cg["path"])
            if not phase:
                continue
            d = phase["duration_s_by_phase"]
            match = re.search(r"k(\d+)_p0?(\d+)", cg["csv"])
            network = cg.get("network_metrics") or {}
            final = cg.get("final") or {}
            rows.append({
                "campaign": campaign,
                "path": cg["path"],
                "csv": cg["csv"],
                "k": int(match[1]) if match else None,
                "chain": int(match[2]) if match else None,
                "wall_s": cg["wall_s"],
                "cg_iterations": cg["attempt_iterations"],
                "network_build_or_load_s": d.get("network_build", 0),
                "network_cache_hit": network.get("cache_hit", False),
                "graph_arcs": network.get("dag_arcs"),
                "graph_nodes": network.get("dag_nodes"),
                "packed_arc_bytes": network.get("packed_arc_bytes"),
                "pricing_batch_inclusive_s": d.get("pricing_extra_columns", 0),
                "pricing_shortest_path_s": d.get("pricing_shortest_path", 0),
                "pricing_enrichment_exclusive_s": (
                    d.get("pricing_extra_columns", 0)
                    - d.get("pricing_shortest_path", 0)
                ),
                "incidence_s": d.get("incidence_construction", 0),
                "master_s": d.get("master_attempt", 0),
                "iteration_and_journal_fsync_s": (
                    d.get("iteration_log_fsync", 0) + d.get("journal_fsync", 0)
                ),
                "weighted_lp": final.get("lp_obj"),
                "fractional_route_weight": final.get("route_weight"),
                "stop_reason": cg["stop_reason"],
                "source_provenance": cg.get("provenance") or {},
            })
    return rows


if __name__ == "__main__":
    expected = json.loads(
        Path(__file__).with_name("runtime_evidence.json").read_text()
    )
    source = Path(sys.argv[1] if len(sys.argv) > 1 else expected["snapshot_path"])
    payload = source.read_bytes()
    observed_hash = hashlib.sha256(payload).hexdigest()
    if observed_hash != expected["snapshot_sha256"]:
        raise SystemExit("Snapshot identity differs from the reviewed snapshot")
    rows = extract(json.loads(payload))
    if rows != expected["records"]:
        raise SystemExit("Extracted records differ from the review evidence")
    if len(rows) != 98 or len({row["path"] for row in rows}) != 98:
        raise SystemExit("Unexpected record count or duplicate source paths")
    if any(row["pricing_enrichment_exclusive_s"] < 0 for row in rows):
        raise SystemExit("Overlapping timing counters are inconsistent")
    print(json.dumps({
        "verified": True,
        "snapshot_sha256": observed_hash,
        "matched_case_records": len(rows),
        "fresh_covering_records": sum(
            r["campaign"] != "overnight_extension_20260912" for r in rows
        ),
        "timing_interpretation": "pricing_batch includes shortest_path",
    }, indent=2))
