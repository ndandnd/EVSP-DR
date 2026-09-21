"""Rebuild compact, independently checked evidence from the archived source files."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "source_manifest.json").read_text())
BY_PATH = {r["local_relative_path"]: r for r in MANIFEST}
for item in MANIFEST:
    assert hashlib.sha256((ROOT / item["local_relative_path"]).read_bytes()).hexdigest() == item["sha256"]


def write_csv(name, rows):
    with (ROOT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stages(path):
    lines = path.read_text().splitlines()
    summaries = []
    for idx, line in enumerate(lines):
        m = re.match(r"Best objective ([\deE+.-]+), best bound ([\deE+.-]+), gap ([\d.]+)%", line)
        if not m:
            continue
        segment_start = next(i for i in range(idx, -1, -1) if lines[i].startswith("Optimize a model"))
        explored_idx = next(i for i in range(idx, segment_start, -1) if lines[i].startswith("Explored "))
        n = re.match(r"Explored (\d+) nodes .* in ([\d.]+) seconds", lines[explored_idx])
        termination = next(lines[i] for i in range(idx, segment_start, -1) if lines[i].startswith(("Optimal solution", "Time limit")))
        summaries.append({"objective": float(m[1]), "bound": float(m[2]), "gap_percent": float(m[3]),
                          "nodes": int(n[1]), "seconds": float(n[2]), "termination": termination,
                          "model_line": segment_start + 1, "explored_line": explored_idx + 1,
                          "objective_bound_line": idx + 1})
    assert len(summaries) == 2, path
    return summaries


all_rows = []
for path in sorted(ROOT.rglob("*gurobi.log")):
    result_path = path.with_name("result.json")
    if not result_path.exists():
        continue
    result = json.loads(result_path.read_text())
    ss = stages(path)
    assert abs(ss[0]["bound"] - result["fleet_bound"]) < 1e-6, path
    assert ss[0]["objective"] == result["stage2_fleet_cap"], path
    rel = str(path.relative_to(ROOT))
    row = {"experiment": rel.split("/")[0], "case_arm": "/".join(rel.split("/")[1:-1]),
           "fleet_buses": int(ss[0]["objective"]), "fleet_bound": ss[0]["bound"],
           "fleet_proven": result["fleet_proven"], "fleet_termination": ss[0]["termination"],
           "fleet_seconds": ss[0]["seconds"], "fleet_nodes": ss[0]["nodes"],
           "fleet_proof_line": ss[0]["objective_bound_line"],
           "charging_termination": ss[1]["termination"], "charging_cost": ss[1]["objective"],
           "charging_bound": ss[1]["bound"], "charging_gap_percent": ss[1]["gap_percent"],
           "charging_seconds": ss[1]["seconds"], "charging_proof_line": ss[1]["objective_bound_line"],
           "final_buses": result["buses"], "physical_replay_validated": result["physical_replay_validated"],
           "duplicate_trip_removal_validated": result["duplicate_trip_removal_validated"],
           "shared_capacity_validated": result["cross_route_charger_capacity_validated"],
           "source_cg_path": result["source_result"], "source_cg_sha256": result["source_result_sha256"],
           "source_journal_path": result["source_journal"], "source_journal_sha256": result["source_journal_sha256"],
           "execution_commit": result["mip_provenance"]["git_commit"],
           "local_log": str(path), "remote_log": BY_PATH[rel]["remote_path"],
           "log_sha256": BY_PATH[rel]["sha256"]}
    all_rows.append(row)
write_csv("solver_summary.csv", all_rows)
for group in ["k8_witness", "k15_12h", "pilot", "c1_followup"]:
    write_csv(group + "_summary.csv", [r for r in all_rows if r["experiment"] == group])

# Recalculate reduced costs and the LP-to-integer objective decomposition from
# saved final duals, not from the human-written research summary.
analysis = json.loads((ROOT / "witness_analysis.json").read_text())
remote_identity = json.loads((ROOT / "pool_identity_remote_audit.json").read_text())
mechanism_rows = []
route_rows = []
for case, data in sorted(analysis["cases"].items()):
    cg = json.loads((ROOT / "k8_sources" / case / "fresh_cg.json").read_text())
    lp = cg["final_lp"]
    dual = {int(k): v for k, v in lp["trip_duals"].items()}
    witness = data["witness_routes"]
    assert cg["certified_rc_optimal"] is True
    assert data["physics_match_warm"] and data["provenance_match_warm"]
    for key, expected in data["hashes"].items():
        assert remote_identity[case][key]["sha256"] == expected, (case, key)
    assert remote_identity[case]["aug_prefix_equals_fresh"]
    assert remote_identity[case]["appended_lines"] == data["witness_rc_summary"]["n_appended"]
    for arm, prefix in [("control", "fresh"), ("augmented", "aug")]:
        row = next(r for r in all_rows if r["experiment"] == "k8_witness" and r["case_arm"].startswith(case + "/" + arm + "/"))
        assert row["source_cg_sha256"] == data["hashes"][prefix + "_cg"]
        assert row["source_journal_sha256"] == data["hashes"][prefix + "_journal"]
    cover = Counter(t for w in witness for t in w["trips"])
    assert set(cover) == set(cg["trip_ids"])
    rc_sum = 0
    for w in witness:
        rc = w["cost"] - sum(dual[t] for t in w["trips"])
        assert abs(rc - w["rc_fresh_duals"]) < 1e-7
        rc_sum += rc
        route_rows.append({"case": case, "witness_index": w["witness_index"],
                           "origin": w["origin"], "n_trips": w["n_trips"], "cost": w["cost"],
                           "rc_recomputed": rc, "trip_set_already_present": w["fresh_pool_identical_trip_set"],
                           "appended": w["appended_to_augmented_pool"]})
    overcoverage_dual = sum(dual[t] * (n - 1) for t, n in cover.items())
    total_cost = sum(w["cost"] for w in witness)
    residual = total_cost - lp["objective"] - rc_sum - overcoverage_dual
    assert abs(residual) < 1e-6
    positives = lp["positive_routes"]
    mechanism_rows.append({"case": case, "trips": data["n_trips"],
                           "pool_columns": cg["columns"], "lp_route_weight": lp["route_weight"],
                           "lp_weighted_objective": lp["objective"], "lp_positive_columns": len(positives),
                           "lp_fractional_positive_columns": sum(1e-7 < r["value"] < 1 - 1e-7 for r in positives),
                           "lp_max_column_weight": max(r["value"] for r in positives),
                           "witness_buses": len(witness), "witness_cost": total_cost,
                           "witness_cost_minus_lp": total_cost - lp["objective"],
                           "witness_rc_sum": rc_sum, "overcoverage_dual_value": overcoverage_dual,
                           "decomposition_residual": residual, "overcovered_trips": sum(n > 1 for n in cover.values()),
                           "missing_witness_trip_sets": sum(not w["fresh_pool_identical_trip_set"] for w in witness),
                           "inherited_witness_routes": sum(w["origin"] == "inherited_event_pool_replayed_in_child_graph" for w in witness),
                           "positive_rc_above_tolerance": sum(w["rc_fresh_duals"] > 1e-4 for w in witness)})
write_csv("mechanism_summary.csv", mechanism_rows)
write_csv("witness_route_audit.csv", route_rows)

k15_manifest = json.loads((ROOT / "k15_manifest.json").read_text())
for cell in k15_manifest["cells"]:
    row = next(r for r in all_rows if r["experiment"] == "k15_12h" and r["case_arm"] == cell["cell"])
    assert row["source_cg_sha256"] == cell["result_sha256"]
    assert row["source_journal_sha256"] == cell["journal_sha256"]
    assert row["execution_commit"] == k15_manifest["runner_commit"]

# Keep the original logs untouched; this navigable index is a reading aid.
excerpts = ["# Exact Gurobi log excerpts", "", "The complete logs are adjacent. Line numbers below refer to the unchanged complete source log.", ""]
for row in all_rows:
    path = Path(row["local_log"])
    lines = path.read_text().splitlines()
    excerpts += ["## " + row["experiment"] + "/" + row["case_arm"], "", f"Local: `{path}`", "",
                 f"Unicorn: `{row['remote_log']}`", "", f"SHA-256: `{row['log_sha256']}`", ""]
    for name, key in [("Fleet stage", "fleet_proof_line"), ("Charging stage", "charging_proof_line")]:
        end = row[key]
        excerpts += ["**" + name + "**", "", "```text"]
        excerpts += [f"{i+1}: {lines[i]}" for i in range(max(0, end - 7), end)]
        excerpts += ["```", ""]
(ROOT / "LOG_EXCERPTS.md").write_text("\n".join(excerpts))
print(json.dumps({"source_files_hash_verified": len(MANIFEST), "two_stage_solves": len(all_rows),
                  "k8_identity_and_dual_checks_passed": len(mechanism_rows),
                  "k15_manifest_checks_passed": len(k15_manifest["cells"])}))
