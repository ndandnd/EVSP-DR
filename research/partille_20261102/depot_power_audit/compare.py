"""Recompute the fixed-visit panel and summarize v15 native outputs without exposing schedules."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

TOL = 1e-6


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def table(path, delimiter):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def score(records, field):
    deltas = [r[field] for r in records if r[field] is not None]
    return {"paired": len(deltas), "joint_cheaper": sum(d > TOL for d in deltas),
            "median_fixed_minus_joint": median(deltas)}


def margin(fixed, joint):
    return fixed - joint if fixed is not None and joint is not None else None


def fixed_visits(native, duplicate_inputs):
    root = native / "plug_retiming_parx60_20261002"
    manifest = root / "retime_tasks.tsv"
    baseline = native / "plug_retiming_20261001/retime_collected.csv"
    historical = {(r["tag"], r["cell"], r["arm"]): r for r in table(baseline, ",")}
    inputs = {(r["tag"],r["cell"],r["arm"]): r for r in duplicate_inputs}
    records = defaultdict(dict)
    statuses = defaultdict(Counter)
    for row in table(manifest, "\t"):
        tag, cell, arm = row["tag"], row["cell"], row["arm"]
        source = root / "results" / tag / cell / (arm + ".json")
        data = read_json(source)
        assert data["label"] == "/".join((tag, cell, arm)) and data["routes_file"] == row["routes"]
        assert data["physics"]["station_kw"] == {"PARX": 60} and data["physics"]["charge_kw"] == 350
        p, n = data["plugs"], data["noplugs"]
        assert p["status"] in (2, 3) and n["status"] in (2, 3)
        statuses[arm][f"{p['status']}/{n['status']}"] += 1
        records[tag, cell][arm] = {"plug_status": p["status"], "noplug_status": n["status"],
            "cost": p.get("cost"), "target_kwh": data["target"], "sha256": digest(source)}
    pairs = []
    for (tag, cell), arms in sorted(records.items()):
        f, j, nt = [arms[a] for a in ("fixed_parx60", "joint_parx60", "fixed_parx60_notarget")]
        assert f["target_kwh"] == j["target_kwh"] and nt["target_kwh"] is None
        assert (f["plug_status"],f["noplug_status"]) == (nt["plug_status"],nt["noplug_status"])
        old_f, old_j = [historical[tag, cell, a] for a in ("fixed", "joint")]
        uniform = [float(r["plugs"]) if int(r["plugs_status"]) == 2 else None for r in (old_f, old_j)]
        pairs.append({"tag": tag, "scenario": cell, "uniform": margin(*uniform),
            "parx60": margin(f["cost"], j["cost"]), "receipts": arms})
    unique = {}
    for row in pairs:
        cell = row["scenario"]
        if cell in unique:
            old = unique[cell]
            for a in ("fixed_parx60", "joint_parx60"):
                first = inputs[old["tag"],cell,a]
                second = inputs[row["tag"],cell,a]
                for field in ("trip_sha256","tariff_sha256","fixed_trip_sequences_sha256","route_file_sha256","terminal_target_kwh"):
                    assert first[field] == second[field], (cell,a,field)
            assert abs(old["parx60"] - row["parx60"]) <= TOL and abs(old["uniform"] - row["uniform"]) <= TOL
        unique[cell] = row
    results = {}
    for unit, rows in (("campaign_records", pairs), ("unique_scenarios", list(unique.values()))):
        both = [r for r in rows if r["parx60"] is not None and r["uniform"] is not None]
        results[unit] = {"total": len(rows), "all_at_each_rate": {key: score(rows, key) for key in ("uniform","parx60")},
            "same_records_at_both_rates": {key: score(both,key) for key in ("uniform","parx60")},
            "changed_joint_win_indicator": sum((r["uniform"] > TOL) != (r["parx60"] > TOL) for r in both)}
    return {"array":911126, "statuses_with_without_plug_rows":statuses, "code_sha256":digest(root/"plug_retime.py"),
            "manifest_sha256":digest(manifest), "baseline_table_sha256":digest(baseline), "win_tolerance":TOL,
            "difference_units":"modeled charging cost, fixed minus joint", "summary":results, "paired_rows":pairs}


def v15(native):
    root = native / "capacity_v15_20261002"
    manifest = root / "tasks.tsv"
    rows = []
    for task in table(manifest, "\t"):
        family, case, arm = [task[k] for k in ("family","case","arm")]
        dest = root / "results" / family / case / arm
        filename = "summary.json" if family == "chain" else "bound_report.json"
        completed = [p for p in dest.glob("job*/" + filename) if "physics" in read_json(p)]
        assert len(completed) == 1
        source = completed[0]
        data = read_json(source)
        s = data if family == "chain" else read_json(source.parent/"conservative/summary.json")
        phys = s["physics"]
        if family == "chain":
            fleet = s["fleet_stage"].get("buses")
            cost = (s.get("charging_stage") or {}).get("cost")
        else:
            fleet = data["upper_bound"]["fleet"]
            cost = data["upper_bound"]["charging_cost"]
        rows.append({"family":family, "case":case, "arm":arm, "fleet":fleet, "pool_charging":cost,
            "pricing_certified":s["cg_pricing_certified"], "weighted_lp_lower_bound":s.get("weighted_full_graph_capacity_lp_lower_bound"),
            "lp_route_weight":s["last_iteration"]["route_weight"], "target":s["terminal_target_kwh"],
            "depot_kw":phys.get("station_kw",{}).get("PARX",phys["charge_kw"]), "opportunity_kw":phys["charge_kw"],
            "master":phys["coverage"], "source_sha256":digest(source),
            "input_sha256":sorted(s["input_hashes"].values())})
    return {"manifest_sha256":digest(manifest), "array":920164, "code":"3a0ab6a8", "rows":rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("native",type=Path,help="directory containing the three native campaign directories")
    parser.add_argument("--duplicates",type=Path,default=Path(__file__).with_name("duplicate_inputs.json"))
    parser.add_argument("--out",type=Path,default=Path(__file__).with_name("aggregate.json"))
    args = parser.parse_args()
    result = {"retiming":fixed_visits(args.native,read_json(args.duplicates)), "v15":v15(args.native)}
    assert len(result["retiming"]["paired_rows"]) == 104 and len(result["v15"]["rows"]) == 34
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
