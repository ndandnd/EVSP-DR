"""One source-bound table for the original k16–32 chain searches.

Keep separate longer MIPs out of this fixed-budget table. Never promote an
operational checkpoint or scheduler completion to a scientific endpoint.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


CAMPAIGNS = tuple(f"chain_extension_{day}" for day in (20260913, 20260914, 20260915)) + ("chain_extension_31_32_20260915",)


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def endpoints(items):
    result = {}
    for item in items:
        parts = Path(item["path"]).parts
        case = parts[parts.index("cases") + 1]
        if case in result:
            raise ValueError(f"Multiple published endpoints for {case}")
        result[case] = item
    return result


def summarize(source):
    rows, settings, checks = [], None, []
    for name in CAMPAIGNS:
        campaign = source["campaigns"].get(name)
        if campaign is None:
            continue
        workflow = campaign["workflow"]
        manifest = workflow["manifest.json"]
        if settings is None:
            settings = manifest["scientific_settings"]
        assert settings == manifest["scientific_settings"], "Scientific settings differ"
        assert settings["master_sense"] == "cover"
        assert settings["mip_seconds"] == 3600
        assert settings["stage1_seconds"] == 1800
        cg, mip = endpoints(campaign["cg"]), endpoints(campaign["mip"])
        # Use actual submitted cases, not all generated future inputs.
        for cid, jobs in workflow["case_jobs.json"].items():
            case = manifest["cases"][cid]
            row = dict(campaign=name, case_id=cid, chain=case["chain"],
                       target_buses=case["k"], trip_count=case["trip_count"],
                       input_sha256=case["input_sha256"],
                       recorded_cg_commit=manifest["execution_commit"],
                       manifest_mip_commit=manifest["mip_execution_commit"],
                       cg_job=jobs["cg"], mip_job=jobs["mip"],
                       cg_result_collected=cid in cg, mip_result_collected=cid in mip)
            if cid in cg:
                item = cg[cid]
                assert item["csv"] == case["csv"]
                assert item["provenance"]["instance_sha256"] == case["input_sha256"]
                assert item["provenance"]["git_commit"] == manifest["execution_commit"]
                assert item["master_sense"] == "cover"
                final = item.get("final_lp") or {}
                assert final.get("objective") is not None, "Final pool re-solve missing"
                assert final.get("route_weight") is not None
                row.update(cg_minutes=item["wall_s"] / 60,
                           cg_pricing_certificate=bool(item["certified_rc_optimal"]),
                           cg_stop_reason=item["stop_reason"],
                           weighted_lp_objective=final["objective"],
                           fractional_route_weight=final["route_weight"],
                           lp_endpoint_source="final_lp pool re-solve",
                           last_pricing_reduced_cost=item["final"].get("min_rc"),
                           cg_path=item["path"], cg_canonical_payload_sha256=sha(item))
                checks.append(dict(case=cid, stage="cg", input_and_commit_match=True))
            if cid in mip:
                item = mip[cid]
                assert item["instance"] == case["csv"]
                assert item["physical_pool_audit"]["input_hashes"]["instance_sha256"] == case["input_sha256"]
                assert item["partitioning"] is False
                assert item["two_stage"]["stage1_time_limit_s"] == 1800
                assert item["two_stage"]["stage2_fleet_constraint"] == "at_most"
                row.update(integer_buses=item["buses"],
                           saved_pool_fleet_bound=item["fleet_bound"],
                           fleet_proved_in_saved_pool=item["fleet_proven"],
                           total_mip_minutes=item["runtime_s"] / 60,
                           fleet_search_minutes=item["two_stage"]["stage1_runtime_s"] / 60,
                           charging_solve_status=item["two_stage"]["stage2_status_name"],
                           individual_route_replay=item["physical_replay_validated"],
                           duplicate_removal_validated=item["duplicate_trip_removal_validated"],
                           shared_capacity_validated=item["cross_route_charger_capacity_validated"],
                           mip_path=item["path"], mip_sha256=item["sha256"])
                checks.append(dict(case=cid, stage="mip", input_and_fleet_cap_match=True))
            rows.append(row)
    assert len({(r["chain"], r["target_buses"]) for r in rows}) == len(rows)
    return sorted(rows, key=lambda r: (r["target_buses"], r["chain"])), settings, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    raw = args.snapshot.read_bytes()
    rows, settings, checks = summarize(json.loads(raw))
    out = args.out_dir or Path(__file__).parent / ("status_" + args.snapshot.stem)
    out.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out / "all_chain_extension_results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    index = {(r["chain"], r["target_buses"]): r for r in rows}
    text = ["# Original one-hour chain MIPs", "",
            "Each cell is the actual integer number of buses found. A dash means no verified MIP result in this collection; it does not mean failure. Separate longer searches are excluded.", "",
            "| GIRO target | Chain 1 | Chain 2 | Chain 3 | Chain 4 | Chain 5 | Chain 6 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for k in sorted({r["target_buses"] for r in rows}):
        text.append("| " + " | ".join([str(k)] + [
            str(index.get((chain, k), {}).get("integer_buses", "—"))
            for chain in range(1, 7)]) + " |")
    text += ["", "## CG and proof status for the latest completed cases", "",
             "| Case | CG minutes | CG stop | Fractional route weight | Integer buses | Saved-pool fleet bound | Fleet proved in pool? |",
             "|---|---:|---|---:|---:|---:|---|"]
    for row in rows:
        if row["target_buses"] < 26 or not row["cg_result_collected"]:
            continue
        bound = row.get("saved_pool_fleet_bound")
        proof = row.get("fleet_proved_in_saved_pool")
        text.append("| " + " | ".join([
            f"C{row['chain']}, k={row['target_buses']}", f"{row['cg_minutes']:.1f}",
            "converged" if row["cg_pricing_certificate"] else {"wall_limit": "time limit", "max_iter": "iteration limit"}.get(row["cg_stop_reason"], row["cg_stop_reason"]),
            f"{row['fractional_route_weight']:.6f}", str(row.get("integer_buses", "—")),
            f"{bound:.6f}" if bound is not None else "—",
            "yes" if proof else "no" if proof is False else "—"]) + " |")
    text += ["", "CG minutes include import and CG at this k; graph preparation, earlier k values and MIP are separate. The weighted LP objective and fractional route weight come from the final pool re-solve. An uncertified restricted-master objective is not a full-model lower bound. A saved-pool fleet proof concerns only those columns; charging proof is separate.", "",
             "These are baseline covering runs: inherited full pools, 240 kWh/240 kW, no reserve, shared-capacity constraint or ending-SOC floor, and a fee of 5 per charging start. Individual-route replay and duplicate-removal validation have separate columns in the source CSV.", "",
             "[All values, units, proof flags, job IDs and source hashes](all_chain_extension_results.csv)."]
    (out / "CHAIN_TABLES.md").write_text("\n".join(text) + "\n")
    result = dict(snapshot_sha256=hashlib.sha256(raw).hexdigest(),
                  registered_cases=len(rows), cg_results=sum(r["cg_result_collected"] for r in rows),
                  mip_results=sum(r["mip_result_collected"] for r in rows),
                  identical_scientific_settings=settings, checks=checks, errors=[])
    (out / "chain_extension_table_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("registered_cases", "cg_results", "mip_results", "errors")}))


if __name__ == "__main__":
    main()
