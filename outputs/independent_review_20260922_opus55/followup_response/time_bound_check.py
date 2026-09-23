"""Reproduce exact time-only certificates; no optimization solver or cluster calls."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import json
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HELPER = ROOT / "outputs/independent_review_20260916/time_only_vsp_20260916/time_only_vsp.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def main():
    started = time.monotonic()
    spec = importlib.util.spec_from_file_location("time_only_certificate", HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    refs, scale, direct, closure, resolve = helper.travel_data()
    expected_static = {
        "Ref_dict.csv": "7bda0e1f439dc8bf5081499566eb2c6a0314190ef27294707f1403fd2c13e3a0",
        "par_ref_dhd.csv": "5993e922c671f053611635578b32a1be13bab87b3b5fd8c02b699b81fe0eb66c",
    }
    for name, digest in expected_static.items():
        assert sha(HELPER.parent / "inputs" / name) == digest
    # Every direct and charger-detour travel time is at least this relaxation.
    assert all(closure[i][j] <= direct[i][j] for i in range(len(refs)) for j in range(len(refs)))
    assert all(closure[i][j] <= direct[i][s] + direct[s][j]
               for i in range(len(refs)) for j in range(len(refs)) for s in range(len(refs)))
    panel_path = ROOT / "outputs/research_management_20260921/paper_results/figure1_paired_budget.csv"
    manifest_path = ROOT / "outputs/chain_extension_20260913/inputs/sources/selection_manifest.csv"
    manifest = {r["instance_file_sha256"]: r for r in rows(manifest_path)}
    work = []
    snapshots = {}
    for row in rows(panel_path):
        selected = manifest[row["input_sha256"]]
        input_path = manifest_path.parent / Path(selected["relative_path"]).name
        if not input_path.exists():
            input_path = ROOT / ".codex-work/integer-columns-20260921" / selected["relative_path"]
        snapshot_path = ROOT / row["snapshot_path"]
        assert sha(snapshot_path) == row["snapshot_sha256"]
        if str(snapshot_path) not in snapshots:
            snapshots[str(snapshot_path)] = json.loads(snapshot_path.read_text())["campaigns"]["cumulative_budget_20260913"]
        campaign = snapshots[str(snapshot_path)]
        assert campaign["source_hashes"]["static_sha256"]["Ref_dict.csv"] == expected_static["Ref_dict.csv"]
        assert campaign["source_hashes"]["static_sha256"]["par_ref_dhd.csv"] == expected_static["par_ref_dhd.csv"]
        endpoints = {r["budget_arm"]: r for r in campaign["mip"] if r["case_id"] == row["case_id"]}
        warm, fresh = endpoints["warm"], endpoints["base"]
        for arm, source in [("warm", warm), ("fresh", fresh)]:
            assert source["input_sha256"] == row["input_sha256"]
            assert source["sha256"] == row[f"{arm}_mip_source_sha256"]
            assert source["physical_replay_validated"] is True
            assert source["two_stage"]["stage1_incumbent_validated"] is True
            assert source["buses"] == int(row[f"{arm}_buses"])
        work.append(dict(case_id=row["case_id"], cohort="figure1", target=int(row["target_k"]),
                         input_path=input_path, input_sha256=row["input_sha256"],
                         sequential_buses=warm["buses"], fresh_buses=fresh["buses"],
                         upper_evidence="Previously collected native individual-route replay and covering validation; not re-simulated here",
                         snapshot_path=str(snapshot_path.relative_to(ROOT)), snapshot_sha256=sha(snapshot_path),
                         warm_mip_source_sha256=warm["sha256"], fresh_mip_source_sha256=fresh["sha256"]))
    for stamp in ["20260922T195842Z", "20260922T235958Z"]:
        base = ROOT / f"outputs/research_management_20260922/monitor_{stamp}/operations"
        for row in rows(base / "cg_endpoints.csv"):
            case = row["case"]
            files = list((base / f"baseline/cases/{case}/cg").glob("*/cg.json"))
            payloads = [p for p in files if sha(p) == row["result_sha256"]]
            assert len(payloads) == 1
            payload = json.loads(payloads[0].read_text())
            assert payload["provenance"]["reference_sha256"] == expected_static["Ref_dict.csv"]
            assert payload["provenance"]["deadhead_sha256"] == expected_static["par_ref_dhd.csv"]
            work.append(dict(case_id=case, cohort="large_endpoints", target=int(row["target_duties"]),
                             input_path=ROOT / f"outputs/week_20260921/chain_extension_40/inputs/{case}.csv",
                             input_sha256=payload["provenance"]["instance_sha256"],
                             reported_RMP_weight=float(row["route_weight"]),
                             cg_payload_sha256=row["result_sha256"]))
    records, certificates = [], []
    assert len({r["case_id"] for r in work}) == len(work)
    for row in work:
        path = row.pop("input_path")
        assert sha(path) == row["input_sha256"]
        trips = []
        for r in rows(path):
            start, end = helper.minutes(r["Start1"]), helper.minutes(r["End1"])
            assert end > start
            trips.append(dict(ordered=int(r["Ordered_Trip_ID"]), start=start, end=end,
                              start_ref=resolve(r["From1"]), end_ref=resolve(r["To1"])))
        assert len({t["ordered"] for t in trips}) == len(trips)
        adj = helper.graph(trips, closure, scale)
        cert = helper.maximum_matching(adj)
        anti = cert["antichain"]
        assert anti is not None
        # Independently validate incompatibility along *all paths*, not just edges.
        reachable = [0] * len(trips)
        for u in sorted(range(len(trips)), key=lambda i: trips[i]["start"], reverse=True):
            for v in adj[u]:
                assert trips[v]["start"] >= trips[u]["end"] > trips[u]["start"]
                reachable[u] |= (1 << v) | reachable[v]
        mask = sum(1 << u for u in anti)
        assert all((reachable[u] & mask) == 0 for u in anti)
        assert len(anti) == cert["minimum_path_cover"]
        overlap = helper.overlap(trips)
        assert len(anti) >= overlap
        row.update(input_path=str(path.relative_to(ROOT)), trips=len(trips), concurrency_lower_bound=overlap,
                   exact_time_only_lower_bound=len(anti), antichain_ordered_trip_ids=[trips[u]["ordered"] for u in anti],
                   no_antichain_pair_connected_by_any_path=True,
                   graph_sha256=helper.digest(adj), graph_edges=sum(map(len, adj)))
        if row["cohort"] == "figure1":
            assert row["sequential_buses"] >= len(anti)
            row["sequential_fleet_matches_lower_bound"] = row["sequential_buses"] == len(anti)
            row["fresh_fleet_matches_lower_bound"] = row["fresh_buses"] == len(anti)
        cert.update(case_id=row["case_id"], input_sha256=row["input_sha256"],
                    graph_sha256=row["graph_sha256"], ordered_trip_ids=[t["ordered"] for t in trips])
        records.append(row)
        certificates.append(cert)
        print(row["case_id"], "concurrency", overlap, "exact time-only bound", len(anti), flush=True)
    out = dict(created_utc=datetime.now(timezone.utc).isoformat(), cases=records,
               source_hashes={str(HELPER.relative_to(ROOT)): sha(HELPER),
                              str(panel_path.relative_to(ROOT)): sha(panel_path),
                              str(manifest_path.relative_to(ROOT)): sha(manifest_path),
                              "time_bound_check.py": sha(__file__), **expected_static},
               exact_time_arithmetic=True, time_scale=scale, external_solver_calls=0, cluster_calls=0,
               scope="Fixed trips and baseline deadhead table, homogeneous covering fleet. Energy, depot, chargers and wait restrictions relaxed. No claim for different raw interval deadheads or full GIRO operation.",
               proof="Each feasible model route induces a path in the relaxed time graph. No path contains two certified antichain trips. Covering k such trips requires at least k routes, including fractionally. Equality with a previously validated k-route covering incumbent proves its fleet optimality in that model; charging optimality does not follow.",
               elapsed_seconds=time.monotonic() - started)
    (HERE / "time_bounds.json").write_text(json.dumps(out, indent=2) + "\n")
    (HERE / "time_certificates.json").write_text(json.dumps(certificates, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
