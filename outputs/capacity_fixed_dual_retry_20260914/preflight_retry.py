#!/usr/bin/env python3
"""Native retry gate: actual nonzero-dual prefix entry with a ten-second call bound."""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import campaign


def main(root: Path) -> None:
    spec_doc = json.loads((root / "preflight_spec.json").read_text())
    case = spec_doc["case"]
    code = Path(spec_doc["code_root"])
    campaign.verify_checkout(code, case["execution_commit"])
    campaign.require_inputs(case)
    driver = campaign.load_driver(code)
    selector = sys.modules.get("capacity_window_selector")
    selector_path = Path(selector.__file__).resolve()
    expected_selector_path = (code / "src/capacity_window_selector.py").resolve()
    if selector_path != expected_selector_path:
        raise ValueError("preloaded selector path mismatch")
    job = os.environ.get("SLURM_JOB_ID", "local")
    out = root / "preflight" / job
    out.mkdir(parents=True, exist_ok=False)
    pool = out / "pool.jsonl"
    campaign.atomic_copy(Path(case["source_pool_path"]), pool, case["source_pool_sha256"])
    args = campaign.case_args(case, out / "unused.json", pool)
    with campaign.working_directory(code):
        problem = driver.build_problem(args.instance.parent, args.instance.name,
                                       reference_data_dir=args.reference_data_dir)
        prices = driver.load_station_hourly_prices(
            args.prices, sorted({driver.base_station_name(x) for x in driver.STATIONS}))
        provenance = driver.provenance(args, args.instance, args.prices, args.reference_data_dir)
        identity = driver.checkpoint_id(args, problem, provenance)
        if identity != case["checkpoint_id"]:
            raise ValueError("checkpoint mismatch")
        network = driver.build_network(args, problem, prices)
        master = driver.ExactCapacityMaster(
            problem.trips, capacity=driver.ARMS[args.arm]["capacity"], threads=1,
            log_path=out / "rmp.gurobi.log")
        routes = driver.load_resume_pool(
            pool, expected_id=identity, trips=problem.trips,
            route_validator=lambda route: driver.validate_injected_route(
                problem, route, args.battery_kwh, args.non_parx_kw,
                args.reserve_kwh, driver.HORIZON_MIN, arrival_grace_min=0.0,
                station_charge_kw=driver.station_power(args.arm)))
        for route in routes:
            master.add_route(route)
        lp = master.solve()
        if not lp["capacity_duals"]:
            raise ValueError("native retry gate requires nonzero capacity duals")
        observed = campaign.ObservedNetwork(network, driver, out / "pricing_call.json", case)
        started = time.perf_counter()
        try:
            observed.min_reduced_cost_route(
                lp["trip_duals"], capacity_duals=lp["capacity_duals"],
                capacity_sites=set(driver.CHARGER_COUNTS), capacity_grid_min=1,
                deadline=started + 10.0, clock=time.perf_counter)
        except driver.PricingDeadlineExceeded:
            pass
    call = json.loads((out / "pricing_call.json").read_text())
    actual = call["raw_dual_vector_sha256"]
    expected = campaign.raw_dual_hash(problem, lp)
    if actual != expected or actual != case["starting_raw_dual_vector_sha256"]:
        raise ValueError("actual native prefix call dual mismatch")
    if call["attempted_calls"] != 1:
        raise ValueError("retry gate did not enter exactly one call")
    result = {
        "schema":"evsp-dr-fixed-capacity-retry-native-preflight-v1",
        "status":"passed", "job_id":job, "case_id":case["case_id"],
        "nonzero_capacity_dual_count":len(lp["capacity_duals"]),
        "actual_raw_dual_vector_sha256":actual,
        "actual_normalized_9dp_dual_vector_sha256":call["normalized_9dp_dual_vector_sha256"],
        "selector_module_path":str(selector_path),
        "selector_module_sha256":campaign.sha256_file(selector_path),
        "pricing_entry_status":call["status"],
        "pricing_entry_elapsed_s":call.get("elapsed_s", time.perf_counter()-started),
        "pricing_entry_bound_s":10.0,
    }
    campaign.atomic_json(out / "preflight_result.json", result)
    campaign.atomic_json(root / "preflight_result.json", result)
    print(json.dumps(result,sort_keys=True))

if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
