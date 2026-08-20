import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from target_pool_feasibility import solve_target_feasibility  # noqa: E402
import exact_pricer_expanded as exact  # noqa: E402
from union_resolution_pools import (  # noqa: E402
    build_union,
    evaluate,
    merge_route_sets,
    route_sha256,
    validate_union_witness,
)


def route(trips, cost=100000.0):
    return {
        "trips": trips,
        "route_nodes": ["D", *trips, "D"],
        "charging_stops": {
            "stations": [], "cst": [], "cet": [], "kwh": [],
        },
        "expanded_grid_charging_stops": {
            "stations": [], "cst": [], "cet": [], "kwh": [],
        },
        "continuous_realized_charging_blocks": [],
        "cost": cost,
    }


def identity(instance="i", g_kwh=300.0):
    return {
        "instance_sha256": instance,
        "prices_sha256": "p",
        "reference_sha256": "r",
        "deadhead_sha256": "d",
        "g_kwh": g_kwh,
        "charge_kw": 300.0,
        "min_soc_frac": 0.0,
        "csv": "instance.csv",
        "prices_csv": "prices.csv",
        "trip_ids": [0, 1],
    }


class ResolutionPoolUnionTests(unittest.TestCase):
    def test_union_deduplicates_hashes_and_is_source_superset(self):
        shared = route([0])
        sources = [
            {
                "identity": identity(),
                "journal_sha256": "a",
                "routes": [shared, route([1])],
            },
            {
                "identity": identity(),
                "journal_sha256": "b",
                "routes": [shared, route([0, 1])],
            },
        ]
        merged, proof = merge_route_sets(sources)
        self.assertEqual(len(merged), 3)
        self.assertTrue(proof["verified"])
        merged_hashes = {route_sha256(item) for item in merged}
        for source in sources:
            self.assertTrue(
                {route_sha256(item) for item in source["routes"]}
                <= merged_hashes
            )

    def test_union_refuses_instance_and_physics_mismatch(self):
        base = {
            "identity": identity(),
            "journal_sha256": "a",
            "routes": [route([0]), route([1])],
        }
        for changed in (
            identity(instance="foreign"),
            identity(g_kwh=240.0),
        ):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    merge_route_sets([
                        base,
                        {
                            "identity": changed,
                            "journal_sha256": "b",
                            "routes": [route([0, 1])],
                        },
                    ])

    def test_union_target_result_is_no_worse_than_best_input(self):
        first = [route([0]), route([1])]
        second = [route([0, 1])]
        sources = [
            {"identity": identity(), "journal_sha256": "a", "routes": first},
            {"identity": identity(), "journal_sha256": "b", "routes": second},
        ]
        merged, _proof = merge_route_sets(sources)
        first_result = solve_target_feasibility(
            first, [0, 1], 1, timelimit=30, threads=1,
        )
        best_result = solve_target_feasibility(
            second, [0, 1], 1, timelimit=30, threads=1,
        )
        union_result = solve_target_feasibility(
            merged, [0, 1], 1, timelimit=30, threads=1,
        )
        self.assertEqual(first_result["outcome"], "INFEASIBLE")
        self.assertEqual(best_result["outcome"], "FEASIBLE")
        self.assertEqual(union_result["outcome"], "FEASIBLE")

    def test_route_hash_distinguishes_realizations(self):
        self.assertNotEqual(
            route_sha256(route([0], cost=100000.0)),
            route_sha256(route([0], cost=100001.0)),
        )

    def test_build_union_binds_audited_sources_and_allows_equal_journal_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);results=[]
            status={
                "csv":"instance.csv","prices_csv":"prices.csv",
                "trip_ids":[0,1],"g_kwh":300.0,"charge_kw":300.0,
                "min_soc_frac":0.0,"soc_step":15.0,"block_min":10,
                "provenance":{
                    "instance_sha256":"i","prices_sha256":"p",
                    "reference_sha256":"r","deadhead_sha256":"d",
                },
            }
            for index in range(2):
                result=root/f"source{index}.json";journal=Path(str(result)+".columns.jsonl")
                journal.write_text(json.dumps(route([0]))+"\n")
                result.write_text(json.dumps({**status,"columns_journal":str(journal)}))
                results.append(result)
            audit={"input_hashes":{
                "instance_sha256":"i","prices_sha256":"p",
                "reference_sha256":"r","deadhead_sha256":"d",
            }}

            def loaded(*_args,**_kwargs):
                return [route([0]),route([1])],[0,1]

            with (
                patch("union_resolution_pools.load_bound_pool_bytes",side_effect=loaded),
                patch("union_resolution_pools.prepare_strict_partition_pool",
                      return_value=([route([0]),route([1])],audit)),
                patch("union_resolution_pools._snapshot_inputs",
                      return_value=(root,root)),
                patch("audit_giro_known_columns.build_problem",
                      return_value=SimpleNamespace(trips=(0,1))),
                patch("union_resolution_pools.verified_mip_code_identity",
                      return_value={"observed_commit":"a"*40}),
            ):
                payload,routes=build_union(
                    results,output_path=root/"union.json",
                )
            self.assertEqual(len(payload["sources"]),2)
            self.assertEqual(len(routes),2)
            self.assertTrue(payload["route_hash_deduplication"]["verified"])

    def test_build_union_rejects_loaded_status_swap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);result=root/"source.json";journal=root/"source.jsonl"
            status={"columns_journal":str(journal),"trip_ids":[0]}
            result.write_text(json.dumps(status));journal.write_text("{}\n")
            def swapped_loaded(*_args):
                result.write_text(json.dumps({**status,"trip_ids":[1]}))
                return [route([1])],[1]
            with (
                patch("union_resolution_pools.load_bound_pool_bytes",
                      side_effect=swapped_loaded),
                patch("union_resolution_pools.verified_mip_code_identity",
                      return_value={}),
                self.assertRaisesRegex(RuntimeError,"changed while loading"),
            ):
                build_union(
                    [result,result],output_path=root/"union.json",
                )

    def test_union_witness_rejects_falsified_immutable_trip_ids(self):
        union={
            "csv":"instance.csv","prices_csv":"prices.csv",
            "trip_ids":[0],"g_kwh":300.0,"charge_kw":300.0,
            "min_soc_frac":0.0,"provenance":identity(),
            "sources":[{
                "status":{"csv":"instance.csv","prices_csv":"prices.csv"},
                "physical_pool_audit":{"input_hashes":identity()},
            }],
        }
        with (
            patch("union_resolution_pools._snapshot_inputs",
                  return_value=(Path("/data"),Path("/reference"))),
            patch("union_resolution_pools.validate_final_selected_routes",
                  side_effect=RuntimeError("trip identity")),
            self.assertRaisesRegex(RuntimeError,"trip identity"),
        ):
            validate_union_witness(union,[route([0])])

    def test_union_rejects_repeated_exact_source_identity(self):
        source={
            "source_id":"same","identity":identity(),
            "journal_sha256":"a","routes":[route([0]),route([1])],
        }
        with self.assertRaisesRegex(ValueError,"repeats an exact source"):
            merge_route_sets([source,dict(source)])

    def test_equal_journal_sources_are_cli_order_independent(self):
        sources=[
            {"source_id":"b","identity":identity(),
             "journal_sha256":"same","routes":[route([0]),route([1])]},
            {"source_id":"a","identity":identity(),
             "journal_sha256":"same","routes":[route([0]),route([0,1])]},
        ]
        forward_routes,forward_proof=merge_route_sets(sources)
        reverse_routes,reverse_proof=merge_route_sets(list(reversed(sources)))
        self.assertEqual(forward_routes,reverse_routes)
        self.assertEqual(forward_proof,reverse_proof)

    def test_end_to_end_union_feasible_witness_is_strictly_replayed(self):
        data=REPO/"data";csv_name="Practice_Selected_1buses.csv"
        problem=exact.build_problem(
            data,csv_name,
            max_station_to_trip_wait_min=exact.HORIZON_MIN,
        )
        hashes={
            "instance_sha256":hashlib.sha256((data/csv_name).read_bytes()).hexdigest(),
            "prices_sha256":hashlib.sha256((data/"hourly_prices_flat.csv").read_bytes()).hexdigest(),
            "reference_sha256":hashlib.sha256((data/"Ref_dict.csv").read_bytes()).hexdigest(),
            "deadhead_sha256":hashlib.sha256((data/"par_ref_dhd.csv").read_bytes()).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sources=[]
            for soc_step in (15.0,5.0):
                records,missing=exact.direct_singleton_seed_records(
                    problem,g_kwh=300.0,soc_step=soc_step,reserve_kwh=0.0,
                )
                self.assertEqual(missing,[])
                for record in records:
                    record["cost_tariff_sha256"]=hashes["prices_sha256"]
                status={
                    "csv":csv_name,"prices_csv":"hourly_prices_flat.csv",
                    "trip_ids":list(problem.trips),"g_kwh":300.0,
                    "charge_kw":300.0,"min_soc_frac":0.0,
                    "soc_step":soc_step,"block_min":10,
                    "provenance":hashes,
                }
                result=root/f"source_{soc_step}.json"
                journal=Path(str(result)+".columns.jsonl")
                journal.write_text("".join(
                    json.dumps(record)+"\n" for record in records
                ))
                result.write_text(json.dumps({
                    **status,"columns_journal":str(journal),
                }))
                sources.append(result)
            args=SimpleNamespace(
                result=sources,target=len(problem.trips),timelimit=30,
                threads=1,seed=0,data_dir=data,reference_data_dir=data,
                out=root/"union.json",mip_out=root/"mip.json",
            )
            union,mip=evaluate(args)
            self.assertEqual(mip["outcome"],"FEASIBLE")
            self.assertTrue(mip["witness_audit"]["validated"])
            self.assertEqual(union["source_count"],2)
            self.assertEqual(
                len({source["journal_sha256"] for source in union["sources"]}),
                1,
            )


if __name__ == "__main__":
    unittest.main()
