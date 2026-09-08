import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exact_pricer_expanded import validated_event_greedy_seed_records
from prepare_event_greedy_seed import canonical_sha256, event_partition


class FakeNetwork:
    def __init__(self, max_length=2):
        self.max_length = max_length

    def fixed_sequence_record(self, trips):
        trips = list(trips)
        if not trips or len(trips) > self.max_length:
            return None
        return {
            "trips": trips,
            "cost": float(sum(trips)),
            "expanded_grid_cost": float(sum(trips)),
            "master_cost_semantics": "expanded_grid_cost",
            "physical_realization": {"status": "valid_event_time_realized"},
        }


class EventGreedySeedTests(unittest.TestCase):
    def test_longest_prefix_split_is_deterministic(self):
        records, splits = event_partition([[1, 2, 3], [4]], FakeNetwork(2))
        self.assertEqual([row["trips"] for row in records], [[1, 2], [3], [4]])
        self.assertEqual(splits, 1)

    def test_validated_loader_reproduces_every_route(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tariff = root / "prices.csv"
            tariff.write_text("price\n1\n")
            tariff_sha = hashlib.sha256(tariff.read_bytes()).hexdigest()
            hashes = {
                "instance_sha256": "a" * 64,
                "prices_sha256": tariff_sha,
                "reference_sha256": "b" * 64,
                "deadhead_sha256": "c" * 64,
            }
            network = FakeNetwork(2)
            routes = []
            for trips in ([1, 2], [3]):
                route = network.fixed_sequence_record(trips)
                route["cost_tariff_sha256"] = tariff_sha
                routes.append(route)
            identity = {"cache": "identity"}
            payload = {
                "schema": "evsp-dr-event-greedy-partition-v1",
                "source": "GREEDY",
                "continuous_cost_pricing_certified": False,
                "exact_trip_partition": True,
                "input_hashes": hashes,
                "physics": {
                    "g_kwh": 240.0, "charge_kw": 240.0,
                    "reserve_kwh": 0.0, "soc_step": 2.5,
                    "block_min": 5,
                },
                "event_network_cache": {
                    "identity": identity, "pickle_sha256": "d" * 64,
                },
                "routes": routes,
                "route_record_sha256": [canonical_sha256(r) for r in routes],
                "route_count": 2,
                "trip_count": 3,
            }
            source = root / "seed.json"
            source.write_text(json.dumps(payload))
            accepted, observed_sha = validated_event_greedy_seed_records(
                source, SimpleNamespace(trips=(1, 2, 3)), network,
                cache_manifest={
                    "identity": identity, "pickle_sha256": "d" * 64,
                },
                provenance=hashes, tariff_path=tariff,
                g_kwh=240.0, charge_kw=240.0, reserve_kwh=0.0,
                soc_step=2.5, block_min=5,
            )
            self.assertEqual([r["trips"] for r in accepted], [[1, 2], [3]])
            self.assertTrue(all(
                r["origin"] == "validated_event_greedy_seed"
                for r in accepted
            ))
            self.assertEqual(observed_sha, hashlib.sha256(source.read_bytes()).hexdigest())

            payload["physics"]["g_kwh"] = 300.0
            source.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "identity/physics/cache"):
                validated_event_greedy_seed_records(
                    source, SimpleNamespace(trips=(1, 2, 3)), network,
                    cache_manifest={
                        "identity": identity, "pickle_sha256": "d" * 64,
                    },
                    provenance=hashes, tariff_path=tariff,
                    g_kwh=240.0, charge_kw=240.0, reserve_kwh=0.0,
                    soc_step=2.5, block_min=5,
                )


if __name__ == "__main__":
    unittest.main()
