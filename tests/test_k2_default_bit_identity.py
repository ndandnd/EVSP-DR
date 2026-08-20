import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import exact_pricer_expanded as current  # noqa: E402


PRE_OBJECTIVE_COMMIT = "2b857f4"


def load_prechange_module():
    source = subprocess.run(
        [
            "git", "show",
            f"{PRE_OBJECTIVE_COMMIT}:src/exact_pricer_expanded.py",
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    module = types.ModuleType("exact_pricer_prechange")
    module.__file__ = str(REPO / "src/exact_pricer_expanded.py")
    module.__package__ = ""
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


class K2DefaultBitIdentityTests(unittest.TestCase):
    def test_omitted_objective_is_bit_identical_to_prechange_k2(self):
        prechange = load_prechange_module()
        clock = SimpleNamespace(
            time=lambda: 1000.0,
            perf_counter=lambda: 1000.0,
        )
        provenance = {
            "instance_sha256": "instance",
            "prices_sha256": "prices",
            "reference_sha256": "reference",
            "deadhead_sha256": "deadhead",
        }
        with tempfile.TemporaryDirectory() as tmp:
            outputs = {}
            for label, module in (
                ("prechange", prechange),
                ("current", current),
            ):
                output = Path(tmp) / f"{label}.json"
                with (
                    patch.object(module, "time", clock),
                    patch.object(
                        module, "_provenance", return_value=provenance,
                    ),
                ):
                    module.main([
                        "--csv",
                        "scale_ladder/instances/"
                        "Practice_Custom_DutyUnion_k02_r1.csv",
                        "--prices_csv", "hourly_prices_flat.csv",
                        "--soc-step", "15",
                        "--block-min", "10",
                        "--max-iters", "400",
                        "--columns_per_iter", "30",
                        "--master-sense", "partition",
                        "--initial-pool", "singletons",
                        "--out", str(output),
                    ])
                journal = Path(str(output) + ".columns.jsonl").read_bytes()
                iterations = Path(str(output) + ".iters.csv").read_bytes()
                route_hashes = [
                    hashlib.sha256(line).hexdigest()
                    for line in journal.splitlines(keepends=True)
                ]
                rows = list(csv.DictReader(
                    iterations.decode().splitlines()
                ))
                outputs[label] = {
                    "journal": journal,
                    "route_hashes": route_hashes,
                    "reduced_costs": [row["min_rc"] for row in rows],
                    "iterations": iterations,
                    "status": json.loads(output.read_text()),
                }
            before, after = outputs["prechange"], outputs["current"]
            self.assertEqual(after["journal"], before["journal"])
            self.assertEqual(after["route_hashes"], before["route_hashes"])
            self.assertEqual(after["reduced_costs"], before["reduced_costs"])
            self.assertEqual(after["iterations"], before["iterations"])
            self.assertTrue(after["status"]["certified_rc_optimal"])
            self.assertEqual(
                after["status"]["final"]["route_weight"],
                before["status"]["final"]["route_weight"],
            )


if __name__ == "__main__":
    unittest.main()
