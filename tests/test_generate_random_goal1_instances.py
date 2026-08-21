import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from generate_random_goal1_instances import (  # noqa: E402
    COMPACT_COLUMNS,
    base_vehicle_task,
    generate_batch,
    load_regular_source,
    select_vehicle_tasks,
    time_to_minutes,
)


SOURCE_COLUMNS = [
    "VehicleTask",
    "Identifier",
    "From1",
    "Start1",
    "End1",
    "To1",
    "Distance1",
    "Usage kWh",
    "count_trip_id",
    "Ordered_Trip_ID",
]


class RandomGoal1InstanceTests(unittest.TestCase):
    def write_source(self, path: Path) -> None:
        task_times = {
            "100": ("7:30", "101"),
            "101": ("4:05", "102"),
            "102": ("25:10", "103"),
            "103": ("8:00", "104"),
            "13316m": ("6:00", "105"),
            "13316uwt": ("6:05", "106"),
            "13324muw": ("9:00", "107"),
            "13324t": ("9:05", "108"),
        }
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SOURCE_COLUMNS)
            writer.writeheader()
            for task, (start, ordered_id) in task_times.items():
                writer.writerow(
                    {
                        "VehicleTask": task,
                        "Identifier": "Regular",
                        "From1": f"A{task}",
                        "Start1": start,
                        "End1": start,
                        "To1": f"B{task}",
                        "Distance1": "1.250",
                        "Usage kWh": "2.500",
                        "count_trip_id": "999",
                        "Ordered_Trip_ID": ordered_id,
                    }
                )
            writer.writerow(
                {
                    "VehicleTask": "100",
                    "Identifier": "Recharge",
                    "From1": "X",
                    "Start1": "5:00",
                    "End1": "5:10",
                    "To1": "X",
                    "Distance1": "",
                    "Usage kWh": "",
                    "count_trip_id": "",
                    "Ordered_Trip_ID": "",
                }
            )

    def test_only_explicit_weekday_variants_are_grouped(self):
        self.assertEqual(base_vehicle_task("13316m"), "13316")
        self.assertEqual(base_vehicle_task("13316uwt"), "13316")
        self.assertEqual(base_vehicle_task("13324muw"), "13324")
        self.assertEqual(base_vehicle_task("13324t"), "13324")
        self.assertEqual(base_vehicle_task("abc123m"), "abc123m")

    def test_time_parser_accepts_after_midnight_hours(self):
        self.assertEqual(time_to_minutes("25:10"), 1510)
        with self.assertRaises(ValueError):
            time_to_minutes("8:99")

    def test_generation_is_deterministic_compact_and_never_mixes_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path = root / "source.csv"
            self.write_source(source_path)
            first_dir = root / "first"
            second_dir = root / "second"

            first = generate_batch(
                source_path=source_path,
                output_dir=first_dir,
                sizes=(3, 4),
                replicates=4,
                seed=17,
            )
            second = generate_batch(
                source_path=source_path,
                output_dir=second_dir,
                sizes=(3, 4),
                replicates=4,
                seed=17,
            )

            self.assertEqual(len(first["instances"]), 8)
            self.assertEqual(first["variant_groups"], second["variant_groups"])
            for left, right in zip(first["instances"], second["instances"]):
                self.assertEqual(left["selected_base_tasks"], right["selected_base_tasks"])
                self.assertEqual(left["selected_literal_tasks"], right["selected_literal_tasks"])
                self.assertEqual(left["output_sha256"], right["output_sha256"])
                self.assertFalse(left["single_day_verified"])

                selected = set(left["selected_literal_tasks"])
                self.assertLessEqual(len(selected & {"13316m", "13316uwt"}), 1)
                self.assertLessEqual(len(selected & {"13324muw", "13324t"}), 1)

                output_path = first_dir / left["output_csv"]
                self.assertEqual(
                    hashlib.sha256(output_path.read_bytes()).hexdigest(),
                    left["output_sha256"],
                )
                with output_path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(tuple(rows[0].keys()), COMPACT_COLUMNS)
                self.assertEqual([int(row["count_trip_id"]) for row in rows], list(range(len(rows))))
                starts = [time_to_minutes(row["Start1"]) for row in rows]
                self.assertEqual(starts, sorted(starts))
                self.assertTrue(all(row["Identifier"] == "Regular" for row in rows))

            manifest_disk = json.loads((first_dir / "manifest.json").read_text())
            self.assertEqual(manifest_disk["source"]["sha256"], first["source"]["sha256"])
            self.assertEqual(manifest_disk["replicate_start"], 1)
            self.assertTrue((first_dir / "manifest.csv").exists())

    def test_replicate_start_generates_only_requested_suffixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path = root / "source.csv"
            self.write_source(source_path)
            manifest = generate_batch(
                source_path=source_path,
                output_dir=root / "output",
                sizes=(2,),
                replicates=3,
                replicate_start=4,
                seed=20260821,
            )
            self.assertEqual(
                [row["replicate"] for row in manifest["instances"]],
                [4, 5, 6],
            )
            self.assertTrue(all(
                f"_r{replicate:02d}.csv" in row["output_csv"]
                for replicate, row in zip(
                    (4, 5, 6), manifest["instances"]
                )
            ))

    def test_selection_is_a_base_task_sample_without_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source.csv"
            self.write_source(source_path)
            source = load_regular_source(source_path)
            bases, literals = select_vehicle_tasks(source, size=6, seed=2, replicate=1)
            self.assertEqual(len(bases), 6)
            self.assertEqual(len(set(bases)), 6)
            self.assertEqual(len(literals), 6)
            self.assertEqual(len(source.literals_by_base_task), 6)
            with self.assertRaises(ValueError):
                select_vehicle_tasks(source, size=7, seed=2, replicate=1)

    def test_different_existing_output_requires_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_path = root / "source.csv"
            self.write_source(source_path)
            output_dir = root / "output"
            generate_batch(
                source_path=source_path,
                output_dir=output_dir,
                sizes=(3,),
                replicates=1,
                seed=1,
            )
            generated = next(output_dir.glob("Practice_*.csv"))
            generated.write_text("different\n")
            with self.assertRaises(FileExistsError):
                generate_batch(
                    source_path=source_path,
                    output_dir=output_dir,
                    sizes=(3,),
                    replicates=1,
                    seed=1,
                )


if __name__ == "__main__":
    unittest.main()
