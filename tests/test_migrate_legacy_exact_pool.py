import hashlib
import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from exact_pricer_expanded import (  # noqa: E402
    load_column_pool,
    resume_identity_mismatches,
    resume_pool_mismatches,
)
from migrate_legacy_exact_pool import (  # noqa: E402
    MigrationError,
    apply_migration,
    build_migration_plan,
)
from durable_io import DurableFileError, exclusive_output_lock  # noqa: E402


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class LegacyExactPoolMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source_data = self.root / "source_data"
        self.destination_data = self.root / "destination_data"
        self.results = self.root / "results"
        self.destination_dir = self.root / "recovered"
        for folder in (
                self.source_data, self.destination_data, self.results,
                self.destination_dir):
            folder.mkdir()
        self.instance_bytes = b"TripID,StartTime\n1,100\n2,200\n"
        self.price_bytes = b"hour,price\n0,1.0\n"
        for data in (self.source_data, self.destination_data):
            (data / "instance.csv").write_bytes(self.instance_bytes)
            (data / "prices.csv").write_bytes(self.price_bytes)
        self.legacy = {
            "csv": "instance.csv",
            "prices_csv": "prices.csv",
            "soc_step": 15.0,
            "block_min": 10,
            "g_kwh": 300.0,
            "charge_kw": 300.0,
            "min_soc_frac": 0.0,
            "master_sense": "partition",
            "trip_ids": [1, 2],
            "iterations": 1,
            "columns": 1,
            "wall_s": 5.0,
            "stop_reason": "running",
            "final": {
                "lp_obj": 200000.0, "route_weight": 2.0,
                "artificials": 0.0, "min_rc": -1.0,
            },
        }
        self.source = self.results / "legacy.json"
        self.source.write_text(json.dumps(self.legacy))
        self.journal = Path(str(self.source) + ".columns.jsonl")
        self.record1 = {"trips": [1], "cost": 100000.0}
        self.record2 = {"trips": [2], "cost": 100001.0}
        self.journal.write_text(
            json.dumps(self.record1) + json.dumps(self.record2)
            + '{"trips":'
        )
        self.iters = Path(str(self.source) + ".iters.csv")
        self.iters.write_text(
            "elapsed_s,iteration,lp_obj,route_weight,artificials,min_rc,"
            "pool_columns\n"
            "10,1,200000,2,0,-1,1\n"
            "20,2,199999,1.9,0,-0.5,2\n"
            "21,3,"
        )
        self.instance_hash = sha(self.instance_bytes)
        self.price_hash = sha(self.price_bytes)
        self._write_witness(
            "instance_witness.json", csv="instance.csv",
            prices_csv="other_prices.csv",
            instance_sha=self.instance_hash, prices_sha="a" * 64,
            trip_ids=[1, 2],
        )
        self._write_witness(
            "price_witness.json", csv="other_instance.csv",
            prices_csv="prices.csv",
            instance_sha="b" * 64, prices_sha=self.price_hash,
            trip_ids=[99],
        )
        self.destination = self.destination_dir / "recovered.json"

    def tearDown(self):
        self.temp.cleanup()

    def _write_witness(
        self, name, *, csv, prices_csv, instance_sha, prices_sha, trip_ids,
    ):
        payload = {
            **{field: self.legacy[field] for field in (
                "soc_step", "block_min", "g_kwh", "charge_kw",
                "min_soc_frac", "master_sense",
            )},
            "csv": csv,
            "prices_csv": prices_csv,
            "trip_ids": trip_ids,
            "stop_reason": "certified",
            "provenance": {
                "git_commit": "witness-commit",
                "instance_sha256": instance_sha,
                "prices_sha256": prices_sha,
            },
        }
        (self.results / name).write_text(json.dumps(payload))

    def _plan(self, *, source_logs=None):
        with patch(
            "migrate_legacy_exact_pool._tool_identity",
            return_value={
                "commit": "c" * 40,
                "branch": "peel-and-price",
                "dirty": False,
            },
        ):
            return build_migration_plan(
                source_result=self.source,
                destination=self.destination,
                source_data_dir=self.source_data,
                destination_data_dir=self.destination_data,
                witness_roots=[self.results],
                source_logs=list(source_logs or []),
                legacy_commit="legacy-commit",
                slurm_array_job="867334",
                slurm_task="32",
            )

    def test_copy_only_migration_repairs_and_attests_legacy_pool(self):
        source_bytes = {
            self.source: self.source.read_bytes(),
            self.journal: self.journal.read_bytes(),
            self.iters: self.iters.read_bytes(),
        }

        attestation = apply_migration(self._plan())

        for path, original in source_bytes.items():
            self.assertEqual(path.read_bytes(), original)
        migrated = json.loads(self.destination.read_text())
        destination_journal = Path(str(self.destination) + ".columns.jsonl")
        records = [
            json.loads(line) for line in destination_journal.read_text().splitlines()
        ]
        self.assertEqual(records, [self.record1, self.record2])
        self.assertEqual(migrated["columns"], 2)
        self.assertEqual(migrated["iterations"], 2)
        self.assertEqual(migrated["wall_s"], 20.0)
        self.assertEqual(migrated["stop_reason"], "prepared_legacy_resume")
        self.assertEqual(
            migrated["provenance"]["instance_sha256"], self.instance_hash
        )
        self.assertEqual(
            migrated["provenance"]["prices_sha256"], self.price_hash
        )
        self.assertTrue(attestation["repairs"]["journal"]["applied"])
        self.assertTrue(attestation["repairs"]["iters"]["applied"])
        raw_journal = (
            self.destination.parent
            / f"{self.destination.name}.legacy_raw"
            / "source_result.json.columns.jsonl"
        )
        self.assertFalse(os.path.samefile(raw_journal, self.journal))
        self.assertEqual(raw_journal.read_bytes(), source_bytes[self.journal])
        self.assertTrue(
            (raw_journal.parent / "journal_changed_tail.bin").is_file()
        )

        args = Namespace(
            csv="instance.csv", prices_csv="prices.csv", soc_step=15.0,
            block_min=10, g_kwh=300.0, charge_kw=300.0,
            min_soc_frac=0.0, master_sense="partition",
        )
        self.assertEqual(
            resume_identity_mismatches(
                migrated, args, [1, 2], migrated["provenance"]
            ),
            [],
        )
        pool = load_column_pool(records, [1, 2])
        self.assertEqual(resume_pool_mismatches(migrated, pool), [])

    def test_migration_is_idempotent_after_resume_extends_journal(self):
        plan = self._plan()
        first = apply_migration(plan)
        destination_journal = Path(str(self.destination) + ".columns.jsonl")
        with destination_journal.open("a") as handle:
            handle.write(json.dumps({"trips": [1, 2], "cost": 100002.0}) + "\n")
        extended = destination_journal.read_bytes()

        second = apply_migration(plan)

        self.assertEqual(first["migration_id"], second["migration_id"])
        self.assertEqual(destination_journal.read_bytes(), extended)

    def test_interior_corruption_fails_without_touching_source(self):
        original = (
            json.dumps(self.record1) + "\nnot-json\n"
            + json.dumps(self.record2) + "\n"
        )
        self.journal.write_text(original)

        with self.assertRaisesRegex(DurableFileError, "before EOF"):
            self._plan()

        self.assertEqual(self.journal.read_text(), original)
        self.assertFalse(self.destination.exists())

    def test_losslessly_normalizes_complete_interior_dict_sequence(self):
        preceding = (json.dumps(self.record1) + "\n").encode()
        concatenated = (
            json.dumps(self.record2) + json.dumps(self.record1) + "\n"
        ).encode()
        original = (
            preceding + concatenated + (json.dumps(self.record2) + "\n").encode()
        )
        self.journal.write_bytes(original)

        plan = self._plan()
        preview_repairs = plan["repair_preview"]["journal"][
            "lossless_interior_normalizations"
        ]
        self.assertEqual(len(preview_repairs), 1)
        self.assertEqual(
            preview_repairs[0]["original_offset"], len(preceding)
        )
        self.assertEqual(preview_repairs[0]["recovered_objects"], 2)
        self.assertEqual(
            preview_repairs[0]["original_line_sha256"],
            hashlib.sha256(concatenated).hexdigest(),
        )

        attestation = apply_migration(plan)

        self.assertEqual(self.journal.read_bytes(), original)
        destination_journal = Path(str(self.destination) + ".columns.jsonl")
        records = [
            json.loads(line) for line in destination_journal.read_text().splitlines()
        ]
        self.assertEqual(
            records, [self.record1, self.record2, self.record1, self.record2]
        )
        migrated = json.loads(self.destination.read_text())
        self.assertEqual(migrated["columns"], 2)
        self.assertFalse(migrated["certified_rc_optimal"])
        self.assertIsNone(migrated["final_lp"])
        self.assertTrue(attestation["repairs"]["journal"]["applied"])
        self.assertEqual(
            attestation["repairs"]["journal"][
                "lossless_interior_normalizations"
            ],
            preview_repairs,
        )
        raw_journal = (
            self.destination.parent
            / f"{self.destination.name}.legacy_raw"
            / "source_result.json.columns.jsonl"
        )
        self.assertEqual(raw_journal.read_bytes(), original)
        repair = attestation["repairs"]["journal"]
        changed_tail = raw_journal.parent / "journal_changed_tail.bin"
        self.assertTrue(changed_tail.is_file())
        self.assertEqual(changed_tail.stat().st_size, repair["original_tail_bytes"])
        self.assertEqual(
            hashlib.sha256(changed_tail.read_bytes()).hexdigest(),
            repair["original_tail_sha256"],
        )

    def test_ambiguous_interior_sequences_still_fail_closed(self):
        malformed_lines = (
            json.dumps(self.record1) + '{"trips":',
            json.dumps(self.record1) + "junk",
            json.dumps(self.record1) + "[]",
            json.dumps(self.record1) + json.dumps(self.record2) + "junk",
            json.dumps(self.record1) + json.dumps(self.record2) + '{"trips":',
        )
        for index, malformed in enumerate(malformed_lines):
            with self.subTest(malformed=malformed):
                original = (
                    malformed + "\n" + json.dumps(self.record2) + "\n"
                )
                self.journal.write_text(original)
                self.destination = self.destination_dir / f"refused_{index}.json"

                with self.assertRaisesRegex(DurableFileError, "before EOF"):
                    self._plan()

                self.assertEqual(self.journal.read_text(), original)
                self.assertFalse(self.destination.exists())

    def test_missing_or_conflicting_witness_fails_closed(self):
        (self.results / "instance_witness.json").unlink()
        with self.assertRaisesRegex(MigrationError, "no authenticated"):
            self._plan()

        self._write_witness(
            "instance_witness.json", csv="instance.csv",
            prices_csv="other_prices.csv",
            instance_sha=self.instance_hash, prices_sha="a" * 64,
            trip_ids=[1, 2],
        )
        self._write_witness(
            "conflict.json", csv="instance.csv", prices_csv="third.csv",
            instance_sha="f" * 64, prices_sha="c" * 64,
            trip_ids=[1, 2],
        )
        with self.assertRaisesRegex(MigrationError, "conflicting"):
            self._plan()

    def test_migration_handles_observed_unparseable_legacy_tail_classes(self):
        for damaged_tail in (b"not-json", b'null{"trips":[2],"cost":2}'):
            with self.subTest(damaged_tail=damaged_tail):
                self.journal.write_bytes(
                    (json.dumps(self.record1) + "\n").encode() + damaged_tail
                )
                destination = self.destination.with_name(
                    f"recovered_{hashlib.sha256(damaged_tail).hexdigest()[:8]}.json"
                )
                original_destination = self.destination
                self.destination = destination
                try:
                    attestation = apply_migration(self._plan())
                finally:
                    self.destination = original_destination
                self.assertTrue(attestation["repairs"]["journal"]["applied"])
                repaired = Path(str(destination) + ".columns.jsonl")
                self.assertEqual(
                    [json.loads(line) for line in repaired.read_text().splitlines()],
                    [self.record1],
                )

    def test_apply_rejects_concurrent_owner_and_orphan_artifacts(self):
        plan = self._plan()
        with exclusive_output_lock(self.destination, {"owner": "test"}):
            with self.assertRaisesRegex(DurableFileError, "another process"):
                apply_migration(plan)

        orphan = Path(str(self.destination) + ".columns.jsonl")
        orphan.write_text(json.dumps(self.record1) + "\n")
        with self.assertRaisesRegex(MigrationError, "status is missing"):
            apply_migration(plan)

    def test_idempotency_detects_corrupt_iterations_and_raw_archive(self):
        plan = self._plan()
        apply_migration(plan)
        destination_iters = Path(str(self.destination) + ".iters.csv")
        destination_iters.write_bytes(b"corrupt")
        with self.assertRaisesRegex(MigrationError, "iteration log"):
            apply_migration(plan)

        self.destination = self.destination_dir / "second.json"
        plan = self._plan()
        apply_migration(plan)
        raw_status = (
            self.destination.parent
            / f"{self.destination.name}.legacy_raw"
            / "source_result.json"
        )
        raw_status.write_bytes(b"corrupt")
        with self.assertRaisesRegex(MigrationError, "raw migration archive"):
            apply_migration(plan)

    def test_requested_source_log_is_archived_and_verified(self):
        source_log = self.results / "legacy.out"
        source_log.write_text("legacy output\n")

        plan = self._plan(source_logs=[source_log])
        attestation = apply_migration(plan)

        log = attestation["source"]["logs"][0]
        archived = (
            self.destination.parent
            / f"{self.destination.name}.legacy_raw"
            / log["archive_name"]
        )
        self.assertEqual(archived.read_bytes(), source_log.read_bytes())

    def test_unrelated_malformed_provenance_does_not_break_witness_scan(self):
        unrelated = {
            **{field: self.legacy[field] for field in (
                "soc_step", "block_min", "g_kwh", "charge_kw",
                "min_soc_frac", "master_sense",
            )},
            "csv": "instance.csv",
            "prices_csv": "prices.csv",
            "trip_ids": [1, 2],
            "stop_reason": "certified",
            "provenance": "not-an-object",
        }
        (self.results / "malformed_provenance.json").write_text(
            json.dumps(unrelated)
        )

        plan = self._plan()

        self.assertEqual(plan["instance_hash"], self.instance_hash)


if __name__ == "__main__":
    unittest.main()
