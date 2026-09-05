import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "scripts" / "event_uniform_envelope"


def make_source(root: Path, cell: str) -> tuple[Path, Path]:
    instance = root / f"{cell}.csv"
    instance.write_text("trip\n1\n")
    status = root / f"M__{cell}__event_2p5_event5.json"
    journal = Path(str(status) + ".columns.jsonl")
    iterations = Path(str(status) + ".iters.csv")
    journal.write_text("".join((
        json.dumps({
            "trips": [1], "cost": 100001.0, "found_iter": 0,
        }) + "\n",
        # A legitimate cheaper replacement has the same incidence. The
        # bounded-memory freezer must retain both journal records while
        # counting one unique pool column.
        json.dumps({
            "trips": [1], "cost": 100000.0, "found_iter": 0,
        }) + "\n",
        # This record was found by the selected iteration and is outside the
        # conservative state represented by that iteration-log row.
        json.dumps({
            "trips": [2], "cost": 100000.0, "found_iter": 1,
        }) + "\n",
    )))
    status.write_text(json.dumps({
        "csv": str(instance.resolve()),
        "prices_csv": "hourly_prices_flat.csv",
        "soc_step": 2.5,
        "block_min": 5,
        "g_kwh": 240.0,
        "charge_kw": 240.0,
        "min_soc_frac": 0.0,
        "master_sense": "partition",
        "initial_pool": "singletons",
        "time_model": "event",
        "columns_per_iter": 30,
        "column_pool_treatment": "RAW",
        "network_metrics": {"arc_mode": "lazy"},
        "trip_ids": [1],
        "columns": 1,
        "columns_journal": str(journal),
        "provenance": {
            "instance_sha256": "a" * 64,
            "prices_sha256": "b" * 64,
            "reference_sha256": "c" * 64,
            "deadhead_sha256": "d" * 64,
        },
    }))
    with iterations.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "elapsed_s", "iteration", "lp_obj", "route_weight",
            "artificials", "min_rc", "pool_columns",
        ))
        writer.writeheader()
        writer.writerow({
            "elapsed_s": 129550,
            "iteration": 1,
            "lp_obj": 500001,
            "route_weight": 5,
            "artificials": 0,
            "min_rc": -2,
            "pool_columns": 1,
        })
    return status, instance


def prepare_four_predeclared_raw_snapshots(tmp_path: Path):
    resume = tmp_path / "resume"
    resume.mkdir()
    cells = ("k05_p5", "k05_xenergy", "k05_xgap", "k05_xtrip")
    rows = []
    for index, cell in enumerate(cells):
        status, instance = make_source(resume, cell)
        rows.append({
            "local_index": str(index),
            "source_panel_index": str(index + 14),
            "cell": cell,
            "target_fleet": "5",
            "instance_csv": str(instance.resolve()),
            "representation_id": "event_2p5_event5",
            "soc_step": "2.5",
            "block_min": "5",
            "source_status": str(status),
            "source_status_sha256": "unused",
            "source_journal": str(status) + ".columns.jsonl",
            "source_journal_sha256": "unused",
            "resume_status": str(status),
            "resume_journal": str(status) + ".columns.jsonl",
            "staged_status_sha256": "unused",
            "staged_journal_sha256": "unused",
        })
    with (resume / "matrix.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    (resume / "execution_plan.json").write_text(json.dumps({
        "schema": "evsp-dr-wall-capped-event-resume-v1",
        "cells": 23,
        "parent_cumulative_wall_limit_s": 43200.0,
        "cumulative_scientific_wall_limit_s": 172800.0,
        "columns_per_iter": 30,
    }))

    # Exercise the historical periodic-status shape that omitted the field;
    # the immutable resume plan remains authoritative.
    first_status = Path(rows[0]["resume_status"])
    first_payload = json.loads(first_status.read_text())
    first_payload.pop("columns_per_iter")
    first_status.write_text(json.dumps(first_payload))

    output = tmp_path / "mip36"
    subprocess.run([
        sys.executable,
        str(TOOLS / "prepare_k5_raw_mip36h.py"),
        "--resume-root", str(resume),
        "--output-root", str(output),
        "--freezer", str(REPO / "src" / "freeze_exact_cg_prefix.py"),
        "--python", sys.executable,
    ], check=True)

    with (output / "snapshot_manifest.csv").open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    assert [row["cell"] for row in manifest] == sorted(cells)
    assert all(row["route_weight_endpoint"] == "5.0" for row in manifest)
    assert all(row["artificials"] == "0.0" for row in manifest)
    assert all(row["pool_columns"] == "1" for row in manifest)
    for row in manifest:
        snapshot = json.loads(Path(row["snapshot"]).read_text())
        assert snapshot["matched_wall_snapshot"]["requested_budget_s"] == 129600
        assert snapshot["matched_wall_snapshot"]["journal_record_count"] == 2
        assert snapshot["matched_wall_snapshot"]["unique_pool_columns"] == 1
        assert Path(snapshot["columns_journal"]).is_file()


def prepare_rejects_non_raw_source(tmp_path: Path):
    status, instance = make_source(tmp_path, "k05_p5")
    payload = json.loads(status.read_text())
    payload["column_pool_treatment"] = "KNOWN"
    status.write_text(json.dumps(payload))
    sys.path.insert(0, str(TOOLS))
    from prepare_k5_raw_mip36h import validate_source

    row = {
        "cell": "k05_p5",
        "target_fleet": "5",
        "instance_csv": str(instance.resolve()),
        "representation_id": "event_2p5_event5",
        "resume_status": str(status),
    }
    try:
        validate_source(row, 129600)
    except SystemExit as exc:
        assert "configuration mismatch" in str(exc)
    else:
        raise AssertionError("KNOWN pool was accepted as RAW")


def audit_writes_pool_scoped_result(tmp_path: Path):
    root = tmp_path / "audit"
    (root / "mip").mkdir(parents=True)
    commit = "e" * 40
    fields = (
        "local_index", "source_panel_index", "cell", "target_fleet",
        "representation_id", "source_status", "snapshot", "snapshot_sha256",
        "journal", "journal_sha256", "iteration", "elapsed_s",
        "pool_columns", "route_weight_endpoint", "artificials", "min_rc",
        "lp_obj", "instance", "instance_sha256",
    )
    sources = []
    for index, cell in enumerate((
        "k05_p5", "k05_xenergy", "k05_xgap", "k05_xtrip",
    )):
        source = dict.fromkeys(fields, "")
        source.update({
            "local_index": str(index),
            "source_panel_index": str(index + 14),
            "cell": cell,
            "target_fleet": "5",
            "representation_id": "event_2p5_event5",
            "snapshot_sha256": "a" * 64,
            "journal_sha256": "b" * 64,
        })
        sources.append(source)
        result = {
            "source_result_sha256": "a" * 64,
            "source_journal_sha256": "b" * 64,
            "incumbent_found": True,
            "partitioning": True,
            "selected_routes": [
                {
                    "trips": [item],
                    "physical_realization": {"status": "valid_as_recorded_mapped"},
                }
                for item in range(5)
            ],
            "buses": 5,
            "overcovered_trips": 0,
            "fleet_bound": 5.0,
            "fleet_proven": True,
            "status_name": "OPTIMAL",
            "optimal_scope": "finite_pool_fleet",
            "physical_pool_audit": {
                "accepted_columns": 10,
                "rejected_columns": 0,
                "added_giro_route_count": 0,
                "post_augmentation_columns": 10,
            },
            "pool_columns": 10,
            "experiment_arm": "B",
            "extra_route_sources": [],
            "mip_provenance": {
                "expected_git_commit": commit,
                "observed_git_commit": commit,
                "final_observed_git_commit": commit,
                "git_detached": True,
                "git_dirty": False,
            },
        }
        result_path = (
            root / "mip"
            / f"M__{cell}__event_2p5_event5.raw_pool_mip36h.json"
        )
        result_path.write_text(json.dumps(result))
    with (root / "snapshot_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sources)
    (root / "jobs_123.tsv").write_text(
        "stage\tarray_job_id\twrapper_commit\n"
        f"k5_raw_mip36h\t123\t{commit}\n"
    )
    sacct = root / "sacct.psv"
    sacct.write_text("".join(
        f"123_{index}|job|COMPLETED|0:0|00:30:00|01:00:00|||node\n"
        f"123_{index}.batch|batch|COMPLETED|0:0|00:30:00|00:59:00|1G|2G|node\n"
        for index in range(4)
    ))
    subprocess.run([
        sys.executable, str(TOOLS / "audit_k5_raw_mip36h.py"),
        "--root", str(root), "--sacct", str(sacct),
    ], check=True)
    with (root / "k5_raw_mip36h_summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert all(row["fleet_proven_over_pool"] == "True" for row in rows)
    assert all(row["physical_witness_valid"] == "True" for row in rows)
    assert all(row["experiment_configuration_valid"] == "True" for row in rows)
    assert all(row["source_hash_match"] == "True" for row in rows)
    assert all(row["slurm_max_rss"] == "1G" for row in rows)


class K5RawMip36hTests(unittest.TestCase):
    def test_prepare_four_predeclared_raw_snapshots(self):
        with tempfile.TemporaryDirectory() as folder:
            prepare_four_predeclared_raw_snapshots(Path(folder))

    def test_prepare_rejects_non_raw_source(self):
        with tempfile.TemporaryDirectory() as folder:
            prepare_rejects_non_raw_source(Path(folder))

    def test_audit_writes_pool_scoped_result(self):
        with tempfile.TemporaryDirectory() as folder:
            audit_writes_pool_scoped_result(Path(folder))


if __name__ == "__main__":
    unittest.main()
