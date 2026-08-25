import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "scripts" / "event_uniform_envelope"


def run_tool(name, *args):
    subprocess.run(
        [sys.executable, str(TOOLS / name), *map(str, args)],
        check=True,
    )


def test_audit_panel_a_writes_normalized_csv(tmp_path):
    root = tmp_path / "panel_a"
    for name in ("cg", "fleet_lp", "mip", "target", "logs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "matrix.tsv").write_text(
        "0\tk02_s1\t2\tinstance.csv\tevent_2p5_event5\tevent\t2.5\t5\n"
    )
    (root / "jobs.tsv").write_text(
        "stage\tarray_job_id\ttasks\n"
        "combined_cost_cg\t100\t1\n"
        "fleet_lp_phase2\t101\t1\n"
        "raw_pool_mip\t102\t1\n"
        "target_feasibility\t103\t1\n"
    )
    stem = "A__k02_s1__event_2p5_event5.json"
    (root / "cg" / stem).write_text(json.dumps({
        "stop_reason": "certified",
        "certified_rc_optimal": True,
        "iterations": 12,
        "columns": 34,
        "wall_s": 56.0,
        "peak_rss_mb": 78.0,
        "final": {"route_weight": 2.0, "min_rc": 0.0},
        "network_metrics": {"dag_nodes": 10, "dag_arcs": 20},
    }))
    (root / "fleet_lp" / stem).write_text(json.dumps({
        "fleet_lp_lower_bound": 2.0,
        "wall_s": 3.0,
        "peak_rss_mb": 4.0,
        "certificate": {
            "certified": True,
            "stop_reason": "certified",
            "iterations": 2,
            "minimum_reduced_cost": 0.0,
        },
    }))
    (root / "mip" / stem).write_text(json.dumps({
        "status_name": "OPTIMAL", "buses": 2, "mip_bound": 2,
        "mip_gap": 0, "fleet_proven": True, "runtime_s": 1.0,
    }))
    (root / "target" / stem).write_text(json.dumps({
        "outcome": "FEASIBLE",
        "solver": {"runtime_s": 0.5, "backend": "gurobi"},
    }))
    (root / "logs" / "mipr_102_0.err").write_text(
        "Traceback (most recent call last):\nRuntimeError: example failure\n"
    )
    sacct = root / "slurm.psv"
    sacct.write_text(
        "100_0|cg|COMPLETED|0:0|00:01|1G||24G|node\n"
        "101_0|l2|COMPLETED|0:0|00:01|1G||24G|node\n"
        "102_0|mip|COMPLETED|0:0|00:01|1G||24G|node\n"
        "103_0|tf|COMPLETED|0:0|00:01|1G||24G|node\n"
    )

    run_tool("audit_panel_a.py", "--root", root, "--sacct", sacct)
    row = next(csv.DictReader((root / "panel_a_summary.csv").open()))
    assert row["cg_certified"] == "True"
    assert row["fleet_lp_lower_bound"] == "2.0"
    assert row["mip_status"] == "OPTIMAL"
    assert row["target_runtime_s"] == "0.5"
    assert row["cg_slurm_state"] == "COMPLETED"
    signature = next(csv.DictReader((root / "stderr_signatures.csv").open()))
    assert signature["stage"] == "mipr"
    assert signature["count"] == "1"
    assert signature["last_line"] == "RuntimeError: example failure"


def test_prepare_integer_manifest_hashes_status_and_journal(tmp_path):
    root = tmp_path / "panel_a"
    (root / "cg").mkdir(parents=True)
    (root / "matrix.tsv").write_text(
        "0\tk02_s1\t2\tinstance.csv\tevent_2p5_event5\tevent\t2.5\t5\n"
    )
    journal = root / "cg" / "pool.columns.jsonl"
    journal.write_text('{"trips":[1],"cost":1}\n')
    status = root / "cg" / "A__k02_s1__event_2p5_event5.json"
    status.write_text(json.dumps({"columns_journal": str(journal)}))
    output = root / "manifest.tsv"
    provenance = root / "provenance.json"

    run_tool(
        "prepare_integer_manifest.py",
        "--root", root,
        "--panel", "A",
        "--source-dir", "cg",
        "--out", output,
        "--provenance", provenance,
        "--wrapper-commit", "a" * 40,
        "--solver-commit", "b" * 40,
    )
    row = next(csv.DictReader(output.open(), delimiter="\t"))
    assert b"\r" not in output.read_bytes()
    assert len(row["source_status_sha256"]) == 64
    assert len(row["source_journal_sha256"]) == 64
    assert json.loads(provenance.read_text())["source_rows"] == 1


def test_prepare_panel_b_writes_five_uniform_arms_per_event_cell(tmp_path):
    panel_a = tmp_path / "panel_a"
    (panel_a / "cg").mkdir(parents=True)
    (panel_a / "panel_b_gate.json").write_text(json.dumps({"eligible": True}))
    rows = []
    representations = [
        ("event_2p5_event5", "event", "2.5", "5"),
        ("uniform_10_10", "uniform", "10", "10"),
        ("uniform_4_5", "uniform", "4", "5"),
        ("uniform_2_5", "uniform", "2", "5"),
        ("uniform_2_2", "uniform", "2", "2"),
        ("uniform_2_1", "uniform", "2", "1"),
    ]
    index = 0
    for cell_index in range(9):
        cell = f"k02_s{cell_index + 1}"
        for rep, time_model, soc, block in representations:
            rows.append([
                index, cell, 2, "instance.csv", rep,
                time_model, soc, block,
            ])
            index += 1
        event_status = panel_a / "cg" / f"A__{cell}__event_2p5_event5.json"
        event_status.write_text(json.dumps({
            "certified_rc_optimal": True,
            "stop_reason": "certified",
            "wall_s": 100.25 + cell_index,
            "peak_rss_mb": 200,
            "iterations": 20,
            "columns_journal": str(panel_a / "cg" / f"{cell}.jsonl"),
        }))
    with (panel_a / "matrix.tsv").open("w", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(rows)
    panel_b = tmp_path / "panel_b"

    run_tool(
        "prepare_panel_b.py",
        "--panel-a", panel_a,
        "--panel-b", panel_b,
        "--execution-repo", REPO,
        "--commit", "c" * 40,
    )
    output_rows = list(csv.reader((panel_b / "matrix.tsv").open(), delimiter="\t"))
    assert len(output_rows) == 45
    assert all(row[4].startswith("uniform_") for row in output_rows)
    assert json.loads((panel_b / "execution_plan.json").read_text())["uniform_arms"] == 45


def test_audit_panel_b_compares_v1_and_v2_frozen_pools(tmp_path):
    root = tmp_path / "panel_b"
    for name in ("cg", "frozen", "frozen_v2"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "matrix.tsv").write_text(
        "0\tk02_s1\t2\tinstance.csv\tuniform_2_2\t2\t2\t10.5\t71\n"
    )
    (root / "jobs.tsv").write_text(
        "stage\tarray_job_id\ttasks\n"
        "cg\t200\t1\nfreeze\t201\t1\n"
    )
    (root / "refreeze_v2_jobs.tsv").write_text(
        "stage\tarray_job_id\ttasks\tsolver_commit\n"
        "freeze_v2\t202\t1\t" + "a" * 40 + "\n"
    )
    stem = "B__k02_s1__uniform_2_2.json"
    (root / "cg" / stem).write_text(json.dumps({
        "stop_reason": "wall_limit", "certified_rc_optimal": False,
    }))
    (root / "frozen" / stem).write_text(json.dumps({
        "iterations": 3, "columns": 10,
    }))
    (root / "frozen_v2" / stem).write_text(json.dumps({
        "iterations": 3, "columns": 12,
    }))
    sacct = root / "slurm.psv"
    sacct.write_text(
        "200_0|cg|COMPLETED|0:0|00:01|1G||24G|node\n"
        "201_0|freeze|COMPLETED|0:0|00:01|1G||4G|node\n"
        "202_0|freeze2|COMPLETED|0:0|00:01|1G||4G|node\n"
    )

    run_tool("audit_panel_b.py", "--root", root, "--sacct", sacct)
    row = next(csv.DictReader((root / "panel_b_summary.csv").open()))
    assert row["frozen_v2_present"] == "True"
    assert row["frozen_v2_column_delta"] == "2"
    assert row["freeze_v2_slurm_state"] == "COMPLETED"


def test_select_missing_integer_indices_validates_json_semantics(tmp_path):
    root = tmp_path / "panel_a"
    (root / "mip").mkdir(parents=True)
    (root / "target").mkdir()
    manifest = root / "manifest.tsv"
    manifest.write_text(
        "index\tcell\ttarget_fleet\trepresentation_id\tsource_status\t"
        "source_status_sha256\tsource_journal\tsource_journal_sha256\n"
        "0\tk02_s1\t2\tevent\t/a\t" + "a" * 64 + "\t/b\t" + "b" * 64 + "\n"
        "1\tk02_s2\t2\tevent\t/c\t" + "c" * 64 + "\t/d\t" + "d" * 64 + "\n"
    )
    (root / "mip" / "A__k02_s1__event.json").write_text(
        json.dumps({
            "status_name": "OPTIMAL",
            "source_result_sha256": "a" * 64,
            "source_journal_sha256": "b" * 64,
        })
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "select_missing_integer_indices.py"),
            "--manifest", str(manifest),
            "--root", str(root),
            "--panel", "A",
            "--stage", "mip",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert completed.stdout.splitlines() == ["1"]
    assert "missing=1" in completed.stderr
    assert "valid=1" in completed.stderr
    (root / "mip" / "A__k02_s2__event.json").write_text(json.dumps({
        "status_name": "TIME_LIMIT",
        "source_result_sha256": "c" * 64,
        "source_journal_sha256": "d" * 64,
    }))
    completed = subprocess.run(completed.args, check=True, text=True,
                               capture_output=True)
    assert completed.stdout == ""
    assert "valid=2" in completed.stderr
    (root / "mip" / "A__k02_s2__event.json").write_text(json.dumps({
        "status_name": "TIME_LIMIT",
        "source_result_sha256": "x" * 64,
        "source_journal_sha256": "d" * 64,
    }))
    completed = subprocess.run(completed.args, check=False, text=True,
                               capture_output=True)
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "1:identity_mismatch" in completed.stderr


def test_select_missing_frozen_v2_indices_checks_source_identity(tmp_path):
    root = tmp_path / "panel_b"
    (root / "cg").mkdir(parents=True)
    output_dir = root / "frozen_v2"
    output_dir.mkdir()
    (root / "matrix.tsv").write_text(
        "0\tk02_s1\t2\tinstance.csv\tuniform_2_2\t2\t2\t10\t70\n"
    )
    source = root / "cg" / "B__k02_s1__uniform_2_2.json"
    source_journal = Path(str(source) + ".columns.jsonl")
    source_journal.write_text('{"trips":[1],"found_iter":0,"cost":1}\n')
    source.write_text(json.dumps({"columns_journal": str(source_journal)}))
    command = [
        sys.executable,
        str(TOOLS / "select_missing_frozen_v2_indices.py"),
        "--root", str(root),
        "--output-dir", str(output_dir),
    ]
    completed = subprocess.run(command, check=True, text=True,
                               capture_output=True)
    assert completed.stdout.splitlines() == ["0"]
    output = output_dir / "B__k02_s1__uniform_2_2.json"
    output_journal = Path(str(output) + ".columns.jsonl")
    output_journal.write_text(source_journal.read_text())
    Path(str(output) + ".iters.csv").write_text("iteration\n0\n")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    output.write_text(json.dumps({
        "stop_reason": "matched_wall_snapshot",
        "columns_journal": str(output_journal),
        "resume_parent": {
            "source_status_sha256": digest(source),
            "source_journal_sha256": digest(source_journal),
        },
        "matched_wall_snapshot": {
            "schema": "evsp-dr-exact-cg-matched-wall-snapshot-v2",
            "conservative_boundary":
                "include_columns_only_through_last_durably_completed_iteration",
            "journal_record_count": 1,
        },
    }))
    completed = subprocess.run(command, check=True, text=True,
                               capture_output=True)
    assert completed.stdout == ""
    assert "valid=1" in completed.stderr
