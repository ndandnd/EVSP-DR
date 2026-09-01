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
    for name in ("cg", "frozen", "frozen_v2", "mip", "target", "logs"):
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
    (root / "integer_v2_jobs.tsv").write_text(
        "stage\tarray_job_id\ttasks\tsolver_commit\n"
        "mip_v2\t203\t1\t" + "a" * 40 + "\n"
        "target_v2\t204\t1\t" + "a" * 40 + "\n"
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
    (root / "mip" / stem).write_text(json.dumps({
        "status_name": "OPTIMAL", "buses": 2, "mip_bound": 2,
        "mip_gap": 0, "fleet_proven": True,
        "optimality_scope": "full_pool_lexicographic",
        "runtime_s": 1.0, "peak_rss_mb": 5.0,
        "physical_witness_valid": True,
        "source_result_sha256": "b" * 64,
        "source_journal_sha256": "c" * 64,
    }))
    (root / "target" / stem).write_text(json.dumps({
        "outcome": "FEASIBLE",
        "solver": {"runtime_s": 0.5, "backend": "gurobi"},
        "source": {
            "result_sha256": "b" * 64,
            "journal_sha256": "c" * 64,
        },
    }))
    (root / "logs" / "mip2_203_0.err").write_text(
        "RuntimeError: example panel B error\n"
    )
    sacct = root / "slurm.psv"
    sacct.write_text(
        "200_0|cg|COMPLETED|0:0|00:01|1G||24G|node\n"
        "201_0|freeze|COMPLETED|0:0|00:01|1G||4G|node\n"
        "202_0|freeze2|COMPLETED|0:0|00:01|1G||4G|node\n"
        "203_0|mip2|COMPLETED|0:0|00:01|2G||24G|node\n"
        "204_0|tf2|COMPLETED|0:0|00:01|2G||24G|node\n"
    )

    run_tool("audit_panel_b.py", "--root", root, "--sacct", sacct)
    row = next(csv.DictReader((root / "panel_b_summary.csv").open()))
    assert row["frozen_v2_present"] == "True"
    assert row["frozen_v2_column_delta"] == "2"
    assert row["freeze_v2_slurm_state"] == "COMPLETED"
    assert row["mip_status"] == "OPTIMAL"
    assert row["finite_pool_proven"] == "True"
    assert row["target_outcome"] == "FEASIBLE"
    assert row["target_solver"] == "gurobi"
    assert row["mip_slurm_state"] == "COMPLETED"
    stages = {
        row["stage"]: row
        for row in csv.DictReader((root / "panel_b_stage_counts.csv").open())
    }
    assert stages["mip_v2"]["artifact_rows"] == "1"
    signature = next(csv.DictReader((root / "stderr_signatures.csv").open()))
    assert signature["stage"] == "mip2"
    assert signature["count"] == "1"


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


def test_prepare_and_select_extended_cg_resume_preserves_sources(tmp_path):
    root = tmp_path / "panel_a"
    (root / "cg").mkdir(parents=True)
    matrix_rows = []
    source_bytes = {}
    for index, cell in enumerate(("k05_s1", "k05_s3")):
        rep = "uniform_2_1"
        stem = f"A__{cell}__{rep}.json"
        status_path = root / "cg" / stem
        journal = Path(str(status_path) + ".columns.jsonl")
        journal.write_text('{"trips":[1],"cost":1}\n')
        Path(str(status_path) + ".iters.csv").write_text(
            "elapsed_s,iteration,lp_obj,route_weight,artificials,min_rc,pool_columns\n"
            "100,1,2,2,0,-1,5\n"
        )
        source_telemetry = Path(
            str(status_path) + ".phase-telemetry.jsonl"
        )
        source_telemetry.write_text('{"source":"telemetry"}\n')
        status_path.write_text(json.dumps({
            "stop_reason": "wall_limit",
            "certified_rc_optimal": False,
            "wall_s": 43200,
            "iterations": 1,
            "columns": 5,
            "columns_journal": str(journal),
        }))
        source_bytes[status_path] = status_path.read_bytes()
        source_bytes[journal] = journal.read_bytes()
        source_bytes[source_telemetry] = source_telemetry.read_bytes()
        matrix_rows.append([
            index, cell, 5, "instance.csv", rep, "uniform", 2, 1,
        ])
    with (root / "matrix.tsv").open("w", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(
            matrix_rows
        )
    resume = root / "cg_resume24h"

    run_tool(
        "prepare_cg_resume.py",
        "--root", root,
        "--out-root", resume,
        "--panel", "A",
        "--representation", "uniform_2_1",
        "--expected-cells", "2",
        "--wall-limit-s", "86400",
        "--solver-commit", "a" * 40,
    )
    for path, payload in source_bytes.items():
        assert path.read_bytes() == payload
    manifest = list(csv.DictReader(
        (resume / "matrix.tsv").open(), delimiter="\t"
    ))
    assert len(manifest) == 2
    for row in manifest:
        status = Path(row["resume_status"])
        archived = Path(str(status) + ".source-phase-telemetry.jsonl")
        live = Path(str(status) + ".phase-telemetry.jsonl")
        assert archived.is_file()
        assert not live.exists()
        archived.rename(live)
    run_tool(
        "repair_cg_resume_telemetry.py",
        "--resume-root", resume,
    )
    repair_rows = list(csv.DictReader(
        (resume / "telemetry_repair.csv").open()
    ))
    assert {row["action"] for row in repair_rows} == {
        "archived_misplaced_source_copy"
    }
    command = [
        sys.executable,
        str(TOOLS / "select_cg_resume_indices.py"),
        "--resume-root", str(resume),
    ]
    completed = subprocess.run(
        command, check=True, text=True, capture_output=True
    )
    assert completed.stdout.splitlines() == ["0", "1"]
    first = Path(manifest[0]["resume_status"])
    first_payload = json.loads(first.read_text())
    first_payload.update({
        "certified_rc_optimal": True,
        "stop_reason": "certified",
        "wall_s": 50000,
    })
    first.write_text(json.dumps(first_payload))
    second = Path(manifest[1]["resume_status"])
    second_payload = json.loads(second.read_text())
    # The solver reserves 60 seconds for durable serialization, so a terminal
    # wall-limit status just below the nominal cap is scientifically capped.
    second_payload["wall_s"] = 86340
    second.write_text(json.dumps(second_payload))
    for row in manifest:
        status = Path(row["resume_status"])
        Path(str(status) + ".phase-telemetry.jsonl").write_text(json.dumps({
            "record_type": "session_start",
            "identity": {
                "output": str(status.resolve()),
                "git_commit": "a" * 40,
            },
        }) + "\n")
    run_tool(
        "repair_cg_resume_telemetry.py",
        "--resume-root", resume,
    )
    repair_rows = list(csv.DictReader(
        (resume / "telemetry_repair.csv").open()
    ))
    assert {row["action"] for row in repair_rows} == {
        "resume_telemetry_active"
    }
    sacct = resume / "slurm.psv"
    sacct.write_text("")
    run_tool(
        "audit_cg_resume.py",
        "--resume-root", resume,
        "--sacct", sacct,
    )
    outcomes = {
        row["local_index"]: row["outcome"]
        for row in csv.DictReader((resume / "resume_summary.csv").open())
    }
    assert outcomes == {"0": "certified", "1": "wall_cap"}
    completed = subprocess.run(
        command, check=True, text=True, capture_output=True
    )
    assert completed.stdout == ""
    assert "certified=1" in completed.stderr
    assert "wall_cap=1" in completed.stderr

    child = root / "cg_resume48h"
    run_tool(
        "prepare_cg_reresume.py",
        "--source-resume-root", resume,
        "--out-root", child,
        "--panel", "A",
        "--representation", "uniform_2_1",
        "--expected-cells", "1",
        "--wall-limit-s", "172800",
        "--max-iters", "50000",
        "--solver-commit", "a" * 40,
    )
    child_rows = list(csv.DictReader(
        (child / "matrix.tsv").open(), delimiter="\t"
    ))
    assert len(child_rows) == 1
    assert child_rows[0]["cell"] == "k05_s3"
    assert Path(child_rows[0]["source_status"]) == second
    assert hashlib.sha256(
        Path(child_rows[0]["resume_status"]).read_bytes()
    ).hexdigest() == child_rows[0]["staged_status_sha256"]
    child_plan = json.loads((child / "execution_plan.json").read_text())
    assert child_plan["parent_cumulative_wall_limit_s"] == 86400
    assert child_plan["cumulative_scientific_wall_limit_s"] == 172800
    assert child_plan["max_iters"] == 50000
    run_tool("repair_cg_resume_telemetry.py", "--resume-root", child)
    child_repair = next(csv.DictReader(
        (child / "telemetry_repair.csv").open()
    ))
    assert child_repair["action"] == "already_archived"


def test_event_audit_and_dynamic_wall_resume_require_lazy_identity(tmp_path):
    root = tmp_path / "medium_event"
    (root / "cg").mkdir(parents=True)
    instance = root / "instance.csv"
    instance.write_text("trip_id\n1\n")
    instance_sha = hashlib.sha256(instance.read_bytes()).hexdigest()
    representation = "event_2p5_event5"
    cell = "k08_s1"
    status = root / "cg" / f"M__{cell}__{representation}.json"
    journal = Path(str(status) + ".columns.jsonl")
    journal.write_text('{"trips":[1],"cost":1}\n')
    Path(str(status) + ".iters.csv").write_text(
        "elapsed_s,iteration\n43200,1\n"
    )
    Path(str(status) + ".phase-telemetry.jsonl").write_text(
        '{"record_type":"phase","phase":"network_build",'
        '"duration_s":1}\n'
    )
    status.write_text(json.dumps({
        "csv": str(instance.resolve()),
        "soc_step": 2.5,
        "block_min": 5,
        "g_kwh": 240.0,
        "charge_kw": 240.0,
        "min_soc_frac": 0.0,
        "time_model": "event",
        "network_metrics": {"arc_mode": "lazy", "dag_nodes": 10,
                            "dag_arcs": 20},
        "stop_reason": "wall_limit",
        "certified_rc_optimal": False,
        "wall_s": 43200,
        "iterations": 1,
        "columns": 5,
        "columns_journal": str(journal),
        "final": {"route_weight": 8.0},
    }))
    (root / "matrix.tsv").write_text(
        f"0\t{cell}\t8\t1\t1\t{instance.resolve()}\t{instance_sha}\t"
        f"{representation}\t2.5\t5\t43200\n"
    )
    (root / "jobs.tsv").write_text(
        "scale\tarray_job_id\tindices\n8\t700\t0\n"
    )
    solver_commit = "44b6d5030a78ddca9c74f582d70ad87572e61794"
    (root / "execution_plan.json").write_text(json.dumps({
        "solver_commit": solver_commit,
        "time_model": "event",
        "event_arc_mode": "lazy",
        "wall_limit_s": 43200,
    }))
    sacct = root / "slurm.psv"
    sacct.write_text(
        "700_0|700|me31_k8|COMPLETED|0:0|12:00:00|12:15:00|"
        "12:00:00|1|1G|2G|32G|node\n"
    )

    run_tool("audit_medium_event_legacy.py", "--root", root,
             "--sacct", sacct)
    audit = next(csv.DictReader(
        (root / "medium_event_summary.csv").open()
    ))
    assert audit["configuration_match"] == "True"

    resume = root / "cg_resume24h_20260831"
    run_tool(
        "prepare_wall_capped_event_resume.py",
        "--source-root", root,
        "--out-root", resume,
        "--solver-commit", solver_commit,
        "--parent-wall-limit-s", "43200",
        "--wall-limit-s", "86400",
    )
    rows = list(csv.DictReader(
        (resume / "matrix.tsv").open(), delimiter="\t"
    ))
    assert len(rows) == 1
    assert rows[0]["cell"] == cell
    assert json.loads(
        (resume / "execution_plan.json").read_text()
    )["cells"] == 1

    explicit = json.loads(status.read_text())
    explicit["network_metrics"]["arc_mode"] = "explicit"
    status.write_text(json.dumps(explicit))
    run_tool("audit_medium_event_legacy.py", "--root", root,
             "--sacct", sacct)
    audit = next(csv.DictReader(
        (root / "medium_event_summary.csv").open()
    ))
    assert audit["configuration_match"] == "False"


def test_audit_backend_reproduction_compares_identical_sources(tmp_path):
    root = tmp_path / "panel_a"
    (root / "mip").mkdir(parents=True)
    (root / "mip_highs_native").mkdir()
    status_hash = "a" * 64
    journal_hash = "b" * 64
    manifest = root / "panel_a_highs_inputs.tsv"
    manifest.write_text(
        "index\tcell\ttarget_fleet\trepresentation_id\tsource_status\t"
        "source_status_sha256\tsource_journal\tsource_journal_sha256\n"
        f"0\tk02_s1\t2\tevent\t/a\t{status_hash}\t/b\t{journal_hash}\n"
    )
    stem = "A__k02_s1__event.json"
    common = {
        "status_name": "OPTIMAL",
        "buses": 2,
        "fleet_bound": 2,
        "mip_gap": 0,
        "fleet_proven": True,
        "optimality_scope": "full_pool_lexicographic",
        "runtime_s": 2.0,
        "physical_witness_valid": True,
        "source_result_sha256": status_hash,
        "source_journal_sha256": journal_hash,
    }
    (root / "mip" / stem).write_text(json.dumps({
        **common, "backend": "gurobi", "runtime_s": 1.0,
    }))
    (root / "mip_highs_native" / stem).write_text(json.dumps({
        **common, "backend": "highspy_native",
    }))
    (root / "highs_native_300_jobs.tsv").write_text(
        "stage\tarray_job_id\ttasks\n"
        "mip_highs_native\t300\t1\n"
    )
    sacct = root / "long_fill.psv"
    sacct.write_text(
        "300_0|highs|COMPLETED|0:0|00:02|2G||24G|node\n"
        "300_0.batch|batch|COMPLETED|0:0|00:02|3G||24G|node\n"
    )
    output = root / "backend_reproduction.csv"

    run_tool(
        "audit_backend_reproduction.py",
        "--root", root,
        "--panel", "A",
        "--manifest", manifest,
        "--sacct", sacct,
        "--out", output,
    )
    row = next(csv.DictReader(output.open()))
    assert row["fleet_agreement"] == "True"
    assert row["proof_agreement"] == "True"
    assert row["highs_source_hash_match"] == "True"
    assert row["runtime_ratio_highs_over_gurobi"] == "2.0"
    assert row["highs_slurm_state"] == "COMPLETED"
    assert row["highs_slurm_max_rss"] == "3G"


def test_audit_highs_retry_records_proof_and_slurm_stats(tmp_path):
    panel_roots = {}
    status_hash = "a" * 64
    journal_hash = "b" * 64
    solver_commit = "44b6d5030a78ddca9c74f582d70ad87572e61794"
    for panel, manifest_name, index in (
        ("A", "panel_a_highs_inputs.tsv", "38"),
        ("B", "panel_b_highs_inputs.tsv", "31"),
    ):
        root = tmp_path / f"panel_{panel.lower()}"
        panel_roots[panel] = root
        for name in ("mip", "mip_highs_native", "mip_highs_native_retry7200"):
            (root / name).mkdir(parents=True, exist_ok=True)
        (root / manifest_name).write_text(
            "index\tcell\ttarget_fleet\trepresentation_id\tsource_status\t"
            "source_status_sha256\tsource_journal\tsource_journal_sha256\n"
            f"{index}\tk05_s1\t5\tuniform_4_5\t/a\t{status_hash}\t/b\t{journal_hash}\n"
        )
        stem = f"{panel}__k05_s1__uniform_4_5.json"
        common = {
            "buses": 6,
            "fleet_bound": 6.0,
            "fleet_proven": True,
            "runtime_s": 100.0,
            "source_result_sha256": status_hash,
            "source_journal_sha256": journal_hash,
            "code_identity": {
                "expected_commit": solver_commit,
                "observed_commit": solver_commit,
            },
        }
        (root / "mip" / stem).write_text(json.dumps({
            **common,
            "partitioning": True,
            "incumbent_found": True,
            "selected_routes": [0, 1, 2, 3, 4, 5],
            "status_name": "OPTIMAL",
            "optimal_scope": "full_pool_lexicographic",
        }))
        (root / "mip_highs_native" / stem).write_text(json.dumps({
            **common,
            "buses": 7,
            "fleet_bound": 5.0,
            "fleet_proven": False,
            "status_name": "TIME_LIMIT_OR_ITERATION_LIMIT",
            "backend": "highspy_native",
            "physical_witness_valid": True,
            "requested_timelimit_s": 1800,
            "threads_requested": 8,
        }))
        (root / "mip_highs_native_retry7200" / stem).write_text(json.dumps({
            **common,
            "status_name": (
                "OPTIMAL" if panel == "A"
                else "TIME_LIMIT_OR_ITERATION_LIMIT"
            ),
            "fleet_bound": 6.0 if panel == "A" else 5.5,
            "fleet_proven": panel == "A",
            "backend": "highspy_native",
            "physical_witness_valid": True,
            "requested_timelimit_s": 7200,
            "threads_requested": 8,
        }))
    jobs = (
        "panel\tarray_job_id\tindices\tsolver_commit\tbackend\t"
        "timelimit_s\tthreads\tpartition\n"
        f"A\t400\t38\t{solver_commit}\thighspy_native\t7200\t8\tscaglione\n"
        f"B\t401\t31\t{solver_commit}\thighspy_native\t7200\t8\tscaglione\n"
    )
    for root in panel_roots.values():
        (root / "highs_disagreement_retry_jobs.tsv").write_text(jobs)
    sacct = tmp_path / "retry.psv"
    sacct.write_text(
        "400_38|402|eua27_h2|COMPLETED|0:0|01:00:00|02:30:00|"
        "07:30:00|8||||scaglione-cpu-01\n"
        "400_38.batch|402.batch|batch|COMPLETED|0:0|01:00:00|"
        "|07:30:00|8|4G|8G||scaglione-cpu-01\n"
        "401_31|403|eub27_h2|COMPLETED|0:0|01:00:00|02:30:00|"
        "07:30:00|8||||scaglione-cpu-02\n"
        "401_31.batch|403.batch|batch|COMPLETED|0:0|01:00:00|"
        "|07:30:00|8|5G|9G||scaglione-cpu-02\n"
    )
    run_tool(
        "audit_highs_disagreement_retry.py",
        "--panel-a", panel_roots["A"],
        "--panel-b", panel_roots["B"],
        "--sacct", sacct,
    )
    a_row = next(csv.DictReader(
        (panel_roots["A"] / "backend_retry7200.csv").open()
    ))
    assert a_row["classification"] == "proven_fleet_agreement"
    assert a_row["retry_progress"] == "became_proven"
    assert a_row["highs30_buses"] == "7"
    assert a_row["highs2_buses"] == "6"
    assert a_row["highs2_requested_timelimit_s"] == "7200"
    assert a_row["highs2_configuration_match"] == "True"
    assert a_row["highs2_became_proven"] == "True"
    assert a_row["highs2_incumbent_changed"] == "True"
    assert a_row["highs2_slurm_state"] == "COMPLETED"
    assert a_row["highs2_slurm_max_rss"] == "4G"
    unresolved = list(csv.DictReader(
        (panel_roots["A"] / "backend_retry7200_unresolved.csv").open()
    ))
    assert unresolved == []
    a_selection = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "select_highs_unresolved_retry_indices.py"),
            "--root", str(panel_roots["A"]), "--panel", "A",
        ],
        check=True, text=True, capture_output=True,
    )
    b_selection = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "select_highs_unresolved_retry_indices.py"),
            "--root", str(panel_roots["B"]), "--panel", "B",
        ],
        check=True, text=True, capture_output=True,
    )
    assert a_selection.stdout == ""
    assert b_selection.stdout == "31\n"

    b_root = panel_roots["B"]
    b_stem = "B__k05_s1__uniform_4_5.json"
    (b_root / "mip_highs_native_retry28800").mkdir()
    two_hour = json.loads(
        (b_root / "mip_highs_native_retry7200" / b_stem).read_text()
    )
    (b_root / "mip_highs_native_retry28800" / b_stem).write_text(json.dumps({
        **two_hour,
        "status_name": "OPTIMAL",
        "fleet_bound": 6.0,
        "fleet_proven": True,
        "requested_timelimit_s": 28800,
    }))
    (b_root / "highs_unresolved_retry28800_jobs.tsv").write_text(
        "panel\tarray_job_id\tindices\twrapper_commit\tsolver_commit\t"
        "backend\ttimelimit_s\tthreads\tpartition\n"
        f"B\t500\t31\t{'d' * 40}\t{solver_commit}\t"
        "highspy_native\t28800\t8\tscaglione\n"
    )
    eight_sacct = tmp_path / "eight.psv"
    eight_sacct.write_text(
        "500_31|501|eub27_h8|COMPLETED|0:0|04:00:00|08:30:00|"
        "31:00:00|8||||scaglione-cpu-01\n"
        "500_31.batch|501.batch|batch|COMPLETED|0:0|04:00:00|"
        "|31:00:00|8|6G|10G||scaglione-cpu-01\n"
    )
    run_tool(
        "audit_highs_unresolved_retry28800.py",
        "--root", b_root, "--panel", "B", "--sacct", eight_sacct,
    )
    eight_row = next(csv.DictReader(
        (b_root / "backend_retry28800.csv").open()
    ))
    assert eight_row["classification"] == "proven_fleet_agreement"
    assert eight_row["retry_progress_from_2h"] == "became_proven"
    assert eight_row["highs8_slurm_state"] == "COMPLETED"
    assert eight_row["highs8_slurm_max_rss"] == "6G"


def test_select_24h_highs_retry_uses_only_safe_unresolved_rows(tmp_path):
    root = tmp_path / "panel_a"
    root.mkdir()
    (root / "highs_unresolved_retry28800_jobs.tsv").write_text(
        "panel\tarray_job_id\tindices\nA\t600\t1,2\n"
    )
    fields = [
        "index", "classification", "highs8_present",
        "highs8_physical_witness_valid", "highs8_source_hash_match",
        "highs8_configuration_match", "highs8_slurm_state",
        "highs8_slurm_exit",
    ]
    with (root / "backend_retry28800.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "index": "1", "classification": "both_unproven",
            "highs8_present": True,
            "highs8_physical_witness_valid": True,
            "highs8_source_hash_match": True,
            "highs8_configuration_match": True,
            "highs8_slurm_state": "COMPLETED",
            "highs8_slurm_exit": "0:0",
        })
        writer.writerow({
            "index": "2", "classification": "proven_fleet_agreement",
            "highs8_present": True,
            "highs8_physical_witness_valid": True,
            "highs8_source_hash_match": True,
            "highs8_configuration_match": True,
            "highs8_slurm_state": "COMPLETED",
            "highs8_slurm_exit": "0:0",
        })
    completed = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "select_highs_unresolved_retry86400_indices.py"),
            "--root", str(root), "--panel", "A",
        ],
        check=True, text=True, capture_output=True,
    )
    assert completed.stdout == "1\n"
    assert "retry=1 resolved=1" in completed.stderr


def test_audit_48h_highs_retry_distinguishes_completed_and_oom(tmp_path):
    root = tmp_path / "panel_a"
    output = root / "mip_highs_native_retry172800"
    output.mkdir(parents=True)
    solver_commit = "44b6d5030a78ddca9c74f582d70ad87572e61794"
    status_hash = "a" * 64
    journal_hash = "b" * 64
    fields = [
        "index", "cell", "target_fleet", "representation_id",
        "source_status_sha256", "source_journal_sha256", "classification",
        "gurobi_buses", "gurobi_bound", "gurobi_fleet_proven",
        "highs24_present", "highs24_buses", "highs24_bound",
        "highs24_fleet_proven", "highs24_runtime_s",
        "highs24_source_hash_match", "highs24_physical_witness_valid",
        "highs24_configuration_match",
    ]
    with (root / "backend_retry86400.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in (1, 2):
            writer.writerow({
                "index": index,
                "cell": f"k05_s{index}",
                "target_fleet": 5,
                "representation_id": "uniform_2_1",
                "source_status_sha256": status_hash,
                "source_journal_sha256": journal_hash,
                "classification": (
                    "both_unproven" if index == 1
                    else "slurm_execution_error"
                ),
                "gurobi_buses": 6,
                "gurobi_bound": 6,
                "gurobi_fleet_proven": True,
                "highs24_present": index == 1,
                "highs24_buses": 7,
                "highs24_bound": 5,
                "highs24_fleet_proven": False,
                "highs24_runtime_s": 86400,
                "highs24_source_hash_match": index == 1,
                "highs24_physical_witness_valid": index == 1,
                "highs24_configuration_match": index == 1,
            })
    fields8 = [
        "index", "classification", "highs8_present", "highs8_buses",
        "highs8_bound", "highs8_fleet_proven", "highs8_runtime_s",
        "highs8_source_hash_match", "highs8_physical_witness_valid",
        "highs8_configuration_match",
    ]
    with (root / "backend_retry28800.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields8)
        writer.writeheader()
        for index in (1, 2):
            writer.writerow({
                "index": index,
                "classification": "both_unproven",
                "highs8_present": True,
                "highs8_buses": 7,
                "highs8_bound": 5,
                "highs8_fleet_proven": False,
                "highs8_runtime_s": 28800,
                "highs8_source_hash_match": True,
                "highs8_physical_witness_valid": True,
                "highs8_configuration_match": True,
            })
    (root / "highs_unresolved_retry172800_jobs.tsv").write_text(
        "panel\tarray_job_id\tindices\twrapper_commit\tsolver_commit\t"
        "agent_tip_observed\tbackend\ttimelimit_s\tthreads\tpartition\n"
        f"A\t700\t1,2\t{'c' * 40}\t{solver_commit}\t{'d' * 40}\t"
        "highspy_native\t172800\t8\tscaglione\n"
    )
    (output / "A__k05_s1__uniform_2_1.json").write_text(json.dumps({
        "backend": "highspy_native",
        "requested_timelimit_s": 172800,
        "threads_requested": 8,
        "code_identity": {
            "expected_commit": solver_commit,
            "observed_commit": solver_commit,
        },
        "source_result_sha256": status_hash,
        "source_journal_sha256": journal_hash,
        "status_name": "OPTIMAL",
        "buses": 6,
        "fleet_bound": 6,
        "fleet_proven": True,
        "physical_witness_valid": True,
        "runtime_s": 100000,
    }))
    sacct = root / "slurm.psv"
    sacct.write_text(
        "700_1|701|eua30_h48|COMPLETED|0:0|1-04:00:00|2-00:30:00|"
        "8-00:00:00|8||||scaglione-cpu-01\n"
        "700_2|702|eua30_h48|OUT_OF_MEMORY|0:125|1-04:00:00|"
        "2-00:30:00|8-00:00:00|8||||scaglione-cpu-01\n"
    )
    run_tool(
        "audit_highs_unresolved_retry172800.py",
        "--root", root, "--panel", "A", "--sacct", sacct,
    )
    rows = {
        row["index"]: row
        for row in csv.DictReader((root / "backend_retry172800.csv").open())
    }
    assert rows["1"]["classification"] == "proven_fleet_agreement"
    assert rows["1"]["retry_progress_from_24h"] == "became_proven"
    assert rows["1"]["prior_validated_stage"] == "24h"
    assert rows["2"]["classification"] == "slurm_execution_error"
    assert rows["2"]["prior_validated_stage"] == "8h_fallback"
    assert rows["2"]["highs48_slurm_state"] == "OUT_OF_MEMORY"
    selection = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "select_highs_oom_retry172800_indices.py"),
            "--root", str(root), "--panel", "A",
        ],
        check=True, text=True, capture_output=True,
    )
    assert selection.stdout == "2\n"
    assert "OOM selection: 1" in selection.stderr
