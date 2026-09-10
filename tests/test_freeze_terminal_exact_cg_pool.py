import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZER = ROOT / "src" / "freeze_terminal_exact_cg_pool.py"
COMMIT = "9bdbb17" + "0" * 33


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_source(tmp_path, name, *, stop="certified", artificials=0.0,
                final_lp_fallback=False, master_sense="partition",
                treatment="RAW", inherited_metadata=False):
    status = tmp_path / f"{name}.json"
    journal = Path(str(status) + ".columns.jsonl")
    iterations = Path(str(status) + ".iters.csv")
    journal.write_text(
        json.dumps({"found_iter": 0, "trips": [0], "cost": 5.0}) + "\n"
        + json.dumps({"found_iter": 1, "trips": [1], "cost": 6.0}) + "\n"
    )
    iterations.write_text("iteration,columns\n0,2\n")
    payload = {
        "time_model": "event", "network_metrics": {"arc_mode": "lazy"},
        "soc_step": 2.5, "block_min": 5, "g_kwh": 240.0,
        "charge_kw": 240.0, "min_soc_frac": 0.0,
        "prices_csv": "hourly_prices_flat.csv", "master_sense": master_sense,
        "initial_pool": "singletons", "columns_per_iter": 30,
        "column_pool_treatment": treatment, "column_selection": "reduced_cost",
        "column_diversity_weight": 0.0, "column_candidate_multiplier": 4,
        "stop_reason": stop, "certified_rc_optimal": stop == "certified",
        "columns": 2,
        "final": None if final_lp_fallback else {"artificials": artificials},
        "final_lp": ({"artificial_total": artificials}
                     if final_lp_fallback else None),
        "columns_journal": str(journal),
        "provenance": {"git_commit": COMMIT, "instance_sha256": "a" * 64,
                       "rc_eps": 0.0001},
    }
    if inherited_metadata:
        inherited_hash = "b" * 64
        payload["inherited_event_pool_status_sha256"] = inherited_hash
        payload["inherited_event_pool_audit"] = {
            "schema": "evsp-dr-inherited-event-pool-audit-v1",
            "child_event_graph_reoptimization": True,
            "inherited_basis": False,
            "inherited_duals": False,
            "inherited_lp_certificate": False,
            "source_status_sha256": inherited_hash,
            "source_journal_sha256": "c" * 64,
            "attempted_unique_columns": 2,
            "source_unique_columns": 2,
            "accepted_columns": 2,
            "rejected_columns": 0,
        }
        payload["provenance"]["inherited_event_pool_status_sha256"] = inherited_hash
    status.write_text(json.dumps(payload))
    return status


def invoke(tmp_path, base, resume, *, master_sense="partition"):
    out = tmp_path / "frozen.json"
    record = tmp_path / "record.json"
    result = subprocess.run([
        sys.executable, str(FREEZER), "--base-status", str(base),
        "--resume-status", str(resume), "--cell", "k09_p1",
        "--instance-relative-to-data", "scale_ladder/instances/x.csv",
        "--instance-sha256", "a" * 64, "--source-solver-commit", COMMIT,
        "--master-sense", master_sense,
        "--out", str(out), "--record", str(record),
    ], text=True, capture_output=True)
    return result, out, record


def test_prefers_valid_terminal_continuation_and_binds_hashes(tmp_path):
    base = make_source(tmp_path, "base")
    resume = make_source(tmp_path, "resume", stop="wall_limit")
    result, out, record = invoke(tmp_path, base, resume)
    assert result.returncode == 0, result.stderr
    frozen = json.loads(out.read_text())
    rec = json.loads(record.read_text())
    assert frozen["terminal_pool_snapshot"]["selected_source_stage"] == "continuation"
    assert frozen["csv"] == "scale_ladder/instances/x.csv"
    assert rec["source_certified"] is False and rec["columns"] == 2
    assert rec["snapshot_sha256"] == digest(out)
    assert rec["journal_sha256"] == digest(Path(str(out) + ".columns.jsonl"))


def test_rejects_nonfinite_artificials_without_publishing(tmp_path):
    base = make_source(tmp_path, "base", artificials=float("nan"))
    missing = tmp_path / "missing.json"
    result, out, record = invoke(tmp_path, base, missing)
    assert result.returncode != 0
    assert "no usable terminal source" in result.stderr
    assert not out.exists() and not record.exists()
    assert not Path(str(out) + ".columns.jsonl").exists()


def test_accepts_terminal_last_good_lp_when_final_iteration_is_absent(tmp_path):
    base = make_source(
        tmp_path, "base", stop="master_failed", final_lp_fallback=True
    )
    result, _out, record = invoke(tmp_path, base, base)
    assert result.returncode == 0, result.stderr
    rec = json.loads(record.read_text())
    assert rec["source_stop_reason"] == "master_failed"
    assert rec["source_certified"] is False
    assert rec["artificials"] == 0.0
    assert rec["artificials_source"] == "final_lp"


def test_freezes_cover_source_only_when_cover_is_explicit(tmp_path):
    cover = make_source(tmp_path, "cover", master_sense="cover")
    result, out, record = invoke(
        tmp_path, cover, cover, master_sense="cover"
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text())["master_sense"] == "cover"
    assert json.loads(record.read_text())["master_sense"] == "cover"


def test_rejects_cover_source_under_partition_default(tmp_path):
    cover = make_source(tmp_path, "cover", master_sense="cover")
    result, out, record = invoke(tmp_path, cover, cover)
    assert result.returncode != 0
    assert "configuration mismatch" in result.stderr
    assert not out.exists() and not record.exists()


def test_freezes_valid_inherited_event_pool_without_relabeling(tmp_path):
    source = make_source(
        tmp_path, "warm", master_sense="cover",
        treatment="WARM-INHERITED-EVENT", inherited_metadata=True,
    )
    result, out, record = invoke(
        tmp_path, source, source, master_sense="cover"
    )
    assert result.returncode == 0, result.stderr
    frozen = json.loads(out.read_text())
    rec = json.loads(record.read_text())
    assert frozen["column_pool_treatment"] == "WARM-INHERITED-EVENT"
    assert frozen["terminal_pool_snapshot"]["column_pool_treatment"] == (
        "WARM-INHERITED-EVENT"
    )
    assert rec["column_pool_treatment"] == "WARM-INHERITED-EVENT"


def test_rejects_inherited_pool_without_child_replay_audit(tmp_path):
    source = make_source(
        tmp_path, "warm_missing_audit", master_sense="cover",
        treatment="WARM-INHERITED-EVENT",
    )
    result, out, record = invoke(
        tmp_path, source, source, master_sense="cover"
    )
    assert result.returncode != 0
    assert "inherited event-pool audit" in result.stderr
    assert not out.exists() and not record.exists()
