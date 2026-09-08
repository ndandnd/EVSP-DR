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
                final_lp_fallback=False, g_kwh=240.0, charge_kw=240.0,
                treatment="RAW", seed_sha=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    status = tmp_path / f"{name}.json"
    journal = Path(str(status) + ".columns.jsonl")
    iterations = Path(str(status) + ".iters.csv")
    journal.write_text(
        json.dumps({"found_iter": 0, "trips": [0], "cost": 5.0}) + "\n"
        + json.dumps({"found_iter": 1, "trips": [1], "cost": 6.0}) + "\n"
    )
    iterations.write_text("iteration,columns\n0,2\n")
    status.write_text(json.dumps({
        "time_model": "event", "network_metrics": {"arc_mode": "lazy"},
        "soc_step": 2.5, "block_min": 5, "g_kwh": g_kwh,
        "charge_kw": charge_kw, "min_soc_frac": 0.0,
        "prices_csv": "hourly_prices_flat.csv", "master_sense": "partition",
        "initial_pool": "singletons", "columns_per_iter": 30,
        "column_pool_treatment": treatment, "column_selection": "reduced_cost",
        "column_diversity_weight": 0.0, "column_candidate_multiplier": 4,
        "stop_reason": stop, "certified_rc_optimal": stop == "certified",
        "columns": 2,
        "final": None if final_lp_fallback else {"artificials": artificials},
        "final_lp": ({"artificial_total": artificials}
                     if final_lp_fallback else None),
        "columns_journal": str(journal),
        "validated_seed_routes_sha256": seed_sha,
        "validated_seed_source_type": "GREEDY" if seed_sha else None,
        "validated_seed_route_count": 3 if seed_sha else None,
        "provenance": {"git_commit": COMMIT, "instance_sha256": "a" * 64,
                       "rc_eps": 0.0001},
    }))
    return status


def invoke(tmp_path, base, resume, *, expected_g=240.0,
           expected_charge=240.0, treatment="RAW", seed_sha=None):
    out = tmp_path / "frozen.json"
    record = tmp_path / "record.json"
    result = subprocess.run([
        sys.executable, str(FREEZER), "--base-status", str(base),
        "--resume-status", str(resume), "--cell", "k09_p1",
        "--instance-relative-to-data", "scale_ladder/instances/x.csv",
        "--instance-sha256", "a" * 64, "--source-solver-commit", COMMIT,
        "--expected-g-kwh", str(expected_g),
        "--expected-charge-kw", str(expected_charge),
        "--expected-column-pool-treatment", treatment,
        *(["--expected-seed-sha256", seed_sha] if seed_sha else []),
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


def test_expected_physics_accepts_240_and_300_and_rejects_mismatch(tmp_path):
    source240 = make_source(tmp_path / "p240", "source", g_kwh=240.0,
                            charge_kw=240.0)
    result240, _out240, _record240 = invoke(
        tmp_path / "p240", source240, source240,
        expected_g=240.0, expected_charge=240.0,
    )
    assert result240.returncode == 0, result240.stderr

    source300 = make_source(tmp_path / "p300", "source", g_kwh=300.0,
                            charge_kw=300.0)
    result300, _out300, _record300 = invoke(
        tmp_path / "p300", source300, source300,
        expected_g=300.0, expected_charge=300.0,
    )
    assert result300.returncode == 0, result300.stderr

    source_wrong = make_source(
        tmp_path / "wrong", "source", g_kwh=240.0, charge_kw=240.0
    )
    result_wrong, out_wrong, record_wrong = invoke(
        tmp_path / "wrong", source_wrong, source_wrong,
        expected_g=300.0, expected_charge=300.0,
    )
    assert result_wrong.returncode != 0
    assert "configuration mismatch" in result_wrong.stderr
    assert not out_wrong.exists() and not record_wrong.exists()


def test_greedy_freeze_requires_matching_seed_identity(tmp_path):
    seed_sha = "d" * 64
    source = make_source(
        tmp_path / "greedy", "source", treatment="GREEDY",
        seed_sha=seed_sha,
    )
    accepted, _out, _record = invoke(
        tmp_path / "greedy", source, source,
        treatment="GREEDY", seed_sha=seed_sha,
    )
    assert accepted.returncode == 0, accepted.stderr

    wrong = make_source(
        tmp_path / "wrong_greedy", "source", treatment="GREEDY",
        seed_sha=seed_sha,
    )
    rejected, out, record = invoke(
        tmp_path / "wrong_greedy", wrong, wrong,
        treatment="GREEDY", seed_sha="e" * 64,
    )
    assert rejected.returncode != 0
    assert "GREEDY seed provenance mismatch" in rejected.stderr
    assert not out.exists() and not record.exists()
