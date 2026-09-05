import copy
import hashlib
import json
import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "scripts" / "event_uniform_envelope"
sys.path.insert(0, str(TOOLS))

from proof_reporting import (  # noqa: E402
    FLEET_CERTIFICATE_SCHEMA,
    FLEET_CERTIFICATE_SCOPE,
    assess_row,
)
import summarize_cg_frontier as frontier  # noqa: E402


def base_row(**overrides):
    row = {
        "cell_id": "k13_s1",
        "representation_id": "event_2p5_event5",
        "scale": "13",
        "result_sha256": "r" * 64,
        "journal_sha256": "j" * 64,
        "instance_sha256": "i" * 64,
        "time_model": "event",
        "artificials": "0",
        "route_weight": "13",
    }
    row.update(overrides)
    return row


def fleet_certificate(row, *, lower=13.0, **overrides):
    inner = {
        "schema": FLEET_CERTIFICATE_SCHEMA,
        "certified": True,
        "certificate_scope": FLEET_CERTIFICATE_SCOPE,
        "objective_definition": "fleet_only",
        "fleet_lp_primal": lower,
        "fleet_lp_lower_bound": lower,
        "primal_dual_gap": 0.0,
        "minimum_reduced_cost": 0.0,
        "pricing_tolerance": 1e-9,
        "iterations": 2,
        "stop_reason": "certified",
        "max_row_violation": 0.0,
        "max_bound_violation": 0.0,
    }
    if "certified" in overrides:
        inner["certified"] = overrides.pop("certified")
    inner["certificate_sha256"] = hashlib.sha256(
        json.dumps(
            inner, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    payload = {
        "certificate": inner,
        "fleet_lp_lower_bound": lower,
        "source_cg": {
            "result": "/campaign/cg/M__k13_s1__event_2p5_event5.json",
            "result_sha256": row["result_sha256"],
            "journal_sha256": row["journal_sha256"],
        },
    }
    payload.update(overrides)
    return payload


def witness(row, **overrides):
    payload = {
        "physical_witness_valid": True,
        "witness_scope": "event_representation",
        "model_optimum_proven_by_sandwich": True,
        "buses": 13,
        "source_result_sha256": row["result_sha256"],
        "source_journal_sha256": row["journal_sha256"],
        "cell_id": row["cell_id"],
        "representation_id": row["representation_id"],
    }
    payload.update(overrides)
    return payload


def test_combined_cost_endpoint_target_is_not_integer_proof_without_witness():
    result = assess_row(base_row())
    assert result["cg_route_weight_endpoint"] == 13.0
    assert result["lp_route_weight_matches_target"] is True
    assert result["L_model"] == ""
    assert result["I_model_proven"] is False


def test_uncertified_fleet_endpoint_is_not_l_model():
    row = base_row()
    certificate = fleet_certificate(row, certified=False)
    result = assess_row(row, fleet_certificate=certificate)
    assert result["fleet_certificate_valid"] is False
    assert result["L_model"] == ""
    assert result["I_model_proven"] is False


def test_residual_artificials_block_both_proof_fields():
    row = base_row(artificials="0.5")
    result = assess_row(
        row,
        fleet_certificate=fleet_certificate(row),
        integer_witness=witness(row),
    )
    assert result["fleet_certificate_valid"] is True
    assert result["integer_witness_valid"] is True
    assert result["L_model"] == ""
    assert result["I_model_proven"] is False
    assert result["proof_blocker"] == "residual_artificials"


def test_authenticated_fleet_certificate_and_event_witness_prove_integer_model():
    row = base_row()
    result = assess_row(
        row,
        fleet_certificate=fleet_certificate(row),
        integer_witness=witness(row),
    )
    assert result["L_model"] == 13.0
    assert result["L_model_proven"] is True
    assert result["I_model"] == 13
    assert result["I_model_proven"] is True


def test_identity_mismatch_blocks_proof():
    row = base_row()
    certificate = fleet_certificate(row, cell_id="k13_s2")
    result = assess_row(
        row,
        fleet_certificate=certificate,
        integer_witness=witness(row, representation_id="uniform_2_1"),
    )
    assert result["L_model"] == ""
    assert result["I_model_proven"] is False
    assert result["proof_blocker"] == "fleet_certificate_cell_id_mismatch"

    result = assess_row(
        row,
        fleet_certificate=fleet_certificate(
            row, representation_id="uniform_2_1"
        ),
        integer_witness=witness(row),
    )
    assert result["L_model"] == ""
    assert result["I_model_proven"] is False

    result = assess_row(
        row,
        fleet_certificate=fleet_certificate(
            row, source_cg={
                "result": "/campaign/renamed-status-payload.dat",
                "result_sha256": "x" * 64,
                "journal_sha256": row["journal_sha256"],
            }
        ),
        integer_witness=witness(row),
    )
    assert result["L_model"] == ""
    assert result["I_model_proven"] is False


FIXTURE_ROOT = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "event_uniform_proof"
    / "k02_s1"
)


def real_fixture_payloads():
    def load(name):
        return json.loads((FIXTURE_ROOT / name).read_text())

    cg = load("cg.json")
    certificate = load("fleet_phase2.json")
    witness_payload = load("witness.json")
    row = {
        "cell_id": "k02_s1",
        "representation_id": "event_2p5_event5",
        "scale": "2",
        "result_sha256": certificate["source_cg"]["result_sha256"],
        "journal_sha256": certificate["source_cg"]["journal_sha256"],
        "instance_sha256": witness_payload["instance_sha256"],
        "time_model": cg["time_model"],
        "soc_step": cg["soc_step"],
        "block_min": cg["block_min"],
        "g_kwh": cg["g_kwh"],
        "charge_kw": cg["charge_kw"],
        "min_soc_frac": cg["min_soc_frac"],
        "artificials": cg["final"]["artificials"],
        "route_weight": cg["final"]["route_weight"],
    }
    return row, certificate, witness_payload


def refresh_certificate_digest(certificate):
    inner = certificate["certificate"]
    unsigned = {
        key: value for key, value in inner.items()
        if key != "certificate_sha256"
    }
    inner["certificate_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def test_minimized_committed_event_fixtures_prove_named_representation():
    row, certificate, witness_payload = real_fixture_payloads()
    result = assess_row(
        row,
        fleet_certificate=certificate,
        integer_witness=witness_payload,
    )
    assert result["L_model"] == certificate["fleet_lp_lower_bound"]
    assert result["L_model_proven"] is True
    assert result["I_model"] == 2
    assert result["I_model_proven"] is True


def test_real_fixture_status_path_is_not_an_identity_requirement():
    row, certificate, witness_payload = real_fixture_payloads()
    certificate = copy.deepcopy(certificate)
    certificate["source_cg"]["result"] = "/renamed/status.payload"
    witness_payload = copy.deepcopy(witness_payload)
    witness_payload["status"] = "/another/name"
    result = assess_row(
        row,
        fleet_certificate=certificate,
        integer_witness=witness_payload,
    )
    assert result["I_model_proven"] is True


def test_real_fixture_altered_hashes_block_proof():
    row, certificate, witness_payload = real_fixture_payloads()
    altered_certificate = copy.deepcopy(certificate)
    altered_certificate["source_cg"]["result_sha256"] = "a" * 64
    result = assess_row(row, fleet_certificate=altered_certificate)
    assert result["L_model_proven"] is False

    altered_witness = copy.deepcopy(witness_payload)
    altered_witness["status_sha256"] = "b" * 64
    result = assess_row(
        row,
        fleet_certificate=certificate,
        integer_witness=altered_witness,
    )
    assert result["L_model_proven"] is True
    assert result["I_model_proven"] is False


def test_real_fixture_representation_and_cell_mismatches_block_proof():
    row, certificate, witness_payload = real_fixture_payloads()
    altered_certificate = copy.deepcopy(certificate)
    altered_certificate["representation"]["time_model"] = "uniform"
    result = assess_row(row, fleet_certificate=altered_certificate)
    assert result["L_model_proven"] is False

    altered_witness = copy.deepcopy(witness_payload)
    altered_witness["cell_id"] = "k02_s2"
    result = assess_row(
        row,
        fleet_certificate=certificate,
        integer_witness=altered_witness,
    )
    assert result["I_model_proven"] is False

    altered_row = dict(row)
    altered_row["instance_sha256"] = "c" * 64
    result = assess_row(
        altered_row,
        fleet_certificate=certificate,
        integer_witness=witness_payload,
    )
    assert result["L_model_proven"] is True
    assert result["I_model_proven"] is False


def assert_real_certificate_physics_mismatch(field, value):
    row, certificate, _ = real_fixture_payloads()
    altered_certificate = copy.deepcopy(certificate)
    altered_certificate["representation"][field] = value
    result = assess_row(row, fleet_certificate=altered_certificate)
    assert result["fleet_certificate_valid"] is False
    assert result["L_model_proven"] is False


def test_real_fixture_battery_capacity_mismatch_blocks_fleet_proof():
    assert_real_certificate_physics_mismatch("g_kwh", 360.0)


def test_real_fixture_charging_power_mismatch_blocks_fleet_proof():
    assert_real_certificate_physics_mismatch("charge_kw", 360.0)


def test_real_fixture_minimum_soc_mismatch_blocks_fleet_proof():
    assert_real_certificate_physics_mismatch("min_soc_frac", 0.2)


def test_real_fixture_time_model_mismatch_blocks_fleet_proof():
    assert_real_certificate_physics_mismatch("time_model", "uniform")


def test_real_fixture_soc_step_mismatch_blocks_fleet_proof():
    assert_real_certificate_physics_mismatch("soc_step", 5.0)


def test_real_fixture_charging_block_duration_mismatch_blocks_fleet_proof():
    assert_real_certificate_physics_mismatch("block_min", 10)


def test_real_fixture_wrong_outer_certificate_schema_blocks_fleet_proof():
    row, certificate, _ = real_fixture_payloads()
    altered_certificate = copy.deepcopy(certificate)
    altered_certificate["schema"] = "wrong-fleet-document-schema"
    result = assess_row(row, fleet_certificate=altered_certificate)
    assert result["fleet_certificate_valid"] is False
    assert result["proof_blocker"] == "wrong_fleet_certificate_document_schema"


def test_real_fixture_wrong_witness_schema_blocks_integer_proof():
    row, certificate, witness_payload = real_fixture_payloads()
    altered_witness = copy.deepcopy(witness_payload)
    altered_witness["schema"] = "wrong-witness-schema"
    result = assess_row(
        row,
        fleet_certificate=certificate,
        integer_witness=altered_witness,
    )
    assert result["L_model_proven"] is True
    assert result["integer_witness_valid"] is False
    assert result["I_model_proven"] is False
    assert result["proof_blocker"] == "wrong_integer_witness_schema"


def test_real_fixture_physical_validity_and_bus_count_mismatches_block_proof():
    row, certificate, witness_payload = real_fixture_payloads()
    altered_witness = copy.deepcopy(witness_payload)
    altered_witness["physical_witness_valid"] = False
    result = assess_row(
        row,
        fleet_certificate=certificate,
        integer_witness=altered_witness,
    )
    assert result["I_model_proven"] is False

    altered_witness = copy.deepcopy(witness_payload)
    altered_witness["integer_fleet_witness"] = 3
    result = assess_row(
        row,
        fleet_certificate=certificate,
        integer_witness=altered_witness,
    )
    assert result["I_model_proven"] is False


def test_real_fixture_certificate_scope_mismatch_blocks_fleet_proof():
    row, certificate, _ = real_fixture_payloads()
    altered_certificate = copy.deepcopy(certificate)
    altered_certificate["certificate"]["certificate_scope"] = "wrong"
    refresh_certificate_digest(altered_certificate)
    result = assess_row(row, fleet_certificate=altered_certificate)
    assert result["fleet_certificate_valid"] is False
    assert result["proof_blocker"] == "wrong_fleet_certificate_scope"


def test_real_fixture_residual_artificials_block_formal_fields():
    row, certificate, witness_payload = real_fixture_payloads()
    row["artificials"] = 1.0
    result = assess_row(
        row,
        fleet_certificate=certificate,
        integer_witness=witness_payload,
    )
    assert result["fleet_certificate_valid"] is True
    assert result["integer_witness_valid"] is True
    assert result["L_model_proven"] is False
    assert result["I_model_proven"] is False


def test_legacy_summary_preserves_certification_without_integer_proof():
    row = frontier.enrich({
        "cell_id": "k13_s1",
        "scale": "13",
        "slurm_state": "COMPLETED",
        "configuration_match": "True",
        "certified_rc_optimal": "True",
        "stop_reason": "certified",
        "L_model": "13",
        "artificials": "0",
    })
    assert row["outcome"] == "certified"
    assert row["certified_rc_optimal"] == "True"
    assert row["cg_route_weight_endpoint"] == 13.0
    assert row["legacy_l_model_endpoint_fallback"] is True
    assert row["fleet_target_proved"] is False


def test_frontier_csv_order_and_schema_are_deterministic(tmp_path):
    rows = [
        frontier.enrich({
            "cell_id": "k13_s2", "scale": "13", "index": "2",
            "L_model": "12", "configuration_match": "True",
            "slurm_state": "COMPLETED",
        }),
        frontier.enrich({
            "cell_id": "k13_s1", "scale": "13", "index": "1",
            "L_model": "13", "configuration_match": "True",
            "slurm_state": "COMPLETED",
        }),
    ]
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    frontier.write_csv(first, rows)
    frontier.write_csv(second, list(reversed(rows)))
    assert first.read_bytes() == second.read_bytes()
    header = first.read_text().splitlines()[0].split(",")
    assert header.index("cg_route_weight_endpoint") < header.index("L_model")
    assert header.index("I_model_proven") < header.index("fleet_target_proved")
