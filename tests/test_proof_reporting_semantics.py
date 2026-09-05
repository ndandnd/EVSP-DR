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
            row, source_cg={
                "result": "/campaign/cg/M__k13_s1__uniform_2_1.json",
                "result_sha256": row["result_sha256"],
                "journal_sha256": row["journal_sha256"],
            }
        ),
        integer_witness=witness(row),
    )
    assert result["L_model"] == ""
    assert result["I_model_proven"] is False

    result = assess_row(
        row,
        fleet_certificate=fleet_certificate(
            row, source_cg={
                "result": "/campaign/cg/M__k13_s1__event_2p5_event5.json",
                "result_sha256": "x" * 64,
                "journal_sha256": row["journal_sha256"],
            }
        ),
        integer_witness=witness(row),
    )
    assert result["L_model"] == ""
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
