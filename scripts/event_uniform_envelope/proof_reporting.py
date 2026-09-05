#!/usr/bin/env python3
"""Fail-closed proof semantics for event-CG reporting.

The combined-cost CG endpoint is useful descriptive evidence, but it is not
the fleet-only model lower bound.  This module keeps that endpoint separate
from the two claims that require authenticated evidence:

``L_model``
    A certified fleet-only LP lower bound tied to the exact CG status and
    column journal represented by the row.
``I_model_proven``
    A physically valid integer witness for the same named representation
    whose bus count equals ``ceil(L_model)``.  The witness need not be a
    member of the finite CG pool used by the fleet certificate.

The functions are reporting-only.  They do not solve, repair, or mutate any
optimization artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any


ARTIFICIAL_TOLERANCE = 1e-7
CEILING_TOLERANCE = 1e-7
FLEET_CERTIFICATE_SCHEMA = "evsp-dr-fleet-lp-phase2-certificate-v1"
FLEET_CERTIFICATE_SCOPE = "fleet_lp_lower_bound_in_named_discrete_route_space"

REPORTING_FIELDS = (
    "cg_route_weight_endpoint",
    "lp_route_weight_matches_target",
    "L_model",
    "L_model_proven",
    "I_model",
    "I_model_proven",
    "fleet_certificate_valid",
    "integer_witness_valid",
    "proof_blocker",
)


# These are the representation identifiers used by the committed campaign
# manifests.  Metadata is checked when an artifact supplies it; no filename
# convention is inferred from a source path.
REPRESENTATIONS = {
    "event_2p5_event5": {
        "time_model": "event",
        "soc_step": 2.5,
        "block_min": 5,
    },
    "uniform_10_10": {
        "time_model": "uniform",
        "soc_step": 10.0,
        "block_min": 10,
    },
    "uniform_4_5": {
        "time_model": "uniform",
        "soc_step": 4.0,
        "block_min": 5,
    },
    "uniform_2_5": {
        "time_model": "uniform",
        "soc_step": 2.0,
        "block_min": 5,
    },
    "uniform_2_2": {
        "time_model": "uniform",
        "soc_step": 2.0,
        "block_min": 2,
    },
    "uniform_2_1": {
        "time_model": "uniform",
        "soc_step": 2.0,
        "block_min": 1,
    },
}


def finite_number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    unsigned = {
        key: value for key, value in payload.items()
        if key != "certificate_sha256"
    }
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _context_value(payload: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    return None


def _matches_optional_identity(
    payload: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    name: str,
    aliases: tuple[str, ...] = (),
) -> bool:
    value = _context_value(payload, name, *aliases)
    if value is None or value == "":
        return True
    return _text(value) == _text(row.get(name))


def _source_hashes_match(
    payload: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    require_journal: bool,
) -> bool:
    source = payload.get("source_cg") or payload.get("source") or payload
    if not isinstance(source, Mapping):
        return False
    result_hash = _context_value(
        source, "result_sha256", "source_result_sha256", "status_sha256"
    )
    journal_hash = _context_value(
        source, "journal_sha256", "source_journal_sha256"
    )
    expected_result = _text(
        row.get("result_sha256") or row.get("source_result_sha256")
    )
    expected_journal = _text(
        row.get("journal_sha256") or row.get("source_journal_sha256")
    )
    if not result_hash or not expected_result:
        return False
    if _text(result_hash) != expected_result:
        return False
    if not require_journal:
        return True
    return (
        bool(journal_hash)
        and bool(expected_journal)
        and _text(journal_hash) == expected_journal
    )


def _matches_instance_hash(
    payload: Mapping[str, Any], row: Mapping[str, Any]
) -> bool:
    instance_hash = _context_value(
        payload, "instance_sha256", "source_instance_sha256"
    )
    source = payload.get("source_cg") or payload.get("source")
    if instance_hash is None and isinstance(source, Mapping):
        instance_hash = _context_value(
            source, "instance_sha256", "source_instance_sha256"
        )
    if instance_hash is None or instance_hash == "":
        return True
    expected = _text(row.get("instance_sha256"))
    return bool(expected) and _text(instance_hash) == expected


def _number_equal(left: Any, right: Any) -> bool:
    left_number = finite_number(left)
    right_number = finite_number(right)
    if left_number is None or right_number is None:
        return _text(left) == _text(right)
    return math.isclose(left_number, right_number, rel_tol=0.0, abs_tol=1e-9)


def _matches_representation(
    payload: Mapping[str, Any], row: Mapping[str, Any]
) -> bool:
    expected_id = _text(
        row.get("representation_id") or row.get("representation")
    )
    declared_id = _context_value(payload, "representation_id")
    representation = payload.get("representation")
    if isinstance(representation, str):
        declared_id = declared_id or representation
        metadata: Mapping[str, Any] = {}
    elif isinstance(representation, Mapping):
        metadata = representation
    else:
        metadata = {}
    if declared_id and expected_id and _text(declared_id) != expected_id:
        return False

    expected_metadata = REPRESENTATIONS.get(expected_id, {})
    aliases = {
        "soc_step": ("soc_step", "soc_step_kwh"),
        "block_min": ("block_min",),
        "time_model": ("time_model",),
        "g_kwh": ("g_kwh", "battery_kwh"),
        "charge_kw": ("charge_kw", "charge_power_kw"),
        "min_soc_frac": ("min_soc_frac", "reserve_frac"),
    }
    for key, expected_value in expected_metadata.items():
        actual_value = _context_value(metadata, *aliases[key])
        if actual_value is not None and not _number_equal(
            actual_value, expected_value
        ):
            return False
    return True


def validate_fleet_certificate(
    certificate: Mapping[str, Any] | None,
    row: Mapping[str, Any],
) -> tuple[bool, str, float | None]:
    """Validate a fleet-only certificate against one audited CG row."""

    if not isinstance(certificate, Mapping):
        return False, "missing_fleet_certificate", None
    inner = certificate.get("certificate", certificate)
    if not isinstance(inner, Mapping):
        return False, "invalid_fleet_certificate_shape", None
    if inner.get("schema") != FLEET_CERTIFICATE_SCHEMA:
        return False, "wrong_fleet_certificate_schema", None
    if inner.get("certified") is not True:
        return False, "fleet_certificate_not_certified", None
    scope = _text(inner.get("certificate_scope"))
    if scope != FLEET_CERTIFICATE_SCOPE:
        return False, "wrong_fleet_certificate_scope", None
    stored_digest = _text(inner.get("certificate_sha256"))
    if not stored_digest or stored_digest != _canonical_digest(inner):
        return False, "fleet_certificate_hash_mismatch", None
    lower = finite_number(
        certificate.get("fleet_lp_lower_bound", inner.get("fleet_lp_lower_bound"))
    )
    if lower is None:
        return False, "missing_fleet_lower_bound", None
    source = certificate.get("source_cg") or certificate.get("source")
    if not isinstance(source, Mapping):
        return False, "missing_fleet_certificate_source", None
    if source.get("certified") is False:
        return False, "fleet_certificate_source_not_certified", None
    if not _source_hashes_match(certificate, row, require_journal=True):
        return False, "fleet_certificate_source_hash_mismatch", None
    if not _matches_instance_hash(certificate, row):
        return False, "fleet_certificate_instance_hash_mismatch", None
    if not _matches_representation(certificate, row):
        return False, "fleet_certificate_representation_mismatch", None
    for name in ("cell_id", "representation_id"):
        if not _matches_optional_identity(certificate, row, name=name):
            return False, f"fleet_certificate_{name}_mismatch", None
        if not _matches_optional_identity(inner, row, name=name):
            return False, f"fleet_certificate_{name}_mismatch", None
    return True, "", lower


def validate_integer_witness(
    witness: Mapping[str, Any] | None,
    row: Mapping[str, Any],
) -> tuple[bool, str, int | None]:
    """Validate a physically replayed integer witness for one CG row."""

    if not isinstance(witness, Mapping):
        return False, "missing_integer_witness", None
    if witness.get("physical_witness_valid") is not True:
        return False, "integer_witness_not_physically_valid", None
    if (
        witness.get("continuous_physical_upper_bound") is True
        or witness.get("known_continuous_physical_upper_bound") is True
    ):
        return False, "continuous_upper_bound_is_not_event_witness", None
    scope = _context_value(witness, "scope", "witness_scope")
    if scope not in (
        None,
        "",
        "event_representation",
        "same_representation",
        "named_discrete_event_model_only",
    ):
        return False, "wrong_integer_witness_scope", None
    if scope == "named_discrete_event_model_only":
        expected_representation = _text(
            row.get("representation_id") or row.get("representation")
        )
        row_time_model = _text(row.get("time_model"))
        if (
            expected_representation
            and expected_representation in REPRESENTATIONS
            and not expected_representation.startswith("event_")
        ) or (row_time_model and row_time_model != "event"):
            return False, "integer_witness_representation_mismatch", None
    if (
        scope == "named_discrete_event_model_only"
        or witness.get("schema")
        == "evsp-dr-event-known-partition-model-witness-v1"
    ) and witness.get("model_optimum_proven_by_sandwich") is not True:
        return False, "integer_witness_model_optimum_not_proven", None
    buses_value = finite_number(
        _context_value(witness, "integer_fleet_witness", "buses")
    )
    buses = int(buses_value) if buses_value is not None else None
    if buses is None or abs(buses_value - buses) > ARTIFICIAL_TOLERANCE:
        return False, "invalid_integer_witness_bus_count", None
    if not _source_hashes_match(witness, row, require_journal=False):
        return False, "integer_witness_source_hash_mismatch", None
    if not _matches_instance_hash(witness, row):
        return False, "integer_witness_instance_hash_mismatch", None
    if not _matches_representation(witness, row):
        return False, "integer_witness_representation_mismatch", None
    for name in ("cell_id", "representation_id"):
        if not _matches_optional_identity(witness, row, name=name):
            return False, f"integer_witness_{name}_mismatch", None
    return True, "", buses


def embedded_evidence(status: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    """Return optional report-side evidence embedded in a status payload.

    Current CG workers do not emit either object.  The explicit lookup keeps
    this layer compatible with future sidecar-aware auditors without changing
    the workers themselves.
    """

    certificate = None
    witness = None
    for key in ("fleet_lp_certificate", "fleet_certificate"):
        value = status.get(key)
        if isinstance(value, Mapping):
            certificate = value
            break
    for key in ("integer_witness", "event_integer_witness", "physical_witness"):
        value = status.get(key)
        if isinstance(value, Mapping):
            witness = value
            break
    return certificate, witness


def assess_row(
    row: Mapping[str, Any],
    *,
    fleet_certificate: Mapping[str, Any] | None = None,
    integer_witness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute reporting fields without promoting descriptive endpoints."""

    endpoint = finite_number(
        row.get("cg_route_weight_endpoint", row.get("route_weight"))
    )
    scale = finite_number(row.get("scale"))
    matches_target = bool(
        endpoint is not None
        and scale is not None
        and math.ceil(endpoint - CEILING_TOLERANCE) == int(scale)
    )
    artificials = finite_number(row.get("artificials"))
    if artificials is None:
        artifact_blocker = "missing_artificials"
    elif artificials > ARTIFICIAL_TOLERANCE:
        artifact_blocker = "residual_artificials"
    else:
        artifact_blocker = ""

    cert_ok, cert_reason, lower = validate_fleet_certificate(
        fleet_certificate, row
    )
    l_model_proven = cert_ok and not artifact_blocker
    witness_ok, witness_reason, buses = validate_integer_witness(
        integer_witness, row
    )
    if not l_model_proven:
        i_model_proven = False
    elif not witness_ok:
        i_model_proven = False
    else:
        i_model_proven = (
            buses == math.ceil(lower - CEILING_TOLERANCE)
        )

    if artifact_blocker:
        blocker = artifact_blocker
    elif not cert_ok:
        blocker = cert_reason
    elif not witness_ok:
        blocker = witness_reason
    elif not i_model_proven:
        blocker = "integer_witness_does_not_close_fleet_lower_bound"
    else:
        blocker = ""

    return {
        "cg_route_weight_endpoint": endpoint if endpoint is not None else "",
        "lp_route_weight_matches_target": matches_target,
        "L_model": lower if l_model_proven else "",
        "L_model_proven": l_model_proven,
        "I_model": buses if i_model_proven else "",
        "I_model_proven": i_model_proven,
        "fleet_certificate_valid": cert_ok,
        "integer_witness_valid": witness_ok,
        "proof_blocker": blocker,
    }


def reporting_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    """Assess a row using only any evidence embedded in its status payload."""

    status = row.get("status")
    certificate = None
    witness = None
    if isinstance(status, Mapping):
        certificate, witness = embedded_evidence(status)
    return assess_row(
        row, fleet_certificate=certificate, integer_witness=witness
    )
