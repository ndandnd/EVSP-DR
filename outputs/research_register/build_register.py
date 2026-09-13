#!/usr/bin/env python3
"""Build a normalized, auditable research-experiment register.

The input is a compact JSON snapshot emitted by ``collect_remote.py``.  The
snapshot is copied byte-for-byte into ``raw/``; all normalized hashes derived
inside this script are labeled as snapshot-payload hashes, never source-file
hashes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import datetime as dt
import hashlib
import io
import json
import math
import re
import shutil
from pathlib import Path


SCHEMA = "evsp-dr-research-register-v1"
UNKNOWN = "unknown"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")

COLUMNS = [
    "row_id", "campaign_id", "campaign_root", "case_id", "result_family",
    "stage", "substage", "arm", "artifact_status", "authority_role",
    "source_path", "source_sha256", "snapshot_payload_sha256",
    "snapshot_time_utc", "file_mtime_utc", "completion_marker_matches",
    "job_ids", "workflow_state", "dependency",
    "input_path", "input_sha256", "trip_count", "target_k", "chain",
    "replication", "tariff_path", "tariff_sha256", "prices_sha256",
    "reference_sha256", "deadhead_sha256", "master_sha256",
    "code_commit", "code_branch", "code_dirty", "solver_backend",
    "solver_version", "battery_kwh", "initial_soc_kwh", "reserve_kwh",
    "charge_kw", "parx_kw", "non_parx_kw", "soc_step_kwh",
    "block_minutes", "capacity_enforced", "charger_counts_json",
    "terminal_energy_policy", "charge_start_cost", "master_sense", "initialization",
    "column_pool_treatment", "pool_size", "cg_iterations", "stop_reason",
    "runtime_s", "phase_runtime_json", "weighted_lp_objective",
    "recorded_lp_objective", "lp_objective_kind",
    "fractional_fleet", "artificial_total", "min_reduced_cost",
    "full_model_lp_certified", "lp_bound_scope", "fleet_lower_bound",
    "fleet_lower_bound_scope", "mip_incumbent_fleet", "mip_bound_fleet",
    "mip_status", "mip_gap", "fleet_proven", "optimal_scope",
    "stage1_incumbent_fleet", "stage1_bound", "stage1_gap",
    "stage1_proven", "stage2_executed", "stage2_status",
    "stage2_charging_cost", "stage2_charging_bound", "stage2_gap",
    "charging_cost_grid", "charging_cost_continuous",
    "charging_cost_exact", "charging_cost_lower", "charging_cost_upper",
    "electricity_cost_grid", "electricity_cost_continuous",
    "electricity_cost_lower", "electricity_cost_upper",
    "charging_start_fees_grid", "charging_start_fees_continuous",
    "charging_starts_grid", "charging_starts_continuous",
    "charged_energy_grid_kwh", "charged_energy_continuous_kwh",
    "charging_metrics_reconcile",
    "comparator_eligible", "beats_original_robustly",
    "terminal_energy_grid_kwh", "terminal_energy_continuous_kwh",
    "physical_pool_validated", "physical_selected_validated",
    "physical_validation_scope", "overcovered_trips",
    "cross_route_capacity_validated", "proof_scope", "limitations", "notes",
]


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_sha(value) -> str:
    return sha_bytes(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode())


def at(value, *paths, default=None):
    for path in paths:
        current = value
        try:
            for key in path.split("."):
                if current is None:
                    raise KeyError(key)
                current = current[int(key)] if isinstance(current, list) else current[key]
            if current is not None:
                return current
        except (KeyError, IndexError, TypeError, ValueError):
            pass
    return default


def finite(value):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value if math.isfinite(float(value)) else None
    return value


def path_case(path, fallback):
    if not path:
        return fallback
    candidate = Path(path)
    stem = candidate.name
    for suffix in (
        ".rejected_physical_replay.json", ".phase-telemetry.jsonl",
        ".columns.jsonl", ".jsonl", ".json", ".tsv", ".log",
    ):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    if stem in {"cg", "mip", "mip_twostage", "comparison", "frontier", "result"}:
        stem = candidate.parent.name
    return stem or fallback


def case_dimensions(case_id, input_path=None):
    text = " ".join(str(value or "") for value in (case_id, input_path))
    target = re.search(r"(?:^|[_-])k(\d{1,2})(?:[_-]|$)", text, re.I)
    chain = re.search(r"(?:^|[_-])p(\d{1,2})(?:[_-]|$)", text, re.I)
    replication = re.search(r"(?:rep|fresh)(\d+)", text, re.I)
    return (
        int(target.group(1)) if target else None,
        int(chain.group(1)) if chain else None,
        int(replication.group(1)) if replication else None,
    )


def job_ids(value):
    found = set()
    def walk(item, key=""):
        if isinstance(item, dict):
            for child_key, child in item.items():
                walk(child, str(child_key).lower())
        elif isinstance(item, list):
            for child in item:
                walk(child, key)
        elif "job" in key:
            for number in re.findall(r"\b\d{5,}\b", str(item)):
                found.add(number)
    walk(value)
    return sorted(found, key=int)


def report_links(campaign_id, root, markdown_files):
    needles = {campaign_id, Path(root).name if root else ""} - {""}
    links = []
    for path, text in markdown_files:
        if any(needle in text for needle in needles):
            links.append(str(path))
    def priority(path):
        name = Path(path).name.lower()
        return (0 if name == "readme.md" else 1 if "report" in name else 2,
                len(path), path)
    return sorted(set(links), key=priority)[:2]


def giro_physical_validation(summary):
    """Return selected-route validation, separate from artifact hash status."""
    if not isinstance(summary, dict):
        return None
    metrics = summary.get("charging_comparison_metrics")
    validation = metrics.get("physical_validation") if isinstance(metrics, dict) else None
    if not isinstance(validation, dict):
        return None
    required = (
        "selected_route_replays_valid", "coverage_complete",
        "grid_terminal_total_matches_summary",
        "continuous_terminal_total_matches_summary",
        "grid_terminal_meets_target", "continuous_terminal_meets_target",
    )
    if any(key not in validation for key in required):
        return None
    return all(validation[key] is True for key in required)


class Register:
    def __init__(self, snapshot, snapshot_path, output_dir, markdown_files):
        self.snapshot = snapshot
        self.snapshot_path = snapshot_path
        self.output_dir = output_dir
        self.snapshot_time = snapshot.get("timestamp_utc")
        self.markdown_files = markdown_files
        self.rows = []
        self.campaigns = {}
        # The overnight manifest is authoritative for case identity and
        # target duties.  Keep this map campaign-scoped: historical and other
        # campaign records must continue using their source payload/path
        # semantics.
        self.authoritative_cases = {}
        overnight = snapshot.get("campaigns", {}).get(
            "overnight_extension_20260912", {}
        )
        manifest = (overnight.get("workflow", {}) or {}).get(
            "manifest.json", {}
        ) or {}
        for manifest_case, metadata in (manifest.get("cases", {}) or {}).items():
            if isinstance(metadata, dict):
                self.authoritative_cases[manifest_case] = {
                    **metadata, "id": manifest_case,
                }

    def authoritative_case(self, campaign_id, source_path, input_path,
                           fallback_case_id):
        """Resolve only overnight cases from the embedded manifest.

        Result paths can contain a scheduler run directory below the case
        directory (for example ``.../d00_g0/mip/949625_r0/result.json``),
        while warm input paths contain an older ``k2_15`` ancestor.  Prefer a
        manifest case directory or CSV basename before any generic regex.
        """
        if campaign_id != "overnight_extension_20260912":
            return None
        candidates = []
        for value in (source_path, input_path, fallback_case_id):
            if not value:
                continue
            text = str(value)
            candidates.extend(Path(text).parts)
            candidates.append(Path(text).stem)
        for candidate in candidates:
            if candidate in self.authoritative_cases:
                return self.authoritative_cases[candidate]
        return None

    def campaign(self, campaign_id, root, family, reports=None):
        record = self.campaigns.setdefault(campaign_id, {
            "campaign_id": campaign_id,
            "root": root,
            "family": family,
            "report_links": reports if reports is not None else report_links(
                campaign_id, root, self.markdown_files,
            ),
            "workflow_job_ids": [],
        })
        return record

    def add(self, campaign_id, root, family, stage, payload, *,
            source_path=None, source_sha256=None, case_id=None,
            substage=None, authority_role="current", artifact_status="result",
            overrides=None):
        self.campaign(campaign_id, root, family)
        payload = payload if isinstance(payload, dict) else {"value": payload}
        case_id = case_id or path_case(source_path, stage)
        input_path = at(payload, "instance", "csv", "cell.instance",
                        "provenance.instance", "case.peak_relative_path",
                        "cache_manifest.identity.instance")
        metadata = self.authoritative_case(
            campaign_id, source_path, input_path, case_id,
        )
        if metadata:
            case_id = metadata["id"]
            input_path = metadata.get("csv") or input_path
        target_k, chain, replication = case_dimensions(case_id, input_path)
        if metadata:
            target_k = metadata.get("target_duties", target_k)
            chain = metadata.get("chain", chain)
            replication = metadata.get("replication", replication)
        provenance = at(payload, "provenance", default={}) or {}
        physics = at(payload, "physics", "vehicle_profile", default={}) or {}
        physical_audit = at(payload, "physical_pool_audit", default={}) or {}
        input_hashes = at(physical_audit, "input_hashes", default={}) or {}
        final = at(payload, "final", default={}) or {}
        final_lp = at(payload, "final_lp", default={}) or {}
        two = at(payload, "two_stage", "joint_solver", default={}) or {}
        stage1 = at(two, "stage1", default=two) or {}
        stage2 = at(two, "stage2", default=two) or {}
        full_lp_cert = at(payload, "certified_rc_optimal",
                          "full_model_lp_bound_certified")
        weighted_lp = at(final_lp, "objective", default=None)
        if weighted_lp is None:
            weighted_lp = at(final, "lp_obj", "objective", default=None)
        fractional = at(final_lp, "route_weight", default=None)
        if fractional is None:
            fractional = at(final, "route_weight", default=None)
        pool_size = at(payload, "pool_columns", "shared_union_pool_size",
                       "final.pool_columns", "last_iteration.pool_columns",
                       "pareto_augmented_pool_routes")
        master_sense = at(payload, "master_sense",
                          "pricer_provenance.args.master_sense",
                          "physical_pool_audit.master_sense")
        initialization = at(payload, "initial_pool", "seed_mode",
                            "mip_start.kind", "initialization")
        source_sha256 = source_sha256 or payload.get("sha256")
        if source_sha256 is not None and not SHA_RE.match(str(source_sha256)):
            raise ValueError(f"invalid source SHA-256 for {source_path}")
        row = {key: None for key in COLUMNS}
        row.update({
            "campaign_id": campaign_id, "campaign_root": root,
            "case_id": case_id, "result_family": family, "stage": stage,
            "substage": substage, "artifact_status": artifact_status,
            "authority_role": authority_role, "source_path": source_path,
            "source_sha256": source_sha256,
            "snapshot_payload_sha256": canonical_sha(payload),
            "snapshot_time_utc": self.snapshot_time,
            "file_mtime_utc": payload.get("file_mtime_utc"),
            "completion_marker_matches": payload.get("completion_marker_matches"),
            "job_ids": job_ids(payload),
            "input_path": input_path,
            "input_sha256": (metadata.get("input_sha256") if metadata else None)
                or at(provenance, "instance_sha256", default=None)
                or at(input_hashes, "instance_sha256", default=None)
                or at(payload, "cell.instance_sha256", "case.instance_sha256"),
            "trip_count": at(payload, "trip_count", "cell.trips"),
            "target_k": target_k, "chain": chain, "replication": replication,
            "tariff_path": at(payload, "tariff", "tariff_path", "cell.tariff",
                              "physics.prices_csv", "provenance.prices"),
            "tariff_sha256": at(payload, "tariff_sha256", "cell.tariff_sha256"),
            "prices_sha256": at(provenance, "prices_sha256", default=None)
                or at(input_hashes, "prices_sha256", default=None),
            "reference_sha256": at(provenance, "reference_sha256", default=None)
                or at(input_hashes, "reference_sha256", default=None),
            "deadhead_sha256": at(provenance, "deadhead_sha256", default=None)
                or at(input_hashes, "deadhead_sha256", default=None),
            "master_sha256": at(payload, "master_sha256"),
            "code_commit": at(provenance, "git_commit", default=None)
                or at(payload, "code_identity.observed_commit", "commit"),
            "code_branch": at(provenance, "git_branch", default=None)
                or at(payload, "code_identity.branch"),
            "code_dirty": at(provenance, "git_dirty", "git_tracked_dirty",
                             default=None),
            "solver_backend": at(payload, "master_backend", default=None)
                or at(provenance, "master_backend", default=None),
            "solver_version": at(provenance, "gurobi_version", default=None),
            "battery_kwh": at(physics, "battery_kwh", "g_kwh", default=None)
                or at(payload, "g_kwh"),
            "initial_soc_kwh": at(physics, "initial_soc_kwh", default=None),
            "reserve_kwh": at(physics, "reserve_kwh", default=None),
            "charge_kw": at(physics, "charge_kw", default=None)
                or at(payload, "charge_kw"),
            "parx_kw": at(physics, "parx_kw", default=None),
            "non_parx_kw": at(physics, "non_parx_kw", default=None),
            "soc_step_kwh": at(physics, "soc_step_kwh", "soc_step", default=None)
                or at(payload, "soc_step"),
            "block_minutes": at(physics, "event_block_min", "block_min",
                                default=None) or at(payload, "block_min"),
            "capacity_enforced": at(physics, "capacity_enforced", default=None),
            "charger_counts_json": at(physics, "charger_counts", default=None),
            "terminal_energy_policy": at(
                physics, "terminal_soc_constraint", "terminal_soc_policy",
                default=None,
            ) or at(payload, "terminal_constraint_semantics"),
            "charge_start_cost": at(payload, "charge_start_cost",
                                    "physics.charge_start_cost",
                                    "provenance.args.charge_start_cost"),
            "electricity_cost_grid": at(payload, "charging_comparison_metrics.expanded_grid_electricity_cost"),
            "electricity_cost_continuous": at(payload, "charging_comparison_metrics.continuous_electricity_cost"),
            "electricity_cost_lower": payload.get("energy_cost_lower"),
            "electricity_cost_upper": payload.get("energy_cost_upper"),
            "charging_start_fees_grid": at(payload, "charging_comparison_metrics.expanded_grid_charge_start_fees"),
            "charging_start_fees_continuous": at(payload, "charging_comparison_metrics.continuous_charge_start_fees"),
            "charging_starts_grid": at(payload, "charging_comparison_metrics.expanded_grid_charging_starts"),
            "charging_starts_continuous": at(payload, "charging_comparison_metrics.continuous_charging_starts"),
            "charged_energy_grid_kwh": at(payload, "charging_comparison_metrics.expanded_grid_charging_kwh"),
            "charged_energy_continuous_kwh": at(payload, "charging_comparison_metrics.continuous_charging_kwh"),
            "charging_metrics_reconcile": at(payload, "charging_comparison_metrics.cost_components_reconcile"),
            "master_sense": master_sense,
            "initialization": initialization,
            "column_pool_treatment": at(payload, "column_pool_treatment",
                                         "provenance.column_pool_treatment"),
            "pool_size": pool_size,
            "cg_iterations": at(payload, "attempt_iterations", "iteration_count"),
            "stop_reason": payload.get("stop_reason"),
            "runtime_s": at(payload, "runtime_s", "wall_s"),
            "weighted_lp_objective": None,
            "recorded_lp_objective": weighted_lp,
            "lp_objective_kind": None,
            "fractional_fleet": fractional,
            "artificial_total": at(final_lp, "artificial_total", default=None)
                or at(final, "artificials", "artificial_total", default=None),
            "min_reduced_cost": at(payload, "terminal_exact_min_reduced_cost",
                                   "last_iteration.min_reduced_cost")
                or at(final, "min_rc"),
            "full_model_lp_certified": full_lp_cert,
            "lp_bound_scope": at(payload, "pricing_certificate_scope",
                                 "reporting_scope.reason"),
            "fleet_lower_bound": payload.get("fleet_bound"),
            "fleet_lower_bound_scope": (
                "restricted-pool integer MIP bound"
                if payload.get("fleet_bound") is not None else None
            ),
            "mip_incumbent_fleet": at(payload, "buses", "fleet"),
            "mip_bound_fleet": payload.get("fleet_bound"),
            "mip_status": at(payload, "status_name", "status"),
            "mip_gap": payload.get("mip_gap"),
            "fleet_proven": payload.get("fleet_proven"),
            "optimal_scope": payload.get("optimal_scope"),
            "stage1_incumbent_fleet": at(stage1, "stage1_buses", "buses"),
            "stage1_bound": at(stage1, "stage1_bound", "bound"),
            "stage1_gap": at(stage1, "stage1_gap", "gap"),
            "stage1_proven": at(stage1, "fleet_proven", "proven"),
            "stage2_executed": at(stage2, "stage2_executed", default=None),
            "stage2_status": at(stage2, "stage2_status_name", "status"),
            "stage2_charging_cost": at(stage2, "stage2_variable_obj", "objective"),
            "stage2_charging_bound": at(stage2, "stage2_variable_bound", "bound"),
            "stage2_gap": at(stage2, "stage2_absolute_gap", "gap"),
            "charging_cost_grid": at(payload, "charging_cost",
                                     "expanded_grid_charging_cost"),
            "charging_cost_continuous": at(
                payload, "continuous_realized_charging_cost",
                "physical_charging_cost",
            ),
            "charging_cost_exact": payload.get("charging_cost_exact"),
            "charging_cost_lower": payload.get("charging_cost_lower"),
            "charging_cost_upper": payload.get("charging_cost_upper"),
            "comparator_eligible": payload.get(
                "matched_physics_comparator_eligible"
            ),
            "beats_original_robustly": payload.get("beats_original_robustly"),
            "terminal_energy_grid_kwh": payload.get(
                "expanded_grid_terminal_energy_kwh"
            ),
            "terminal_energy_continuous_kwh": payload.get(
                "continuous_terminal_energy_kwh"
            ),
            "physical_pool_validated": (
                physical_audit.get("rejected_columns") == 0
                if physical_audit else None
            ),
            "physical_selected_validated": payload.get("physical_replay_validated"),
            "physical_validation_scope": payload.get("physical_replay_scope"),
            "overcovered_trips": payload.get("overcovered_trips"),
            "cross_route_capacity_validated": payload.get(
                "cross_route_charger_capacity_validated"
            ),
            "proof_scope": at(payload, "pricing_certificate_scope",
                              "certificate_scope", "claim_scope",
                              "proof_scope.joint"),
            "limitations": at(payload, "limitation", "reporting_scope.reason"),
            "notes": payload.get("provenance_warning"),
        })
        if overrides:
            row.update(overrides)
        # The compatibility field has deliberately narrow semantics.  An
        # objective of 2.0 from the small same-pool LP is fleet-only and must
        # never appear as a bus-plus-charging weighted objective.
        row["weighted_lp_objective"] = (
            row.get("recorded_lp_objective")
            if row.get("lp_objective_kind") == "bus_plus_charging"
            else None
        )
        locator = "|".join(str(row.get(key) or UNKNOWN) for key in (
            "campaign_id", "case_id", "result_family", "stage", "substage",
            "arm", "source_path", "source_sha256",
        ))
        row["row_id"] = hashlib.sha256(locator.encode()).hexdigest()
        row = {key: finite(row.get(key)) for key in COLUMNS}
        unknown_fields = [key for key in COLUMNS if key != "row_id" and row[key] is None]
        row["unknown_fields"] = unknown_fields
        row["details"] = payload
        self.rows.append(row)
        self.campaigns[campaign_id]["workflow_job_ids"] = sorted(set(
            self.campaigns[campaign_id]["workflow_job_ids"] + row["job_ids"],
        ), key=int)
        return row

    def add_giro_fee_comparison(self, campaign_id, campaign, item):
        """Expand one authenticated fee cell without treating it as CG proof."""
        root = campaign.get("root")
        result = item.get("result") or {}
        cell = result.get("cell") or {}
        pair_id = item.get("pair_id") or cell.get("id")
        fee = item.get("destination_charge_start_fee")
        if fee is None:
            fee = at(result, "unchanged_conditions.charge_start_fee")
        proof = result.get("proof_scope") or {}
        conditions = result.get("unchanged_conditions") or {}
        completed = (
            item.get("completion_marker_matches") is True
            and item.get("mip_completion_marker_matches") is True
        )
        common = {
            "cell": cell,
            "charge_start_cost": fee,
            "physics": {
                "charge_start_cost": fee,
                "charge_kw": conditions.get("charge_power_kw"),
                "initial_soc_kwh": conditions.get("initial_energy_per_bus_kwh"),
                "capacity_enforced": conditions.get("station_capacity_modeled"),
            },
            "terminal_constraint_semantics": result.get(
                "terminal_constraint_semantics"),
            "target_terminal_energy_kwh": result.get(
                "target_terminal_energy_kwh"),
            "fee_provenance": result.get("fee_provenance"),
            "commit": campaign.get("code_commit"),
            "completion_marker_matches": completed,
        }
        original = result.get("original_giro")
        fixed = result.get("fixed_duties_optimized")
        joint = result.get("joint_pool_optimized")
        solver = result.get("joint_solver") or {}
        stage1 = solver.get("stage1") or {}
        stage2 = solver.get("stage2") or {}
        arms = []
        if isinstance(original, dict):
            arms.append(("original_giro", original, {
                "mip_incumbent_fleet": original.get("fleet"),
                "charging_cost_exact": original.get("charging_cost_exact"),
                "charging_cost_lower": original.get("charging_cost_lower"),
                "charging_cost_upper": original.get("charging_cost_upper"),
                "terminal_energy_continuous_kwh": original.get(
                    "terminal_surplus_total_kwh"),
                "comparator_eligible": original.get(
                    "matched_physics_comparator_eligible"),
                "proof_scope": original.get("comparator"),
                "limitations": "; ".join(original.get("cost_errors") or []) or None,
            }))
        if isinstance(fixed, dict):
            arms.append(("fixed_duties", fixed, {
                "mip_incumbent_fleet": fixed.get("fleet"),
                "charging_cost_grid": fixed.get("expanded_grid_charging_cost"),
                "charging_cost_continuous": fixed.get("physical_charging_cost"),
                "terminal_energy_grid_kwh": fixed.get(
                    "expanded_grid_terminal_energy_kwh"),
                "terminal_energy_continuous_kwh": fixed.get(
                    "continuous_terminal_energy_kwh"),
                "overcovered_trips": fixed.get("overcovered_trip_count"),
                "physical_selected_validated": giro_physical_validation(fixed),
                "physical_validation_scope": proof.get("physical"),
                "proof_scope": proof.get("fixed"),
                "optimal_scope": "fixed ordered duties over enumerated terminal frontier",
            }))
        if isinstance(joint, dict):
            stage2_executed = stage2.get("status") is not None
            runtime_parts = [value for value in (
                stage1.get("runtime_s"), stage2.get("runtime_s"),
                result.get("pool_replay_s"),
            ) if isinstance(value, (int, float))]
            arms.append(("joint", joint, {
                "mip_incumbent_fleet": joint.get("fleet"),
                "mip_status": stage2.get("status") if stage2_executed else stage1.get("status"),
                "mip_gap": stage2.get("gap") if stage2_executed else stage1.get("gap"),
                "fleet_proven": stage1.get("proven"),
                "optimal_scope": "finite saved pool plus common fee-frontier union",
                "stage1_incumbent_fleet": stage1.get("buses"),
                "stage1_bound": stage1.get("bound"),
                "stage1_gap": stage1.get("gap"),
                "stage1_proven": stage1.get("proven"),
                "stage2_executed": stage2_executed,
                "stage2_status": stage2.get("status"),
                "stage2_charging_cost": stage2.get("objective"),
                "stage2_charging_bound": stage2.get("bound"),
                "stage2_gap": stage2.get("gap"),
                "charging_cost_grid": joint.get("expanded_grid_charging_cost"),
                "charging_cost_continuous": joint.get("physical_charging_cost"),
                "terminal_energy_grid_kwh": joint.get(
                    "expanded_grid_terminal_energy_kwh"),
                "terminal_energy_continuous_kwh": joint.get(
                    "continuous_terminal_energy_kwh"),
                "overcovered_trips": joint.get("overcovered_trip_count"),
                "pool_size": result.get("pareto_augmented_pool_routes"),
                "runtime_s": sum(runtime_parts) if runtime_parts else None,
                "physical_selected_validated": giro_physical_validation(joint),
                "physical_validation_scope": proof.get("physical"),
                "proof_scope": proof.get("joint"),
            }))
        for arm, arm_result, overrides in arms:
            payload = {**common, **arm_result}
            self.add(
                campaign_id, root, "giro_zero_fee_comparison",
                "comparison_arm", payload,
                source_path=item.get("path"), source_sha256=item.get("sha256"),
                case_id=pair_id, substage=arm,
                artifact_status=("result" if completed
                                 else "unverified_result"),
                overrides={
                    **overrides, "arm": arm,
                    "input_path": cell.get("instance") or item.get("instance_path"),
                    "input_sha256": cell.get("instance_sha256") or item.get("instance_sha256"),
                    "tariff_path": cell.get("tariff_path") or item.get("tariff_path"),
                    "tariff_sha256": cell.get("tariff_sha256") or item.get("tariff_sha256"),
                    "charge_start_cost": fee,
                    "capacity_enforced": conditions.get("station_capacity_modeled"),
                    "full_model_lp_certified": False,
                    "lp_bound_scope": "no full-model column-generation pricing certificate",
                },
            )

    @staticmethod
    def fee_case_fields(campaign_id, campaign, item):
        if campaign_id != "zero_charge_start_fee_20260913":
            return {}
        case = item.get("source_case_id")
        match = re.fullmatch(r"w(\d+)_k(\d+)", str(case))
        arm = item.get("arm")
        if not match or arm not in {"fee0", "fee5"}:
            raise ValueError("fee result lacks explicit chain/target/arm identity")
        declared_pairs = {(p["id"], p["case_id"]) for p in campaign.get("pairs", [])}
        if (item.get("pair_id"), case) not in declared_pairs:
            raise ValueError("fee result differs from frozen campaign pair")
        cg = next((r for r in campaign.get("cg", [])
                   if r.get("source_case_id") == case and r.get("arm") == arm), {})
        input_path = item.get("csv") or item.get("instance") or cg.get("csv")
        # Read the actual CSV basename, never the enclosing k2_15 directory.
        csv_match = re.search(r"_k(\d+)_p(\d+)_", Path(input_path or "").name)
        if not csv_match or tuple(map(int, csv_match.groups())) != (int(match[2]), int(match[1])):
            raise ValueError("fee result CSV differs from explicit chain/target")
        fee = float(item["charge_start_cost"])
        if fee != {"fee0": 0.0, "fee5": 5.0}[arm]:
            raise ValueError("fee arm and recorded charge-start cost differ")
        metrics = item.get("charging_comparison_metrics") or {}
        return {"case_id": case, "substage": arm, "overrides": {
            "target_k": int(match[2]), "chain": int(match[1]),
            "input_path": input_path,
            "input_sha256": at(cg, "provenance.instance_sha256"),
            "master_sense": cg.get("master_sense"),
            "initialization": "saved sequences replayed under destination fee plus singletons",
            "column_pool_treatment": "WARM-INHERITED-EVENT",
            "charging_start_fees_grid": metrics.get("expanded_grid_start_fees"),
            "charging_start_fees_continuous": metrics.get("continuous_start_fees"),
            "terminal_energy_grid_kwh": metrics.get("expanded_grid_terminal_kwh"),
            "terminal_energy_continuous_kwh": metrics.get("continuous_terminal_kwh"),
        }}

    def standard_campaigns(self):
        for campaign_id, campaign in self.snapshot.get("campaigns", {}).items():
            root = campaign.get("root")
            family = ("giro_zero_fee_comparison"
                      if campaign_id == "giro_zero_start_fee_20260913"
                      else "production_or_historical")
            self.campaign(campaign_id, root, family)
            if (campaign_id == "giro_zero_start_fee_20260913"
                    and campaign.get("schema")
                    != "evsp-dr-terminal-energy-fee-comparison-collection-v1"):
                raise ValueError("unexpected GIRO zero-fee collection schema")
            legacy = any(token in campaign_id for token in (
                "legacy", "historical", "old70",
            ))
            for item in campaign.get("mip", []):
                authority = "legacy_historical" if legacy else "current"
                if item.get("authoritative_772009_completion") is False:
                    authority = "superseded_duplicate_772031"
                elif item.get("authoritative_772009_result") is True:
                    authority = "authoritative_772009_recovery"
                self.add(
                    campaign_id, root, "pool_mip", "mip", item,
                    source_path=item.get("path"), source_sha256=item.get("sha256"),
                    authority_role=authority,
                    artifact_status=("superseded" if authority.startswith("superseded")
                                     else "result"),
                    **self.fee_case_fields(campaign_id, campaign, item),
                )
            for item in campaign.get("cg", []):
                fee_fields = self.fee_case_fields(campaign_id, campaign, item)
                fee_overrides = fee_fields.pop("overrides", {})
                self.add(campaign_id, root, "column_generation", "cg", item,
                         source_path=item.get("path"),
                         source_sha256=item.get("sha256"), authority_role=(
                             "legacy_historical" if legacy else "current"
                         ), overrides={"lp_objective_kind": "bus_plus_charging", **fee_overrides},
                         **fee_fields)
            for item in campaign.get("phases", []):
                self.add(
                    campaign_id, root, "cg_phase_telemetry", "telemetry", item,
                    source_path=item.get("path"), authority_role=(
                        "legacy_historical" if legacy else "current"
                    ),
                    overrides={
                        "runtime_s": sum((item.get("duration_s_by_phase") or {}).values()),
                        "phase_runtime_json": item.get("duration_s_by_phase"),
                        "pool_size": at(item, "last_record.pool_columns"),
                        "cg_iterations": at(item, "last_record.iteration"),
                        "artifact_status": "telemetry",
                    },
                )
            for item in campaign.get("comparisons", []):
                if campaign_id == "giro_zero_start_fee_20260913":
                    self.add_giro_fee_comparison(campaign_id, campaign, item)
                    continue
                result = item.get("result") or {}
                if "fleet_sum" in result and "components" in result:
                    self.add(
                        campaign_id, root, "decomposition", "decomposition_join",
                        result, source_path=item.get("path"),
                        source_sha256=item.get("sha256"), overrides={
                            "mip_incumbent_fleet": result["fleet_sum"],
                            "proof_scope": result.get("scope"),
                            "capacity_enforced": result.get("shared_capacity_enforced"),
                            "limitations": "Component schedules combined; no full-model integer proof or shared-capacity validation.",
                        },
                    )
                    continue
                cell = at(result, "cell.cell")
                fixed = result.get("fixed_duties_reoptimized") or {}
                joint = result.get("joint") or {}
                original = result.get("original") or {}
                common = {
                    "cell": result.get("cell"),
                    "absolute_cost_gap": result.get("absolute_cost_gap"),
                    "cost_optimal_scope": result.get("cost_optimal_scope"),
                    "fee_scope": result.get("fee_scope"),
                    "limitation": result.get("limitation"),
                    "beats_original_robustly": result.get("beats_original_robustly"),
                }
                arms = (
                    ("original_giro", original, {
                        "mip_incumbent_fleet": original.get("fleet"),
                        "charging_cost_exact": original.get("charging_cost_exact"),
                        "charging_cost_lower": original.get("charging_cost_lower"),
                        "charging_cost_upper": original.get("charging_cost_upper"),
                        "terminal_energy_continuous_kwh": original.get(
                            "terminal_surplus_total_kwh"
                        ),
                        "comparator_eligible": original.get(
                            "matched_physics_comparator_eligible"
                        ),
                        "proof_scope": original.get("comparator"),
                        "limitations": "; ".join(original.get("cost_errors") or [])
                            or result.get("limitation"),
                    }),
                    ("fixed_duties", fixed, {
                        "mip_incumbent_fleet": fixed.get("fleet"),
                        "charging_cost_grid": fixed.get(
                            "conservative_grid_charging_cost"
                        ),
                        "charging_cost_continuous": fixed.get(
                            "physical_charging_cost"
                        ),
                        "terminal_energy_continuous_kwh": fixed.get(
                            "terminal_surplus_kwh"
                        ),
                        "proof_scope": "fixed GIRO duties with event charging reoptimized",
                        "limitations": result.get("limitation"),
                    }),
                    ("joint", joint, {
                        "mip_incumbent_fleet": joint.get("fleet"),
                        "charging_cost_grid": joint.get(
                            "conservative_grid_charging_cost"
                        ),
                        "charging_cost_continuous": joint.get(
                            "physical_charging_cost"
                        ),
                        "terminal_energy_continuous_kwh": joint.get(
                            "terminal_surplus_kwh"
                        ),
                        "proof_scope": result.get("cost_optimal_scope"),
                        "limitations": result.get("limitation"),
                    }),
                )
                for arm, arm_result, overrides in arms:
                    self.add(
                        campaign_id, root, "matched_tariff_comparison",
                        "comparison_arm", {**common, "arm_result": arm_result},
                        source_path=item.get("path"),
                        source_sha256=item.get("sha256"), case_id=cell,
                        substage=arm, overrides={
                            **overrides, "arm": arm,
                            "beats_original_robustly": result.get(
                                "beats_original_robustly"
                            ),
                            "input_path": at(result, "cell.instance"),
                            "input_sha256": at(result, "cell.instance_sha256"),
                            "tariff_path": at(result, "cell.tariff"),
                            "tariff_sha256": at(result, "cell.tariff_sha256"),
                        },
                    )
            for item in campaign.get("rejected_mip_outputs", []):
                result = item.get("result") or {}
                self.add(
                    campaign_id, root, "rejected_mip_output", "mip_rejected",
                    result, source_path=item.get("path"),
                    source_sha256=item.get("sha256"),
                    authority_role="rejected_observational",
                    artifact_status="rejected_physical_replay",
                    overrides={
                        "physical_selected_validated": False,
                        "physical_validation_scope": at(result, "failure.reason"),
                        "limitations": "Rejected output is observational only and is not a usable physical schedule.",
                    },
                )
            for name, value in campaign.get("workflow", {}).items():
                if isinstance(value, str) and name.endswith(".tsv"):
                    records = list(csv.DictReader(io.StringIO(value), delimiter="\t"))
                    for record in records:
                        scale = record.get("scale") or UNKNOWN
                        if name == "cg_jobs.tsv":
                            stages = (("cg", "job_id", "dependency"),)
                        else:
                            stages = (
                                ("cg_reference", "cg_job", "cg_dependency"),
                                ("freeze", "freeze_job", "freeze_dependency"),
                                ("mip", "mip_job", "mip_dependency"),
                            )
                        for workflow_stage, job_key, dependency_key in stages:
                            job = record.get(job_key)
                            if not job:
                                continue
                            self.add(
                                campaign_id, root, "workflow",
                                f"workflow_{workflow_stage}", record,
                                source_path=str(Path(root) / name),
                                case_id=(f"k{int(scale):02d}_p{int(record['replicate'])}"
                                         if record.get("replicate") else f"k{int(scale):02d}"),
                                substage=f"{name}:{workflow_stage}",
                                artifact_status="workflow_record",
                                overrides={
                                    "job_ids": [job],
                                    "dependency": record.get(dependency_key),
                                    "workflow_state": "submitted_or_planned",
                                    "target_k": int(scale),
                                },
                            )
                    continue
                self.add(
                    campaign_id, root, "workflow", "workflow", value,
                    source_path=str(Path(root) / name), case_id=name,
                    artifact_status="workflow_record",
                    overrides={
                        "workflow_state": at(value, "state", "status")
                            if isinstance(value, dict) else None,
                        "dependency": at(value, "dependency")
                            if isinstance(value, dict) else None,
                    },
                )

    def capacity_speed(self, campaign_id="capacity_speed_pilot"):
        value = self.snapshot.get(campaign_id, self.snapshot.get("campaigns", {}).get(campaign_id, {}))
        root = value.get("root")
        self.campaign(campaign_id, root, "controlled_physics_pilot")
        for item in value.get("records", []):
            result = item.get("result") or {}
            overrides = None
            if item.get("phase") == "mip":
                solved = result.get("result") or {}
                stage1 = solved.get("stage1") or {}
                stage2 = solved.get("stage2") or {}
                capacity_audit = result.get("physical_station_capacity_audit") or {}
                duplicate_audit = result.get("duplicate_service_audit") or {}
                overrides = {
                    "mip_incumbent_fleet": solved.get("fleet"),
                    "mip_bound_fleet": stage1.get("fleet_bound_raw"),
                    "mip_status": solved.get("status"),
                    "mip_gap": stage2.get("charging_cost_gap"),
                    "fleet_proven": stage1.get("fleet_proven"),
                    "stage1_incumbent_fleet": stage1.get("incumbent_fleet"),
                    "stage1_bound": stage1.get("fleet_bound_raw"),
                    "stage1_gap": stage1.get("mip_gap"),
                    "stage1_proven": stage1.get("fleet_proven"),
                    "stage2_executed": stage2.get("executed"),
                    "stage2_status": stage2.get("status"),
                    "stage2_charging_cost": stage2.get("charging_cost"),
                    "stage2_charging_bound": stage2.get("charging_cost_bound"),
                    "stage2_gap": stage2.get("charging_cost_gap"),
                    "charging_cost_grid": solved.get("charging_related_cost"),
                    "physical_selected_validated": stage1.get("validated_incumbent"),
                    "physical_validation_scope": stage1.get(
                        "incumbent_validation_scope"
                    ),
                    "overcovered_trips": duplicate_audit.get(
                        "overcovered_trip_count"
                    ),
                    "cross_route_capacity_validated": capacity_audit.get("valid"),
                    "capacity_enforced": result.get("capacity_enforced_in_mip"),
                    "pool_size": at(result, "pool_acceptance.route_count"),
                    "proof_scope": at(result, "pool_acceptance.classification"),
                }
            self.add(
                campaign_id, root, "capacity_speed_exact_event", item.get("phase"),
                result, source_path=item.get("path"),
                source_sha256=item.get("sha256"),
                case_id=path_case(item.get("path"), item.get("phase")),
                overrides={**(overrides or {}), **({
                    "lp_objective_kind": "bus_plus_charging",
                } if item.get("phase") == "cg" else {})},
            )
        for name in ("submission", "authoritative_mip_submission"):
            if name in value:
                self.add(
                    campaign_id, root, "workflow", "workflow", value[name],
                    source_path=str(Path(root) / (
                        "submission_twostage.json" if name.startswith("authoritative")
                        else "submission.json"
                    )), case_id=name, artifact_status="workflow_record",
                    authority_role=("authoritative" if name.startswith("authoritative")
                                    else "superseded_weighted_mip_submission"),
                )

    def terminal_energy(self):
        campaign_id = "terminal_energy_fair"
        value = self.snapshot.get(campaign_id, {})
        root = value.get("root")
        self.campaign(campaign_id, root, "matched_terminal_energy")
        for item in value.get("records", []):
            result = item.get("result") or {}
            overrides = {"completion_marker_matches": item.get(
                "completion_marker_matches"
            )}
            if item.get("phase") == "frontier":
                fixed = result.get("fixed_solution") or {}
                master = result.get("fixed_master") or {}
                overrides.update({
                    "mip_incumbent_fleet": fixed.get("fleet"),
                    "mip_status": master.get("status"),
                    "mip_gap": master.get("gap"),
                    "charging_cost_grid": fixed.get(
                        "expanded_grid_charging_cost"
                    ),
                    "charging_cost_continuous": fixed.get(
                        "physical_charging_cost"
                    ),
                    "terminal_energy_grid_kwh": fixed.get(
                        "expanded_grid_terminal_energy_kwh"
                    ),
                    "terminal_energy_continuous_kwh": fixed.get(
                        "continuous_terminal_energy_kwh"
                    ),
                    "overcovered_trips": fixed.get("overcovered_trip_count"),
                    "physical_selected_validated": True,
                    "physical_validation_scope": (
                        "selected fixed-duty event routes with continuous replay"
                    ),
                    "proof_scope": result.get("certificate_scope"),
                    "runtime_s": result.get("network_build_s"),
                })
            self.add(
                campaign_id, root, "terminal_energy", item.get("phase"),
                result, source_path=item.get("path"),
                source_sha256=item.get("sha256"), case_id=item.get("cell"),
                overrides=overrides,
            )
        for name, record in value.get("workflow", {}).items():
            self.add(
                campaign_id, root, "workflow", "workflow", record,
                source_path=str(Path(root) / name), case_id=name,
                artifact_status="workflow_record",
            )
        cell_by_task = {"0": "peak08", "1": "peak12", "2": "peak18"}
        for path, lines in value.get("stderr_tails", {}).items():
            match = re.search(r"mip_778802_(\d+)\.err$", path)
            task = match.group(1) if match else None
            cell = cell_by_task.get(task, path_case(path, "mip_failure"))
            text = "\n".join(lines)
            failure = (
                "saved-pool route missing expanded_grid_terminal_soc_kwh; "
                "failed during pre-optimization replay"
                if "KeyError: 'expanded_grid_terminal_soc_kwh'" in text
                else "terminal-energy MIP failed; see stderr tail"
            )
            self.add(
                campaign_id, root, "terminal_energy", "mip_failed_preoptimization",
                {"stderr_tail": lines, "job_id": "778802", "state": "FAILED"},
                source_path=path, case_id=cell,
                artifact_status="failed_preoptimization",
                authority_role="failed_attempt_retained",
                overrides={
                    "job_ids": ["778802"], "workflow_state": "FAILED",
                    "mip_status": "FAILED_BEFORE_OPTIMIZATION",
                    "physical_selected_validated": False,
                    "limitations": failure,
                    "notes": failure,
                },
            )

    def small_cg(self):
        for campaign_id in (
            "giro_small_cg_newphysics", "giro_small_cg_capacity_duals",
        ):
            value = self.snapshot.get(campaign_id, {})
            root = value.get("root")
            self.campaign(campaign_id, root, "small_reference_cg")
            for item in value.get("results", []):
                base = {key: item.get(key) for key in (
                    "instance", "trip_count", "vehicle_profile", "seed_mode",
                    "shared_union_pool_size", "reporting_scope",
                )}
                case_id = path_case(item.get("path"), "small_case")
                for index, arm in enumerate(item.get("cg_arms", [])):
                    payload = {**base, **arm}
                    self.add(
                        campaign_id, root, "small_reference_cg", "cg_arm", payload,
                        source_path=item.get("path"),
                        source_sha256=item.get("sha256"), case_id=case_id,
                        substage=f"arm{index}:{arm.get('sense')}",
                        overrides={
                            "master_sense": arm.get("sense"),
                            "pool_size": arm.get("final_arm_pool_size"),
                            "full_model_lp_certified": arm.get(
                                "full_model_lp_bound_certified"
                            ),
                            "proof_scope": at(base, "reporting_scope.reason"),
                        },
                    )
                for capacity, senses in item.get(
                    "final_same_pool_comparisons", {}
                ).items():
                    for sense, stages in senses.items():
                        for stage in ("lp", "mip"):
                            result = stages.get(stage)
                            if not result:
                                continue
                            payload = {**base, **result,
                                       "capacity_treatment": capacity,
                                       "master_sense": sense}
                            overrides = {
                                "master_sense": sense,
                                "capacity_enforced": capacity == "constrained",
                                "pool_size": item.get("shared_union_pool_size"),
                                "proof_scope": at(base, "reporting_scope.reason"),
                            }
                            if stage == "lp":
                                overrides.update({
                                    "weighted_lp_objective": None,
                                    "recorded_lp_objective": result.get("objective"),
                                    "lp_objective_kind": "fleet_only",
                                    "fractional_fleet": result.get("route_weight"),
                                    "fleet_lower_bound": result.get(
                                        "restricted_pool_fleet_lp"
                                    ),
                                    "fleet_lower_bound_scope": "restricted-pool LP only",
                                    "full_model_lp_certified": False,
                                    "lp_bound_scope": "restricted shared union pool",
                                })
                            else:
                                overrides.update({
                                    "mip_incumbent_fleet": result.get("objective"),
                                    "mip_bound_fleet": result.get("objective_bound"),
                                    "mip_status": result.get("status"),
                                    "mip_gap": result.get("mip_gap"),
                                    "fleet_lower_bound_scope": "restricted-pool MIP",
                                })
                            self.add(
                                campaign_id, root, "small_same_pool_comparison",
                                stage, payload, source_path=item.get("path"),
                                source_sha256=item.get("sha256"), case_id=case_id,
                                substage=f"{capacity}:{sense}:{stage}",
                                overrides=overrides,
                            )

    def targeted(self):
        for audit_id, value in self.snapshot.get("targeted_audits", {}).items():
            root = value.get("root")
            campaign_id = f"targeted_audit:{audit_id}"
            self.campaign(campaign_id, root, "targeted_audit")
            for item in value.get("records", []):
                data = item.get("data") or {}
                name = Path(item.get("path", "record")).name
                stage = "workflow" if "submission" in name else "audit_result"
                self.add(
                    campaign_id, root, "targeted_audit", stage, data,
                    source_path=item.get("path"), source_sha256=item.get("sha256"),
                    case_id=at(data, "case.case_id") or path_case(item.get("path"), stage),
                    artifact_status=("workflow_record" if stage == "workflow" else "result"),
                    overrides={
                        "fleet_lower_bound": at(data, "peak_bound.value"),
                        "fleet_lower_bound_scope": at(data, "peak_bound.definition"),
                        "mip_incumbent_fleet": at(data, "integer_witness_routes"),
                        "fleet_proven": at(data, "model_integer_optimum_proven"),
                    },
                )
            for path, tail in value.get("log_tails", {}).items():
                self.add(
                    campaign_id, root, "targeted_audit_log", "log_tail",
                    {"tail": tail}, source_path=path,
                    artifact_status="diagnostic_log",
                )

    def remaining_workflow(self):
        root = at(self.snapshot, "campaigns.stage2_cap_license_recovery.root")
        for path, payload in self.snapshot.get("post_meeting_log_tails", {}).items():
            self.add(
                "post_meeting_log_tails", root, "solver_progress_log", "log_tail",
                payload, source_path=path, artifact_status="diagnostic_log",
            )
        queue = self.snapshot.get("squeue", {})
        for line in (queue.get("stdout") or "").splitlines():
            parts = line.split("|", 3)
            if len(parts) != 4:
                continue
            job, state, runtime, reason = parts
            self.add(
                "scheduler_snapshot", None, "scheduler", "queue_state",
                {"job_id": job, "state": state, "runtime": runtime,
                 "reason_or_node": reason},
                source_path=f"snapshot:squeue:{job}", case_id=job,
                artifact_status="workflow_state",
                overrides={"job_ids": [re.sub(r"_.*", "", job)],
                           "workflow_state": state, "notes": reason},
            )

    def build(self):
        self.standard_campaigns()
        self.capacity_speed()
        for retry_name in ["capacity_timeout6_rerun", "capacity_deadline5_retry"]:
            if self.snapshot.get("campaigns", {}).get(retry_name, {}).get("records"):
                self.capacity_speed(retry_name)
        self.terminal_energy()
        self.small_cg()
        self.targeted()
        self.remaining_workflow()
        self.validate()

    def validate_overnight_case_metadata(self):
        """Validate metadata for manifest-backed overnight rows that exist."""
        if not self.authoritative_cases:
            return
        overnight_rows = [
            row for row in self.rows
            if row["campaign_id"] == "overnight_extension_20260912"
        ]
        for row in overnight_rows:
            metadata = self.authoritative_cases.get(row["case_id"])
            if not metadata:
                continue
            expected_target = metadata.get("target_duties")
            if (expected_target is not None
                    and row["target_k"] != expected_target):
                raise ValueError(
                    f"overnight manifest target mismatch for {row['stage']}:"
                    f"{row['case_id']}: expected target_k={expected_target}"
                )
            expected_input = metadata.get("csv")
            if expected_input and row["input_path"] != expected_input:
                raise ValueError(
                    f"overnight manifest input mismatch for {row['case_id']}"
                )
            expected_hash = metadata.get("input_sha256")
            if expected_hash and row["input_sha256"] != expected_hash:
                raise ValueError(
                    f"overnight manifest input hash mismatch for {row['case_id']}"
                )

    def validate(self):
        self.validate_overnight_case_metadata()
        ids = [row["row_id"] for row in self.rows]
        if len(ids) != len(set(ids)):
            duplicates = [key for key, count in Counter(ids).items() if count > 1]
            raise ValueError(f"duplicate stable row IDs: {duplicates[:5]}")
        for row in self.rows:
            for key in ("source_sha256", "snapshot_payload_sha256"):
                if row[key] is not None and not SHA_RE.match(str(row[key])):
                    raise ValueError(f"invalid {key} in {row['row_id']}")
            if row["weighted_lp_objective"] is not None and row["fractional_fleet"] is None:
                # Some historical artifacts do not retain route weight; leave
                # it unknown instead of deriving objective / 100000.
                pass
            if (row["weighted_lp_objective"] is not None
                    and row["lp_objective_kind"] != "bus_plus_charging"):
                raise ValueError(
                    "weighted_lp_objective requires lp_objective_kind="
                    f"bus_plus_charging: {row['row_id']}"
                )
            if row["full_model_lp_certified"] is True and not row["lp_bound_scope"]:
                row["lp_bound_scope"] = (
                    "pricing certificate within the source graph/approximation; "
                    "objective and fractional fleet remain separate"
                )
        same_pool_lp = [
            row for row in self.rows
            if row["result_family"] == "small_same_pool_comparison"
            and row["stage"] == "lp"
        ]
        if not same_pool_lp or any(
            row["lp_objective_kind"] != "fleet_only"
            or row["weighted_lp_objective"] is not None
            or row["recorded_lp_objective"] is None
            for row in same_pool_lp
        ):
            raise ValueError(
                "same-pool LP rows must retain a fleet-only recorded objective "
                "without populating weighted_lp_objective"
            )
        comparison_arms = Counter(
            row["arm"] for row in self.rows
            if row["result_family"] == "matched_tariff_comparison"
        )
        if comparison_arms and set(comparison_arms) != {
            "original_giro", "fixed_duties", "joint"
        }:
            raise ValueError(f"incomplete tariff comparison arms: {comparison_arms}")
        giro_rows = [
            row for row in self.rows
            if row["result_family"] == "giro_zero_fee_comparison"
        ]
        giro_by_case = defaultdict(list)
        for row in giro_rows:
            giro_by_case[row["case_id"]].append(row)
            if row["charge_start_cost"] not in {0, 0.0, 5, 5.0}:
                raise ValueError(
                    f"GIRO fee arm has missing/unexpected fee: {row['case_id']}"
                )
            if row["full_model_lp_certified"] is not False:
                raise ValueError(
                    f"GIRO fee arm must not claim full-model CG proof: {row['row_id']}"
                )
        for case_id, rows in giro_by_case.items():
            if {row["arm"] for row in rows} != {
                    "original_giro", "fixed_duties", "joint"}:
                raise ValueError(f"incomplete GIRO fee comparison: {case_id}")
            joint = next(row for row in rows if row["arm"] == "joint")
            if (joint["stage2_executed"] is True
                    and joint["mip_incumbent_fleet"] is not None
                    and joint["stage1_incumbent_fleet"] is not None
                    and joint["mip_incumbent_fleet"]
                    > joint["stage1_incumbent_fleet"]):
                raise ValueError(
                    f"GIRO stage 2 exceeds stage-1 fleet incumbent: {case_id}"
                )
        superseded = [r for r in self.rows
                      if r["authority_role"] == "superseded_duplicate_772031"]
        authoritative = [r for r in self.rows
                         if r["authority_role"] == "authoritative_772009_recovery"]
        if len(superseded) != 1 or len(authoritative) != 1:
            raise ValueError(
                "expected exactly one superseded 772031 and one authoritative 772009 row"
            )
        expected_campaigns = set(self.snapshot.get("campaigns", {})) | {
            "capacity_speed_pilot", "terminal_energy_fair",
            "giro_small_cg_newphysics", "giro_small_cg_capacity_duals",
        } | {f"targeted_audit:{name}" for name in
             self.snapshot.get("targeted_audits", {})}
        missing = expected_campaigns - set(self.campaigns)
        if missing:
            raise ValueError(f"campaigns missing from register: {sorted(missing)}")


def read_markdown(root, exclude_root=None):
    if root is None or not root.exists():
        return []
    output = []
    for path in root.rglob("*.md"):
        try:
            resolved = path.resolve()
            # This register is regenerated in place, and older generated
            # REGISTER/RESULTS files can mention every campaign.  Never treat
            # those derivative files as campaign source reports, even when a
            # dry run writes to a different output directory.
            if "research_register" in resolved.parts:
                continue
            if exclude_root is not None:
                try:
                    resolved.relative_to(exclude_root)
                    continue
                except ValueError:
                    pass
            if path.stat().st_size <= 2_000_000:
                output.append((resolved, path.read_text(errors="replace")))
        except OSError:
            continue
    return output


def csv_value(value):
    if value is None:
        return UNKNOWN
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False)
    return value


def md(value):
    value = csv_value(value)
    text = str(value).replace("|", "\\|").replace("\n", " ")
    if (text.startswith("[") and "](<" in text) or "<br>" in text:
        return text
    return text if len(text) <= 160 else text[:157] + "..."


def markdown_table(headers, records):
    output = ["| " + " | ".join(headers) + " |",
              "|" + "|".join("---" for _ in headers) + "|"]
    for record in records:
        output.append("| " + " | ".join(md(record.get(key)) for key in headers) + " |")
    return "\n".join(output)


def markdown_file_links(paths):
    links = []
    for value in paths or []:
        path = Path(value)
        label = f"{path.parent.name}/{path.name}"
        links.append(f"[{label}](<{path}>)")
    return "<br>".join(links) if links else UNKNOWN


def write_outputs(register, raw_name, raw_sha, supplemental_sources=None):
    out = register.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(register.rows, key=lambda row: (
        row["campaign_id"], row["case_id"], row["stage"],
        str(row["substage"]), row["row_id"],
    ))
    campaign_counts = Counter(row["campaign_id"] for row in rows)
    campaigns = []
    for campaign_id, record in sorted(register.campaigns.items()):
        campaigns.append({**record, "row_count": campaign_counts[campaign_id]})
    campaigns_markdown = [{
        **record,
        "workflow_job_ids": "<br>".join(record["workflow_job_ids"]) or UNKNOWN,
        "report_links": markdown_file_links(record["report_links"]),
    } for record in campaigns]
    payload = {
        "schema": SCHEMA,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_snapshot": {
            "path": f"raw/{raw_name}", "sha256": raw_sha,
            "timestamp_utc": register.snapshot_time,
        },
        "supplemental_sources": supplemental_sources or [],
        "campaign_count": len(campaigns), "row_count": len(rows),
        "campaigns": campaigns, "rows": rows,
    }
    (out / "register.json").write_text(json.dumps(
        payload, sort_keys=True, indent=2, allow_nan=False,
    ) + "\n")
    with (out / "register.csv").open("w", newline="") as handle:
        fields = COLUMNS + ["unknown_fields", "details"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})
    history_fields = COLUMNS + ["unknown_fields"]
    with (out / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key))
                             for key in history_fields})
    campaign_fields = ["campaign_id", "family", "root", "row_count",
                       "workflow_job_ids", "report_links"]
    for filename in ("campaign_index.csv", "campaigns.csv"):
        with (out / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=campaign_fields)
            writer.writeheader()
            for record in campaigns:
                writer.writerow({key: csv_value(record.get(key))
                                 for key in campaign_fields})
    (out / "campaigns.json").write_text(json.dumps(
        campaigns, sort_keys=True, indent=2, allow_nan=False,
    ) + "\n")

    counts = Counter((row["result_family"], row["stage"]) for row in rows)
    overview = [
        "# Research experiment register",
        "",
        f"Source snapshot: `{raw_name}` (`{raw_sha}`), captured {register.snapshot_time}.",
        "",
        f"This build contains **{len(rows)} normalized artifact/stage rows** across **{len(campaigns)} source groups/campaigns**. These groups include scheduler and log-only collections and are not 26 independent experiments. `unknown` means the compact source snapshot did not retain that field; no value is inferred from another metric.",
        "",
        "## Campaign index",
        "",
        markdown_table(
            ["campaign_id", "family", "root", "row_count", "workflow_job_ids", "report_links"],
            campaigns_markdown,
        ),
        "",
        "## Row coverage by family and stage",
        "",
        markdown_table(
            ["result_family", "stage", "rows"],
            [{"result_family": family, "stage": stage, "rows": count}
             for (family, stage), count in sorted(counts.items())],
        ),
        "",
        "## Interpretation rules",
        "",
        "- `recorded_lp_objective` is the exact source objective and `lp_objective_kind` records whether it is fleet-only or bus-plus-charging. `weighted_lp_objective` is populated only for bus-plus-charging LPs. `fractional_fleet` is the separately stored sum of route weights. The register never divides an objective by 100,000 to manufacture a fleet value.",
        "- `fleet_lower_bound_scope` distinguishes a simultaneous-overlap bound, a restricted-pool LP/MIP bound, and a full-model statement. Missing scope remains unknown.",
        "- Rows marked `superseded_duplicate_772031` or `rejected_observational` are retained for provenance and are not authoritative results.",
        "- Physical route replay, duplicate passenger coverage, and cross-route charger capacity are separate fields.",
        "",
    ]
    (out / "REGISTER.md").write_text("\n".join(overview))

    result_headers = [
        "campaign_id", "case_id", "stage", "substage", "arm",
        "authority_role", "workflow_state", "recorded_lp_objective",
        "lp_objective_kind", "weighted_lp_objective", "fractional_fleet",
        "full_model_lp_certified", "fleet_lower_bound",
        "fleet_lower_bound_scope", "mip_incumbent_fleet", "mip_bound_fleet",
        "charging_cost_exact", "charging_cost_lower", "charging_cost_upper",
        "charging_cost_grid", "charging_cost_continuous",
        "mip_status", "mip_gap", "stop_reason", "runtime_s", "pool_size",
        "physical_selected_validated", "proof_scope", "source_path",
    ]
    (out / "RESULTS.md").write_text(
        "# Normalized result and workflow rows\n\n"
        + markdown_table(result_headers, rows) + "\n"
    )
    validation = {
        "schema": "evsp-dr-research-register-validation-v1",
        "source_snapshot_sha256": raw_sha,
        "campaign_count": len(campaigns), "row_count": len(rows),
        "unique_row_ids": len({row["row_id"] for row in rows}),
        "rows_with_source_sha256": sum(row["source_sha256"] is not None for row in rows),
        "rows_without_source_sha256": sum(row["source_sha256"] is None for row in rows),
        "superseded_772031_rows": sum(
            row["authority_role"] == "superseded_duplicate_772031" for row in rows
        ),
        "authoritative_772009_rows": sum(
            row["authority_role"] == "authoritative_772009_recovery" for row in rows
        ),
        "rejected_rows": sum(row["artifact_status"].startswith("rejected")
                             for row in rows),
        "families_and_stages": {
            f"{family}:{stage}": count
            for (family, stage), count in sorted(counts.items())
        },
        "checks": [
            "all stable row IDs unique",
            "all retained SHA-256 fields are 64 lowercase hexadecimal characters",
            "all collector campaign families represented",
            "exactly one superseded duplicate-772031 row retained",
            "exactly one authoritative recovered-772009 row retained",
            "same-pool LP objectives labeled fleet-only and excluded from weighted_lp_objective",
            "tariff comparisons expanded into original, fixed-duty, and joint arm rows",
            "non-finite numeric values normalized to null/unknown",
        ],
    }
    (out / "validation.json").write_text(json.dumps(
        validation, sort_keys=True, indent=2, allow_nan=False,
    ) + "\n")
    return payload, validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out-dir", "--output-dir", dest="output_dir",
                        type=Path, required=True)
    parser.add_argument("--reports-root", type=Path)
    parser.add_argument(
        "--supplement", action="append", type=Path, default=[],
        help="Optional standalone JSON audit to preserve and index.",
    )
    args = parser.parse_args()
    snapshot_path = args.snapshot.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    raw = snapshot_path.read_bytes()
    snapshot = json.loads(raw)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_target = raw_dir / snapshot_path.name
    if raw_target.exists() and raw_target.read_bytes() != raw:
        raise FileExistsError(f"different raw snapshot already exists: {raw_target}")
    if not raw_target.exists():
        shutil.copyfile(snapshot_path, raw_target)
    markdown_files = read_markdown(
        args.reports_root.expanduser().resolve() if args.reports_root else None,
        exclude_root=output_dir,
    )
    register = Register(snapshot, snapshot_path, output_dir, markdown_files)
    register.build()
    supplemental_sources = []
    supplemental_dir = output_dir / "raw" / "supplemental"
    for supplied in args.supplement:
        supplied = supplied.expanduser().resolve()
        supplied_raw = supplied.read_bytes()
        supplied_value = json.loads(supplied_raw)
        supplemental_dir.mkdir(parents=True, exist_ok=True)
        target = supplemental_dir / supplied.name
        if target.exists() and target.read_bytes() != supplied_raw:
            raise FileExistsError(f"different supplement already exists: {target}")
        if not target.exists():
            shutil.copyfile(supplied, target)
        digest = sha_bytes(supplied_raw)
        supplemental_sources.append({
            "path": f"raw/supplemental/{supplied.name}", "sha256": digest,
        })
        campaign_id = f"supplemental:{supplied.stem}"
        register.add(
            campaign_id, None, "supplemental_audit", "audit_result",
            supplied_value if isinstance(supplied_value, dict)
            else {"data": supplied_value},
            source_path=str(supplied), source_sha256=digest,
            case_id=supplied.stem,
        )
    register.validate()
    payload, validation = write_outputs(
        register, snapshot_path.name, sha_bytes(raw), supplemental_sources,
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "campaign_count": payload["campaign_count"],
        "row_count": payload["row_count"],
        "source_snapshot_sha256": validation["source_snapshot_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
