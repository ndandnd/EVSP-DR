#!/usr/bin/env python3
"""Normalize the event-versus-uniform envelope experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path


SCHEMA = "evsp-dr-event-uniform-envelope-summary-v1"


def _load(path: Path, manifest: dict) -> dict:
    payload = path.read_bytes()
    manifest[str(path)] = {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    return json.loads(payload)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _representation(plan, representation_id):
    return next(
        row for row in plan["representations"]
        if row["representation_id"] == representation_id
    )


def _uniform_source(root, panel, cell, representation):
    panel_a = (
        root / "panel_a/uniform" / cell / representation / "cg.json"
    )
    if panel == "A" and panel_a.is_file():
        return panel_a
    if panel == "B":
        snapshot = (
            root / "panel_b/snapshots" / cell
            / representation / "cg.json"
        )
        if snapshot.is_file():
            return snapshot
    return (
        root / "panel_b/uniform" / cell
        / representation / "cg.json"
    )


def _panel_a_mip(root, cell, representation, time_model):
    if time_model == "event":
        return root / "panel_b/mip_native/event" / cell / "mip.json"
    candidate = (
        root / "panel_a/mip_native/uniform"
        / cell / representation / "mip.json"
    )
    return candidate if candidate.is_file() else (
        root / "panel_b/mip_native/uniform"
        / cell / representation / "mip.json"
    )


def _target_results(folder: Path, manifest: dict):
    rows = []
    if not folder.is_dir():
        return rows
    for path in sorted(folder.glob("target*.json")):
        payload = _load(path, manifest)
        rows.append({
            "path": str(path),
            "target": int(payload["target_fleet"]),
            "outcome": payload["outcome"],
            "physical": (
                payload["outcome"] != "FEASIBLE"
                or payload.get("witness_route_count")
                == payload["target_fleet"]
            ),
        })
    return rows


def _model_proof(root, cell, representation, time_model, manifest):
    if time_model == "event":
        path = (
            root / "panel_a/model_witness/event"
            / cell / "witness.json"
        )
        payload = _load(path, manifest)
        return (
            payload["model_integer_optimum"],
            True,
            "industrial_event_partition_sandwich",
            str(path),
        )
    witness_path = (
        root / "panel_a/model_witness/uniform"
        / cell / representation / "witness-v2.json"
    )
    witness = _load(witness_path, manifest)
    if witness["model_optimum_proven_by_sandwich"]:
        return (
            witness["model_integer_optimum"],
            True,
            "industrial_uniform_partition_sandwich",
            str(witness_path),
        )
    arcflow_path = (
        root / "panel_a/arcflow" / cell
        / representation / "oracle.json"
    )
    if arcflow_path.is_file():
        arcflow = _load(arcflow_path, manifest)
        if arcflow["fleet_proof"]["proven"]:
            return (
                int(round(arcflow["fleet_proof"]["integral_witness"])),
                True,
                "arcflow_integral_witness_sandwich",
                str(arcflow_path),
            )
    return None, False, "censored_after_sandwich_and_arcflow", None


def summarize(plan_path: Path, execution_root: Path, output_dir: Path):
    manifest = {}
    plan = _load(plan_path.resolve(strict=True), manifest)
    if execution_root.resolve() != Path(plan["execution_root"]).resolve():
        raise ValueError("execution root differs from immutable plan")
    root = execution_root.resolve(strict=True)
    panel_a = []
    panel_b = []
    representations = [
        row["representation_id"] for row in plan["representations"]
    ]
    for instance in plan["instances"]:
        cell = instance["cell_id"]
        target_fleet = int(instance["target_fleet"])
        for representation_id in representations:
            representation = _representation(plan, representation_id)
            time_model = representation["time_model"]
            if time_model == "event":
                cg_path = root / "panel_a/event" / cell / "cg.json"
                phase2_path = (
                    root / "panel_a/event"
                    / cell / "fleet_phase2.json"
                )
            else:
                cg_path = _uniform_source(
                    root, "A", cell, representation_id
                )
                phase2_path = (
                    root / "panel_a/uniform" / cell
                    / representation_id / "fleet_phase2.json"
                )
            cg = _load(cg_path, manifest)
            phase2 = _load(phase2_path, manifest)
            mip_path = _panel_a_mip(
                root, cell, representation_id, time_model
            )
            mip = _load(mip_path, manifest)
            lp = float(phase2["fleet_lp_lower_bound"])
            integer_lower = math.ceil(lp - 1e-6)
            model_value, model_proven, model_method, model_evidence = (
                _model_proof(
                    root, cell, representation_id, time_model, manifest
                )
            )
            target_folder = (
                root / "panel_a/target/event" / cell
                if time_model == "event"
                else root / "panel_a/target/uniform"
                / cell / representation_id
            )
            targets = _target_results(target_folder, manifest)
            pool_lower = integer_lower
            pool_upper = (
                int(mip["buses"]) if mip.get("buses") is not None else None
            )
            for target in targets:
                if target["outcome"] == "INFEASIBLE":
                    pool_lower = max(pool_lower, target["target"] + 1)
                elif target["outcome"] == "FEASIBLE":
                    pool_upper = (
                        target["target"] if pool_upper is None
                        else min(pool_upper, target["target"])
                    )
            if mip.get("fleet_proven"):
                pool_lower = pool_upper = int(mip["buses"])
            pool_proven = (
                pool_upper is not None and pool_lower == pool_upper
            )
            pool_value = pool_lower if pool_proven else None
            model_upper = (
                int(model_value) if model_proven
                else pool_upper
            )
            if (
                not model_proven
                and model_upper is not None
                and int(model_upper) == integer_lower
            ):
                model_value = integer_lower
                model_proven = True
                model_method = "finite_pool_witness_sandwich"
                model_evidence = (
                    str(mip_path)
                    if mip.get("buses") == integer_lower
                    else next(
                        (
                            target["path"] for target in targets
                            if target["outcome"] == "FEASIBLE"
                            and target["target"] == integer_lower
                        ),
                        None,
                    )
                )
                model_upper = integer_lower
            panel_a.append({
                "cell_id": cell,
                "target_fleet": target_fleet,
                "representation_id": representation_id,
                "time_model": time_model,
                "soc_step": representation["soc_step"],
                "block_min": representation["block_min"],
                "L_model": lp,
                "L_model_certified":
                    phase2["certificate"]["certified"],
                "I_model": model_value,
                "I_model_lower": integer_lower,
                "I_model_upper": model_upper,
                "I_model_proven": model_proven,
                "I_model_method": model_method,
                "I_pool": pool_value,
                "I_pool_lower": pool_lower,
                "I_pool_upper": pool_upper,
                "I_pool_proven": pool_proven,
                "I_timed": mip.get("buses"),
                "timed_fleet_proven": mip.get("fleet_proven"),
                "representation_gap": (
                    model_value - target_fleet
                    if model_proven else None
                ),
                "lp_integrality_gap": (
                    model_value - lp if model_proven else None
                ),
                "pool_composition_gap": (
                    pool_value - model_value
                    if pool_proven and model_proven else None
                ),
                "mip_search_gap": (
                    int(mip["buses"]) - pool_value
                    if pool_proven and mip.get("buses") is not None
                    else None
                ),
                "pool_columns": mip["pool_columns"],
                "source_cg_certified":
                    cg.get("certified_rc_optimal"),
                "source_cg_stop_reason": cg.get("stop_reason"),
                "source_cg_iterations": cg.get("iterations"),
                "source_cg_wall_s": cg.get("wall_s"),
                "source_cg_peak_rss_mb": cg.get("peak_rss_mb"),
                "optimality_scope": mip.get("optimality_scope"),
                "physical_witness_valid":
                    mip.get("physical_witness_valid"),
                "cg_status_sha256":
                    manifest[str(cg_path)]["sha256"],
                "phase2_sha256":
                    manifest[str(phase2_path)]["sha256"],
                "mip_sha256": manifest[str(mip_path)]["sha256"],
                "model_evidence": model_evidence,
            })

            if time_model == "event":
                panel_b_cg_path = cg_path
                target_paths = [
                    root / "panel_b/target/event" / cell
                    / "target-v2.json",
                    root / "panel_b/target/event" / cell
                    / "target.json",
                ]
                panel_b_mip_path = (
                    root / "panel_b/mip_native/event"
                    / cell / "mip.json"
                )
            else:
                panel_b_cg_path = _uniform_source(
                    root, "B", cell, representation_id
                )
                target_paths = [
                    root / "panel_b/target/uniform"
                    / cell / representation_id / "target.json"
                ]
                panel_b_mip_path = (
                    root / "panel_b/mip_native/uniform"
                    / cell / representation_id / "mip.json"
                )
            panel_b_cg = _load(panel_b_cg_path, manifest)
            panel_b_mip = _load(panel_b_mip_path, manifest)
            target_path = next(
                path for path in target_paths if path.is_file()
            )
            target_result = _load(target_path, manifest)
            panel_b.append({
                "cell_id": cell,
                "target_fleet": target_fleet,
                "representation_id": representation_id,
                "time_model": time_model,
                "soc_step": representation["soc_step"],
                "block_min": representation["block_min"],
                "I_timed": panel_b_mip.get("buses"),
                "reaches_industrial_fleet": (
                    panel_b_mip.get("buses") is not None
                    and int(panel_b_mip["buses"]) <= target_fleet
                ),
                "target_feasibility_outcome":
                    target_result["outcome"],
                "timed_fleet_proven":
                    panel_b_mip.get("fleet_proven"),
                "pool_columns": panel_b_mip["pool_columns"],
                "source_cg_certified":
                    panel_b_cg.get("certified_rc_optimal"),
                "source_cg_stop_reason":
                    panel_b_cg.get("stop_reason"),
                "source_cg_iterations":
                    panel_b_cg.get("iterations"),
                "source_cg_wall_s": panel_b_cg.get("wall_s"),
                "source_cg_peak_rss_mb":
                    panel_b_cg.get("peak_rss_mb"),
                "event_matched_budget_s": (
                    cg.get("wall_s") if time_model == "event"
                    else _load(
                        root / "panel_a/event" / cell / "cg.json",
                        manifest,
                    ).get("wall_s")
                ),
                "optimality_scope":
                    panel_b_mip.get("optimality_scope"),
                "physical_witness_valid":
                    panel_b_mip.get("physical_witness_valid"),
                "cg_status_sha256":
                    manifest[str(panel_b_cg_path)]["sha256"],
                "mip_sha256":
                    manifest[str(panel_b_mip_path)]["sha256"],
                "target_sha256":
                    manifest[str(target_path)]["sha256"],
            })
    envelope_rows = []
    for instance in plan["instances"]:
        cell = instance["cell_id"]
        event = next(
            row for row in panel_b
            if row["cell_id"] == cell and row["time_model"] == "event"
        )
        uniform = [
            row for row in panel_b
            if row["cell_id"] == cell and row["time_model"] == "uniform"
        ]
        best = min(
            row["I_timed"] for row in uniform
            if row["I_timed"] is not None
        )
        winners = [
            row["representation_id"] for row in uniform
            if row["I_timed"] == best
        ]
        comparison = (
            "event_better" if event["I_timed"] < best
            else "tie" if event["I_timed"] == best
            else "uniform_better"
        )
        envelope_rows.append({
            "cell_id": cell,
            "target_fleet": instance["target_fleet"],
            "event_I_timed": event["I_timed"],
            "event_reaches_target":
                event["reaches_industrial_fleet"],
            "uniform_envelope_I_timed": best,
            "uniform_envelope_reaches_target":
                best <= int(instance["target_fleet"]),
            "uniform_winning_representations": "|".join(winners),
            "comparison": comparison,
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_a_path = output_dir / "panel_a.csv"
    panel_b_path = output_dir / "panel_b.csv"
    envelope_path = output_dir / "panel_b_envelope.csv"
    _write_csv(panel_a_path, panel_a)
    _write_csv(panel_b_path, panel_b)
    _write_csv(envelope_path, envelope_rows)
    summary = {
        "schema": SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "panel_a_rows": len(panel_a),
        "panel_a_exact_model_rows": sum(
            bool(row["I_model_proven"]) for row in panel_a
        ),
        "panel_a_exact_pool_rows": sum(
            bool(row["I_pool_proven"]) for row in panel_a
        ),
        "panel_b_rows": len(panel_b),
        "event_target_count": sum(
            bool(row["event_reaches_target"]) for row in envelope_rows
        ),
        "uniform_envelope_target_count": sum(
            bool(row["uniform_envelope_reaches_target"])
            for row in envelope_rows
        ),
        "event_better_count": sum(
            row["comparison"] == "event_better"
            for row in envelope_rows
        ),
        "tie_count": sum(
            row["comparison"] == "tie" for row in envelope_rows
        ),
        "uniform_better_count": sum(
            row["comparison"] == "uniform_better"
            for row in envelope_rows
        ),
    }
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    summary_path.write_text(json.dumps(summary, indent=1) + "\n")
    manifest_rows = sorted(manifest.values(), key=lambda row: row["path"])
    _write_csv(output_dir / "evidence_manifest.csv", manifest_rows)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = summarize(
        args.plan, args.execution_root, args.output_dir
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
