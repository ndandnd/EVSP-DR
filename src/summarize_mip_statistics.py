#!/usr/bin/env python3
"""Build deterministic CSV/plot summaries for a MIP-statistics campaign."""

from __future__ import annotations

import argparse
import ctypes
import csv
import errno
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from config import BUS_COST_KX


CHECKPOINT_FIELDS = (
    "campaign", "cell_id", "scale", "replicate", "arm",
    "augmentation_changes_column_set", "cg_age_hours",
    "budget_hours", "checkpoint_elapsed_s", "observed_total_elapsed_s",
    "latest_statistics_observed_s",
    "stage", "incumbent_state", "incumbent_fleet",
    "incumbent_objective", "fleet_bound", "objective_bound",
    "fleet_gap", "node_count", "solution_count",
    "first_feasible_incumbent_s", "route_vector_sha256",
    "solver_ended_before_checkpoint",
)
FINAL_FIELDS = (
    "campaign", "cell_id", "scale", "replicate", "arm",
    "augmentation_changes_column_set", "cg_age_hours", "budget_hours",
    "output_exists", "incumbent_found", "status_name", "buses",
    "mip_obj", "mip_bound", "mip_gap", "fleet_bound", "fleet_proven",
    "runtime_s", "optimal_scope", "route_space_scope",
    "giro_target_buses", "time_to_le_giro_target_s",
    "time_below_giro_target_s", "time_to_finite_pool_proof_s",
    "source_result_sha256", "source_journal_sha256",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _validate_result(result: dict, job: dict, manifest: dict) -> None:
    source = job["source"]
    arm = job["arm"]
    if result.get("partitioning") is not True:
        raise ValueError(f"{job['cell_id']} is covering, not an integer schedule")
    expected_arm = "D" if arm == "GIRO" else "B"
    if result.get("experiment_arm") != expected_arm:
        raise ValueError(f"{job['cell_id']} experiment arm mismatch")
    if (
        result.get("source_result_sha256") != source["status_sha256"]
        or result.get("source_journal_sha256") != source["journal_sha256"]
    ):
        raise ValueError(f"{job['cell_id']} result source mismatch")
    provenance = result.get("mip_provenance")
    arguments = provenance.get("arguments") if isinstance(
        provenance, dict
    ) else None
    if (
        not isinstance(arguments, dict)
        or arguments.get("two_stage") is not True
        or arguments.get("cover") is not False
        or int(arguments.get("threads", -1)) != 8
        or provenance.get("observed_git_commit")
        != manifest["checkout_identity"]["expected_commit"]
        or provenance.get("expected_git_commit")
        != manifest["checkout_identity"]["expected_commit"]
    ):
        raise ValueError(f"{job['cell_id']} solver provenance mismatch")
    start = result.get("mip_start") or {}
    if arm == "GIRO":
        if (
            start.get("kind") != "validated_exact_partition"
            or start.get("source_sha256")
            != job["validated_start"]["sha256"]
        ):
            raise ValueError(f"{job['cell_id']} GIRO start mismatch")
    elif start.get("kind") == "validated_exact_partition":
        raise ValueError(f"{job['cell_id']} RAW result used GIRO columns")
    incumbent = result.get("incumbent_found") is True
    buses = result.get("buses")
    if incumbent != (isinstance(buses, int) and buses > 0):
        raise ValueError(f"{job['cell_id']} incumbent/bus mismatch")
    selected = result.get("selected_routes") or []
    if incumbent:
        status_path = Path(job["execution"]["status"])
        if (
            not status_path.is_file()
            or _sha(status_path)
            != job["execution"]["status_sha256"]
        ):
            raise ValueError(f"{job['cell_id']} staged status hash mismatch")
        source_status = json.loads(status_path.read_text())
        trip_ids = source_status.get("trip_ids")
        counts = defaultdict(int)
        for route in selected:
            for trip in route.get("trips") or []:
                counts[trip] += 1
        if (
            len(selected) != buses
            or not isinstance(trip_ids, list)
            or set(counts) != set(trip_ids)
            or any(counts[trip] != 1 for trip in trip_ids)
        ):
            raise ValueError(
                f"{job['cell_id']} selected routes are not an exact partition"
            )
    if result.get("fleet_proven") is True and (
        not incumbent
        or result.get("optimal_scope") not in {
            "fleet_only", "full_pool_lexicographic"
        }
        or _float(result.get("fleet_bound")) is None
        or math.ceil(float(result["fleet_bound"]) - 1e-6) < buses
    ):
        raise ValueError(f"{job['cell_id']} finite-pool proof is inconsistent")
    if (
        result.get("optimal_scope")
        in {"fleet_only", "full_pool_lexicographic"}
    ) != (result.get("fleet_proven") is True):
        raise ValueError(
            f"{job['cell_id']} proof flag/scope are inconsistent"
        )
    if result.get("optimal_scope") not in {
        "none", "fleet_only", "full_pool_lexicographic"
    }:
        raise ValueError(f"{job['cell_id']} has invalid two-stage scope")
    if result.get("optimal_scope") == "full_pool_lexicographic":
        detail = result.get("two_stage")
        mip_obj = _float(result.get("mip_obj"))
        mip_bound = _float(result.get("mip_bound"))
        stage_obj = _float(
            detail.get("stage2_variable_obj")
            if isinstance(detail, dict) else None
        )
        stage_bound = _float(
            detail.get("stage2_variable_bound")
            if isinstance(detail, dict) else None
        )
        stage_gap = _float(
            detail.get("stage2_absolute_gap")
            if isinstance(detail, dict) else None
        )
        stage1_buses = (
            detail.get("stage1_buses")
            if isinstance(detail, dict) else None
        )
        if (
            result.get("status_name") != "OPTIMAL"
            or not isinstance(detail, dict)
            or detail.get("stage2_executed") is not True
            or detail.get("stage2_status") != 2
            or _float(result.get("mip_gap")) != 0.0
            or _float(result.get("absolute_cost_gap")) != 0.0
            or mip_obj is None
            or mip_bound is None
            or not math.isclose(
                mip_obj, mip_bound, rel_tol=1e-10, abs_tol=1e-6
            )
            or stage_obj is None
            or stage_bound is None
            or stage_gap != 0.0
            or not math.isclose(
                stage_obj, stage_bound, rel_tol=1e-10, abs_tol=1e-6
            )
            or not isinstance(stage1_buses, int)
            or stage1_buses != buses
            or not math.isclose(
                mip_obj,
                BUS_COST_KX * stage1_buses + stage_obj,
                rel_tol=1e-10,
                abs_tol=1e-6,
            )
            or not math.isclose(
                mip_bound,
                BUS_COST_KX * stage1_buses + stage_bound,
                rel_tol=1e-10,
                abs_tol=1e-6,
            )
            or not math.isclose(
                mip_obj - mip_bound,
                float(result["absolute_cost_gap"]),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise ValueError(
                f"{job['cell_id']} lacks optimal cost-stage closure"
            )


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace publication unavailable")
    renameat2.argtypes = [
        ctypes.c_int, ctypes.c_char_p,
        ctypes.c_int, ctypes.c_char_p, ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100, os.fsencode(source), -100, os.fsencode(destination), 1
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(f"output directory exists: {destination}")
    raise OSError(error, os.strerror(error), str(destination))


def _load_campaign(root: Path) -> tuple[dict, list[dict], list[dict]]:
    manifest = json.loads((root / "campaign.json").read_text())
    plan_path = root / "approved-plan.json"
    if not plan_path.is_file():
        raise ValueError("campaign lacks immutable approved-plan.json")
    plan_raw = plan_path.read_bytes()
    plan_sha = hashlib.sha256(plan_raw).hexdigest()
    if plan_sha != manifest.get("approval_sha256"):
        raise ValueError("campaign approval SHA mismatch")
    approved = json.loads(plan_raw)
    approved_jobs = {
        job["cell_id"]: job for job in approved.get("jobs") or []
    }
    manifest_jobs = {
        job["cell_id"]: job for job in manifest.get("jobs") or []
    }
    if (
        len(manifest.get("jobs") or []) != len(manifest_jobs)
        or len(approved.get("jobs") or []) != len(approved_jobs)
        or len(manifest_jobs) != len(approved_jobs)
        or set(manifest_jobs) != set(approved_jobs)
    ):
        raise ValueError("campaign job set differs from approved plan")
    for key in (
        "campaign", "mode", "checkout_identity", "worker_sha256",
        "runner_sha256", "code_hashes", "python_identity",
        "environment_whitelist",
    ):
        if manifest.get(key) != approved.get(key):
            raise ValueError(f"campaign {key} differs from approved plan")
    campaign = manifest["campaign"]
    checkpoint_rows = []
    final_rows = []
    targets = {}
    for job in manifest.get("jobs") or []:
        if job["arm"] != "GIRO":
            continue
        target = (
            (job.get("validated_start") or {}).get("route_count")
        )
        if target is not None:
            targets[(
                job["scale"], job["replicate"],
                job["source"]["status_sha256"],
            )] = int(target)
    for job in sorted(
            manifest.get("jobs") or [], key=lambda item: item["cell_id"]):
        approved_job = approved_jobs.get(job["cell_id"])
        mutable_fields = {
            "job_id", "submission_state", "submission_error",
            "deduplication_comment",
        }
        if approved_job is None or any(
            job.get(key) != value
            for key, value in approved_job.items()
            if key not in mutable_fields
        ):
            raise ValueError(f"{job['cell_id']} differs from approved plan")
        progress_dir = Path(job["progress_dir"])
        improvements = []
        if progress_dir.is_dir():
            for checkpoint_path in sorted(
                    progress_dir.glob("checkpoint_*.json")):
                payload = json.loads(checkpoint_path.read_text())
                if payload.get("schema") != "evsp-dr-mip-convergence-v1":
                    raise ValueError(
                        f"invalid convergence schema: {checkpoint_path}"
                    )
                metadata = payload.get("metadata") or {}
                source = job["source"]
                if (
                    metadata.get("source_result_sha256")
                    != source["status_sha256"]
                    or metadata.get("source_journal_sha256")
                    != source["journal_sha256"]
                ):
                    raise ValueError(
                        f"checkpoint source mismatch: {checkpoint_path}"
                    )
                expected_arm = "D" if job["arm"] == "GIRO" else "B"
                parameters = metadata.get("parameters")
                if (
                    metadata.get("experiment_arm") != expected_arm
                    or metadata.get("git_commit")
                    != manifest["checkout_identity"]["expected_commit"]
                    or not isinstance(parameters, dict)
                    or parameters.get("two_stage") is not True
                    or parameters.get("cover") is not False
                    or int(parameters.get("threads", -1)) != 8
                    or (
                        job["arm"] == "GIRO"
                        and metadata.get(
                            "source_initial_partition_sha256"
                        ) != job["validated_start"]["sha256"]
                    )
                    or (
                        job["arm"] == "RAW"
                        and metadata.get(
                            "source_initial_partition_sha256"
                        ) is not None
                    )
                ):
                    raise ValueError(
                        f"checkpoint treatment/provenance mismatch: "
                        f"{checkpoint_path}"
                    )
                incumbent = payload.get("incumbent") or {}
                stats = payload.get("latest_statistics") or {}
                checkpoint_rows.append({
                    "campaign": campaign,
                    "cell_id": job["cell_id"],
                    "scale": job["scale"],
                    "replicate": job["replicate"],
                    "arm": job["arm"],
                    "augmentation_changes_column_set": (
                        job["augmentation_changes_column_set"]
                    ),
                    "cg_age_hours": job.get("age_hours"),
                    "budget_hours": job["budget_hours"],
                    "checkpoint_elapsed_s": payload.get(
                        "checkpoint_elapsed_s"
                    ),
                    "observed_total_elapsed_s": payload.get(
                        "observed_total_elapsed_s"
                    ),
                    "latest_statistics_observed_s": payload.get(
                        "latest_statistics_observed_s"
                    ),
                    "stage": payload.get("stage"),
                    "incumbent_state": payload.get("incumbent_state"),
                    "incumbent_fleet": incumbent.get("fleet"),
                    "incumbent_objective": incumbent.get("objective"),
                    "fleet_bound": stats.get("fleet_bound"),
                    "objective_bound": stats.get("objective_bound"),
                    "fleet_gap": stats.get("fleet_gap"),
                    "node_count": stats.get("node_count"),
                    "solution_count": stats.get("solution_count"),
                    "first_feasible_incumbent_s": payload.get(
                        "first_feasible_incumbent_s"
                    ),
                    "route_vector_sha256": incumbent.get(
                        "route_vector_sha256"
                    ),
                    "solver_ended_before_checkpoint": payload.get(
                        "solver_ended_before_checkpoint"
                    ),
                })
                improvements = payload.get("incumbent_improvements") or []
        output = Path(job["output"])
        result = json.loads(output.read_text()) if output.is_file() else {}
        if result:
            _validate_result(result, job, manifest)
            progress_final = progress_dir / "final.json"
            if not progress_final.is_file():
                raise ValueError(f"{job['cell_id']} lacks progress final.json")
            progress_payload = json.loads(progress_final.read_text())
            final = progress_payload.get("final") or {}
            if (
                final.get("incumbent_found")
                != result.get("incumbent_found")
                or final.get("buses") != result.get("buses")
                or final.get("fleet_proven") != result.get("fleet_proven")
            ):
                raise ValueError(
                    f"{job['cell_id']} final result/progress mismatch"
                )
        key = (
            job["scale"], job["replicate"],
            job["source"]["status_sha256"],
        )
        target = targets.get(key)
        at_or_below = [
            event["total_elapsed_s"] for event in improvements
            if target is not None and event.get("fleet") is not None
            and int(event["fleet"]) <= target
        ]
        below = [
            event["total_elapsed_s"] for event in improvements
            if target is not None and event.get("fleet") is not None
            and int(event["fleet"]) < target
        ]
        proof_times = []
        for row in checkpoint_rows:
            if row["cell_id"] != job["cell_id"]:
                continue
            fleet = _float(row["incumbent_fleet"])
            bound = _float(row["fleet_bound"])
            if (
                fleet is not None and bound is not None
                and math.ceil(bound - 1e-6) >= int(round(fleet))
                and not row["solver_ended_before_checkpoint"]
            ):
                observed = row["latest_statistics_observed_s"]
                if observed is None:
                    raise ValueError(
                        f"{job['cell_id']} proving checkpoint lacks "
                        "statistics observation time"
                    )
                if (
                    observed < 0
                    or observed > row["observed_total_elapsed_s"] + 1e-9
                    or observed > row["checkpoint_elapsed_s"] + 1e-9
                ):
                    raise ValueError(
                        f"{job['cell_id']} proving checkpoint timestamp "
                        "is inconsistent"
                    )
                proof_times.append(
                    observed
                )
        final_rows.append({
            "campaign": campaign,
            "cell_id": job["cell_id"],
            "scale": job["scale"],
            "replicate": job["replicate"],
            "arm": job["arm"],
            "augmentation_changes_column_set": (
                job["augmentation_changes_column_set"]
            ),
            "cg_age_hours": job.get("age_hours"),
            "budget_hours": job["budget_hours"],
            "output_exists": output.is_file(),
            "incumbent_found": result.get("incumbent_found"),
            "status_name": result.get("status_name"),
            "buses": result.get("buses"),
            "mip_obj": result.get("mip_obj"),
            "mip_bound": result.get("mip_bound"),
            "mip_gap": result.get("mip_gap"),
            "fleet_bound": result.get("fleet_bound"),
            "fleet_proven": result.get("fleet_proven"),
            "runtime_s": result.get("runtime_s"),
            "optimal_scope": result.get("optimal_scope"),
            "route_space_scope": (
                "finite_augmented_pool" if job["arm"] == "GIRO"
                else "finite_raw_cg_pool"
            ),
            "giro_target_buses": target,
            "time_to_le_giro_target_s": min(at_or_below)
            if at_or_below else None,
            "time_below_giro_target_s": min(below) if below else None,
            "time_to_finite_pool_proof_s": min(proof_times)
            if proof_times else (
                result.get("runtime_s")
                if result.get("fleet_proven") is True else None
            ),
            "source_result_sha256": result.get("source_result_sha256"),
            "source_journal_sha256": result.get("source_journal_sha256"),
        })
    return manifest, checkpoint_rows, final_rows


def _write_csv(path: Path, fields, rows) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _plot(staging: Path, checkpoint_rows, final_rows) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metadata = {
        "Creator": "EVSP-DR deterministic convergence summarizer",
        "CreationDate": None,
        "ModDate": None,
    }

    def save(fig, stem):
        fig.tight_layout()
        fig.savefig(
            staging / f"{stem}.png", dpi=160,
            metadata={"Software": "EVSP-DR"},
        )
        fig.savefig(staging / f"{stem}.pdf", metadata=metadata)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    grouped = defaultdict(list)
    for row in checkpoint_rows:
        if (
            row["incumbent_fleet"] is not None
            and not row["solver_ended_before_checkpoint"]
        ):
            grouped[row["cell_id"]].append(row)
    for cell, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["checkpoint_elapsed_s"])
        arm = rows[0]["arm"]
        ax.step(
            [row["checkpoint_elapsed_s"] / 3600 for row in rows],
            [row["incumbent_fleet"] for row in rows],
            where="post", label=f"{cell} ({arm})",
        )
    ax.set(xlabel="MIP elapsed hours", ylabel="Incumbent buses",
           title="Finite-pool incumbent buses vs MIP time")
    if grouped:
        ax.legend(fontsize=6, ncol=2)
    else:
        ax.text(0.5, 0.5, "No available MIP checkpoints",
                transform=ax.transAxes, ha="center")
    save(fig, "buses_vs_mip_time")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for cell, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["checkpoint_elapsed_s"])
        x = [row["checkpoint_elapsed_s"] / 3600 for row in rows]
        ax.step(
            x, [row["incumbent_fleet"] for row in rows],
            where="post", label=f"{cell} incumbent",
        )
        bounds = [row["fleet_bound"] for row in rows]
        if any(value is not None for value in bounds):
            ax.step(x, bounds, where="post", linestyle="--",
                    label=f"{cell} fleet bound")
    ax.set(xlabel="MIP elapsed hours", ylabel="Buses / fleet bound",
           title="Incumbent and finite-pool fleet-bound curves")
    if grouped:
        ax.legend(fontsize=5, ncol=2)
    else:
        ax.text(0.5, 0.5, "No available MIP checkpoints",
                transform=ax.transAxes, ha="center")
    save(fig, "incumbent_fleet_bound_curves")

    scales = sorted({
        int(row["scale"]) for row in final_rows
        if row["buses"] is not None and row["cg_age_hours"] is not None
    })
    ages = sorted({
        float(row["cg_age_hours"]) for row in final_rows
        if row["buses"] is not None and row["cg_age_hours"] is not None
    })
    matrix = np.full((len(scales), len(ages)), np.nan)
    for i, scale in enumerate(scales):
        for j, age in enumerate(ages):
            values = [
                float(row["buses"]) for row in final_rows
                if int(row["scale"]) == scale
                and float(row["cg_age_hours"]) == age
                and row["arm"] == "GIRO"
                and row["buses"] is not None
            ]
            if values:
                matrix[i, j] = sum(values) / len(values)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if matrix.size and np.isfinite(matrix).any():
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        fig.colorbar(image, ax=ax, label="Final buses (finite pool)")
        ax.set_xticks(range(len(ages)), [f"{age:g}" for age in ages])
        ax.set_yticks(range(len(scales)), [str(scale) for scale in scales])
    else:
        ax.text(0.5, 0.5, "No verified CG-age MIP results",
                transform=ax.transAxes, ha="center")
    ax.set(xlabel="CG age (hours)", ylabel="Scale",
           title="GIRO-augmented CG age × final MIP buses")
    save(fig, "cg_age_final_buses_heatmap")


def summarize(
    campaign_root: Path,
    output_dir: Path,
    *,
    replace_output=False,
) -> dict:
    root = campaign_root.expanduser().resolve()
    output = output_dir.expanduser().resolve()
    if replace_output:
        raise ValueError(
            "MIP statistics summaries are immutable; choose a new output path"
        )
    if output.exists():
        raise FileExistsError(f"output directory exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest, checkpoints, finals = _load_campaign(root)
    staging = Path(tempfile.mkdtemp(
        dir=output.parent, prefix=f".{output.name}.tmp."
    ))
    try:
        _write_csv(
            staging / "checkpoint_long.csv",
            CHECKPOINT_FIELDS,
            sorted(checkpoints, key=lambda row: (
                row["cell_id"], row["checkpoint_elapsed_s"]
            )),
        )
        _write_csv(
            staging / "job_final.csv",
            FINAL_FIELDS,
            sorted(finals, key=lambda row: row["cell_id"]),
        )
        _plot(staging, checkpoints, finals)
        notes = {
            "schema": "evsp-dr-mip-statistics-summary-v1",
            "campaign": manifest["campaign"],
            "jobs": len(finals),
            "checkpoint_rows": len(checkpoints),
            "interpretation": [
                "All proof labels are limited to each finite input pool.",
                "RAW and GIRO are separate feasible column sets; GIRO is not "
                "only a warm-start treatment.",
                "No covering LP value is represented as an integer schedule.",
                "Convergence JSON files are observations, not Gurobi tree "
                "restart checkpoints.",
            ],
        }
        (staging / "summary.json").write_text(
            json.dumps(notes, indent=2) + "\n"
        )
        _rename_noreplace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "output_dir": str(output),
        "jobs": len(finals),
        "checkpoint_rows": len(checkpoints),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--replace-output", action="store_true")
    args = parser.parse_args(argv)
    result = summarize(
        args.campaign_root,
        args.out_dir,
        replace_output=args.replace_output,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
