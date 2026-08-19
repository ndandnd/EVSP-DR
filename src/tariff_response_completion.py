"""Shared fail-closed identity checks for tariff worker completions."""

from __future__ import annotations


SCHEMA = "evsp-dr-tariff-response-worker-completion-v2"


def validate_completion_identity(completion, job, plan_sha):
    expected = {
        "schema": SCHEMA,
        "job_key": job["job_key"],
        "execution_digest": job["execution_digest"],
        "phase": job["phase"],
        "treatment": job["treatment"],
        "analysis_role": job["analysis_role"],
        "scale": job["scale"],
        "tariff_id": job["tariff_id"],
        "plan_sha256": plan_sha,
        "instance_sha256": job["instance"]["sha256"],
        "tariff_sha256": job.get("tariff_sha256"),
    }
    errors = {
        field: {
            "expected": value,
            "observed": completion.get(field),
        }
        for field, value in expected.items()
        if completion.get(field) != value
    }
    slurm_job_id = str(completion.get("slurm_job_id") or "")
    if not slurm_job_id.isdigit():
        errors["slurm_job_id"] = {
            "expected": "numeric Slurm job ID",
            "observed": completion.get("slurm_job_id"),
        }
    hashes = completion.get("artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        errors["artifact_sha256"] = {
            "expected": "nonempty artifact hash map",
            "observed": type(hashes).__name__,
        }
    if errors:
        raise ValueError(
            f"worker completion identity mismatch for "
            f"{job['job_key']}: {errors}"
        )
    return hashes
