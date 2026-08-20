"""Strict read-only validation shared by profile monitor/summarizer/archive."""

from __future__ import annotations

import math
import statistics

EXPECTED_LABELS = {"historical", "ca", "cs", "pa", "ps"}
EXPECTED_PREFIXES = {1000, 5000, 10000, 25000, 50000}
EXPECTED_METHODS = {"highs", "highs-ds", "highs-ipm"}
PROFILE_SCHEMA = "evsp-dr-frozen-pool-prefix-profile-v2"
CAMPAIGN_SCHEMA = "evsp-dr-exact-cg-profile-campaign-v1"


def _nonnegative_number(value) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= 0.0


def _close(left, right) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=1e-9, abs_tol=1e-6
        )
    except (OverflowError, TypeError, ValueError):
        return False


def validate_campaign_manifest(manifest) -> list[str]:
    errors = []
    if not isinstance(manifest, dict):
        return ["campaign manifest is not a JSON object"]
    if manifest.get("schema") != CAMPAIGN_SCHEMA:
        errors.append("unexpected campaign schema")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        return [*errors, "campaign jobs is not a list"]
    labels = []
    malformed_job = False
    for job in jobs:
        if not isinstance(job, dict):
            malformed_job = True
            continue
        label = job.get("label")
        if not isinstance(label, str):
            malformed_job = True
        else:
            labels.append(label)
        if (not isinstance(job.get("output"), str)
                or not isinstance(job.get("job_name"), str)
                or not isinstance(job.get("job_spec"), dict)):
            malformed_job = True
    if (malformed_job or len(jobs) != 5
            or len(labels) != len(jobs)
            or set(labels) != EXPECTED_LABELS):
        errors.append("campaign does not contain exactly historical/CA/CS/PA/PS")
    if len(labels) != len(set(labels)):
        errors.append("campaign contains duplicate labels")
    identity = manifest.get("checkout_identity")
    if not isinstance(identity, dict):
        errors.append("campaign checkout_identity is not an object")
        identity = {}
    expected_commit = identity.get("expected_commit")
    if not isinstance(expected_commit, str) or len(expected_commit) != 40:
        errors.append("campaign lacks expected commit")
    return errors


def validate_profile_payload(payload, job: dict, manifest: dict) -> list[str]:
    errors = []
    if not isinstance(payload, dict):
        return ["profile output is not a JSON object"]
    if payload.get("schema") != PROFILE_SCHEMA:
        errors.append("unexpected profile schema")
    if payload.get("source_unchanged") is not True:
        errors.append("profile does not attest unchanged sources")
    before = payload.get("source_hashes_before")
    after = payload.get("source_hashes_after")
    if not isinstance(before, dict) or after != before:
        errors.append("profile source hashes are missing or changed")
    if not isinstance(job, dict):
        return ["campaign job is not an object"]
    spec = job.get("job_spec")
    if not isinstance(spec, dict):
        errors.append("job spec is not an object")
        spec = {}
    expected_hashes = {
        "result": spec.get("staged_result_sha256"),
        "journal": spec.get("staged_journal_sha256"),
        "instance": spec.get("staged_instance_sha256"),
        "prices": spec.get("staged_prices_sha256"),
    }
    if isinstance(before, dict) and before != expected_hashes:
        errors.append("profile hashes differ from staged job spec")
    identity = manifest.get("checkout_identity")
    if not isinstance(identity, dict):
        errors.append("campaign checkout_identity is not an object")
        identity = {}
    expected_commit = identity.get("expected_commit")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("profile provenance is not an object")
        provenance = {}
    if provenance.get("git_commit") != expected_commit:
        errors.append("profile commit differs from campaign")
    if provenance.get("git_dirty") is not False:
        errors.append("profile worktree is dirty or unreported")
    expected_repeat = spec.get("repeat")
    if (not isinstance(expected_repeat, int)
            or isinstance(expected_repeat, bool)
            or expected_repeat <= 0):
        errors.append("job spec repeat must be a positive integer")
        expected_repeat = None
    if payload.get("repeat") != expected_repeat:
        errors.append("profile repeat count differs from job spec")
    expected_time_limit = spec.get("time_limit_s")
    if (not _nonnegative_number(expected_time_limit)
            or float(expected_time_limit) <= 0.0
            or not _nonnegative_number(payload.get("time_limit_s"))
            or payload.get("time_limit_s") != expected_time_limit):
        errors.append("profile time limit differs from job spec")
    requested_prefixes = payload.get("requested_prefixes")
    try:
        prefix_set = set(requested_prefixes) \
            if isinstance(requested_prefixes, list) else set()
    except TypeError:
        prefix_set = set()
    if prefix_set != EXPECTED_PREFIXES:
        errors.append("profile prefix set is incomplete")
    requested_methods = payload.get("methods")
    try:
        method_set = set(requested_methods) \
            if isinstance(requested_methods, list) else set()
    except TypeError:
        method_set = set()
    if method_set != EXPECTED_METHODS:
        errors.append("profile method set is incomplete")
    profiles = payload.get("profiles")
    if not isinstance(profiles, list):
        return [*errors, "profiles is not a list"]
    prefix_map = {}
    malformed_prefix = False
    for profile in profiles:
        if not isinstance(profile, dict):
            malformed_prefix = True
            continue
        prefix = profile.get("prefix_columns")
        try:
            if prefix in prefix_map:
                malformed_prefix = True
            prefix_map[prefix] = profile
        except TypeError:
            malformed_prefix = True
    if (len(profiles) != len(EXPECTED_PREFIXES)
            or len(prefix_map) != len(profiles)
            or set(prefix_map) != EXPECTED_PREFIXES
            or malformed_prefix):
        errors.append("profile rows do not cover every expected prefix")
        return errors
    for prefix, profile in prefix_map.items():
        if profile.get("available") is False:
            if not isinstance(profile.get("reason"), str):
                errors.append(f"prefix {prefix} unavailable without reason")
            if profile.get("methods") not in (None, []):
                errors.append(f"prefix {prefix} unavailable but has methods")
            continue
        if profile.get("available") is not True:
            errors.append(f"prefix {prefix} has invalid availability")
            continue
        methods = profile.get("methods")
        if not isinstance(methods, list):
            errors.append(f"prefix {prefix} methods is not a list")
            continue
        method_map = {}
        malformed_method = False
        for method in methods:
            if not isinstance(method, dict):
                malformed_method = True
                continue
            name = method.get("method")
            try:
                if name in method_map:
                    malformed_method = True
                method_map[name] = method
            except TypeError:
                malformed_method = True
        if (len(methods) != len(EXPECTED_METHODS)
                or len(method_map) != len(methods)
                or set(method_map) != EXPECTED_METHODS
                or malformed_method):
            errors.append(f"prefix {prefix} method coverage is incomplete")
            continue
        method_solutions = []
        for method_name, method in method_map.items():
            repetitions = method.get("repetitions")
            requested = method.get("requested_repetitions")
            if (not isinstance(requested, int)
                    or isinstance(requested, bool)
                    or requested <= 0
                    or requested != expected_repeat):
                errors.append(
                    f"prefix {prefix} {method_name} requested repeat mismatch"
                )
            if (not isinstance(repetitions, list)
                    or not isinstance(requested, int)
                    or isinstance(requested, bool)
                    or len(repetitions) != requested):
                errors.append(
                    f"prefix {prefix} {method_name} repetitions are incomplete"
                )
                continue
            repetition_ids = []
            invalid_repetition = False
            for repetition in repetitions:
                if not isinstance(repetition, dict):
                    invalid_repetition = True
                    continue
                repetition_id = repetition.get("repetition")
                outcome = repetition.get("outcome")
                if (not isinstance(repetition_id, int)
                        or isinstance(repetition_id, bool)
                        or outcome not in {"ok", "error"}):
                    invalid_repetition = True
                numeric_fields = ("total_s", "peak_rss_bytes")
                if outcome == "ok":
                    numeric_fields = (*numeric_fields, "backend_s")
                if any(
                        not _nonnegative_number(repetition.get(field))
                        for field in numeric_fields):
                    invalid_repetition = True
                if (outcome == "error"
                        and not isinstance(repetition.get("error"), str)):
                    invalid_repetition = True
                repetition_ids.append(repetition_id)
            if (invalid_repetition
                    or sorted(
                        identifier for identifier in repetition_ids
                        if isinstance(identifier, int)
                        and not isinstance(identifier, bool)
                    ) != list(range(1, requested + 1))):
                errors.append(
                    f"prefix {prefix} {method_name} repetition IDs invalid"
                )
            successful = sum(
                isinstance(repetition, dict)
                and repetition.get("outcome") == "ok"
                for repetition in repetitions
            )
            reported_successes = method.get("successful_repetitions")
            if (not isinstance(reported_successes, int)
                    or isinstance(reported_successes, bool)
                    or reported_successes != successful):
                errors.append(
                    f"prefix {prefix} {method_name} success count mismatch"
                )
            expected_outcome = "ok" if successful == requested else "error"
            if method.get("outcome") != expected_outcome:
                errors.append(
                    f"prefix {prefix} {method_name} outcome/count mismatch"
                )
            if successful:
                timing = method.get("timing")
                solution = method.get("solution")
                timing_fields = (
                    "total_min_s", "total_median_s", "total_max_s",
                    "backend_min_s", "backend_median_s", "backend_max_s",
                )
                solution_fields = (
                    "objective", "route_weight", "artificial_total",
                    "max_row_violation", "max_bound_violation",
                )
                timing_valid = not (
                    not isinstance(timing, dict)
                    or any(
                        not _nonnegative_number(timing.get(field))
                        for field in timing_fields
                    )
                )
                solution_valid = not (
                    not isinstance(solution, dict)
                    or any(
                        not _nonnegative_number(solution.get(field))
                        for field in solution_fields
                    )
                )
                if not timing_valid:
                    errors.append(
                        f"prefix {prefix} {method_name} timing is incomplete"
                    )
                if not solution_valid:
                    errors.append(
                        f"prefix {prefix} {method_name} solution is incomplete"
                    )
                successful_records = [
                    repetition for repetition in repetitions
                    if isinstance(repetition, dict)
                    and repetition.get("outcome") == "ok"
                ]
                if timing_valid:
                    totals = [
                        float(repetition["total_s"])
                        for repetition in successful_records
                        if _nonnegative_number(repetition.get("total_s"))
                    ]
                    backends = [
                        float(repetition["backend_s"])
                        for repetition in successful_records
                        if _nonnegative_number(repetition.get("backend_s"))
                    ]
                    expected_timing = (
                        {
                            "total_min_s": min(totals),
                            "total_median_s": statistics.median(totals),
                            "total_max_s": max(totals),
                            "backend_min_s": min(backends),
                            "backend_median_s": statistics.median(backends),
                            "backend_max_s": max(backends),
                        }
                        if len(totals) == successful
                        and len(backends) == successful
                        else None
                    )
                    if (expected_timing is None
                            or any(not _close(
                                timing[field], expected_timing[field]
                            ) for field in timing_fields)):
                        errors.append(
                            f"prefix {prefix} {method_name} timing summary "
                            "does not match repetitions"
                        )
                    if not (
                            float(timing["total_min_s"])
                            <= float(timing["total_median_s"])
                            <= float(timing["total_max_s"])
                            and float(timing["backend_min_s"])
                            <= float(timing["backend_median_s"])
                            <= float(timing["backend_max_s"])):
                        errors.append(
                            f"prefix {prefix} {method_name} timing order invalid"
                        )
                if solution_valid:
                    method_solutions.append((method_name, solution))
                    for repetition in successful_records:
                        if any(
                                not _nonnegative_number(
                                    repetition.get(field)
                                )
                                or not _close(
                                    repetition[field], solution[field]
                                )
                                for field in solution_fields):
                            errors.append(
                                f"prefix {prefix} {method_name} repetition "
                                "solution differs from method summary"
                            )
                            break
            else:
                if method.get("timing") is not None:
                    errors.append(
                        f"prefix {prefix} {method_name} all-error timing "
                        "must be null"
                    )
                if method.get("solution") is not None:
                    errors.append(
                        f"prefix {prefix} {method_name} all-error solution "
                        "must be null"
                    )
            if method.get("outcome") not in {"ok", "error"}:
                errors.append(
                    f"prefix {prefix} {method_name} has invalid outcome"
                )
        if method_solutions:
            reference_method, reference = method_solutions[0]
            for method_name, solution in method_solutions[1:]:
                for field in (
                    "objective", "route_weight", "artificial_total",
                    "max_row_violation", "max_bound_violation",
                ):
                    if not _close(reference[field], solution[field]):
                        errors.append(
                            f"prefix {prefix} method solutions disagree: "
                            f"{reference_method} versus {method_name}"
                        )
                        break
    return errors
