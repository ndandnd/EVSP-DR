#!/usr/bin/env python3
"""Inventory and approval-gated launcher for MIP convergence statistics."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


_PREIMPORT_REPO_ROOT = Path(__file__).resolve().parents[1]


def _preimport_runtime_artifacts(root: Path) -> list[str]:
    tracked_result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    tracked = (
        {
            value.decode()
            for value in tracked_result.stdout.split(b"\0") if value
        }
        if tracked_result.returncode == 0 else set()
    )
    unsafe = (
        [] if tracked_result.returncode == 0
        else ["<cannot-enumerate-tracked-files>"]
    )
    for scan_root in (root, root / "src"):
        if not scan_root.exists():
            continue
        for current, dirs, files in os.walk(scan_root, followlinks=False):
            current_path = Path(current)
            excluded = {".git", "results", "logs"}
            if current_path == root:
                excluded.add("src")
            retained = []
            for name in dirs:
                path = current_path / name
                relative = str(path.relative_to(root))
                if name in excluded:
                    continue
                if name == "__pycache__" or path.is_symlink():
                    unsafe.append(relative)
                    continue
                retained.append(name)
            dirs[:] = retained
            for name in files:
                path = current_path / name
                relative = str(path.relative_to(root))
                suffix = path.suffix.lower()
                if path.is_symlink():
                    unsafe.append(relative)
                elif suffix in {".pyc", ".pyo", ".so", ".pth"}:
                    unsafe.append(relative)
                elif suffix == ".py" and relative not in tracked:
                    unsafe.append(relative)
    return sorted(set(unsafe))


if __name__ == "__main__":
    _preimport_unsafe = _preimport_runtime_artifacts(_PREIMPORT_REPO_ROOT)
    if _preimport_unsafe:
        raise SystemExit(
            "checkout contains unreviewed importable runtime artifacts before "
            f"launcher import: {_preimport_unsafe[:10]}"
        )


from durable_io import flush_and_fsync
from mip_statistics_inventory import (
    PILOT_BUDGET_HOURS,
    SECONDARY_AGES,
    SECONDARY_SCALES,
    inventory,
    representative_candidates,
    select_age_candidate,
    sha256_file,
    validate_candidate,
)
from run_exact_pool_mip import (
    load_pool,
    merge_validated_partition_start,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REVIEWED_BASE = "ae736fbc9c5fef71f39d7d758b7062355c485313"
WORKER_PATH = REPO_ROOT / "src/submit_mip_statistics.sub"
RUNNER_PATH = REPO_ROOT / "src/run_exact_pool_mip.py"
CODE_PATHS = (
    "src/run_reviewed_python.py",
    "src/launch_mip_statistics_campaign.py",
    "src/run_exact_pool_mip.py",
    "src/expanded_path_realization.py",
    "src/mip_convergence.py",
    "src/durable_io.py",
    "src/audit_giro_known_columns.py",
    "src/config.py",
    "src/utils_v2.py",
    "src/master_lp_scipy.py",
    "src/matching_init.py",
    "src/pricing_dp_og.py",
    "src/install_exact_cg_profile_input.py",
    "src/mip_statistics_environment.py",
    "src/validate_raw_k40_mip_plan.py",
    "src/validate_k40_cs_overnight_plan.py",
    "src/validate_k40_cs_overnight_result.py",
)
DEFAULT_ROOTS = {
    "repool_small": REPO_ROOT / "results/repool_small",
    "exact_big": REPO_ROOT / "results/exact_big",
    "k40_factorial": REPO_ROOT / "results/k40_factorial",
    "bigtar_snapshots": REPO_ROOT / "results/bigtar_snapshots",
    "fresh_preparation": REPO_ROOT / "results/mip_statistics_prep",
}

RAW_K40_BUDGET_HOURS = 8
RAW_K40_SMOKE_BUDGET_HOURS = 0.5
K40_CS_OVERNIGHT_MODE = "k40_cs_overnight"
K40_CS_PACKAGING_BASE_COMMIT = (
    "bf81adb7d2249e049eae1a785ec03b08aba664b5"
)
K40_CS_RAW_BUDGET_HOURS = 8
K40_CS_GIRO40_BUDGET_HOURS = 2
GIRO40_AUGMENTED = "GIRO40-AUGMENTED"
GIRO40_PARTITION_FILE_SHA256 = (
    "8f9944f93f26cf0121e9ecab2fa412d573e90a0189b7a38008d3b2535f54d428"
)
GIRO40_PARTITION_SHA256 = (
    "9e007d51c6bbbdc4f01a00a26ba3bcfa1ec4340df9aab8227a12cf0dc35ecb11"
)
GIRO40_ROUTE_SET_SHA256 = (
    "9b42579ae2d013706cc8d523eb9313fdef4e36eb492a99356483cb526d00085a"
)
RAW_K40_PHYSICAL_COMMIT = (
    "e2b6939b5a5af7033acabec033f6b3d8dde3af4c"
)
RAW_K40_PHYSICAL_CODE_HASHES = {
    "src/run_exact_pool_mip.py":
        "53da4c800d411b657e7a44d1860c01907843a4c43fa00cc34189e75e1ddce6f0",
    "src/expanded_path_realization.py":
        "90764e1b7c17a23580b9c1ffcdffc44a8b9b4f16b6025765cec5ddd0e3a3a91a",
}
RAW_K40_MODES = {"raw_k40", "raw_k40_smoke"}
EXPLICIT_K40_MODES = RAW_K40_MODES | {K40_CS_OVERNIGHT_MODE}
K40_CS_LABELS = ("R1_CS", "R2_CS")
K40_CS_FROZEN_HASHES = {
    "R1_CS": {
        "status":
            "04c3d5d9fe701fbb3bc4fd343e58480fabebf27bb18ef2c60e23a34e29b0200b",
        "journal":
            "128e3d841842bba08e4eba2d9a073322caa8a1de5c64c0e9efe2e747a08c01d4",
    },
    "R2_CS": {
        "status":
            "780431fea40763d42576272bd8e9260f3ed2c8541b6d77f751e17f342dfb1202",
        "journal":
            "8290771a7ca3b6f185070f68a9934e6eaa8894c802ae02ac37f013c25a4b7c31",
    },
}
RAW_K40_INSTANCE_SHA256 = (
    "3508a11f73d1186ae87588656d65ea62812c6e222623ae85488eff26cafb35fd"
)
RAW_K40_TARIFF_SHA256 = (
    "1f51f2e1f6ca303838ebaaf6272a28ff2d6bbee97146cb04d330e10f191f8200"
)
RAW_K40_SOURCE_COMMIT = "eb85ca0cc439956939ba6bf9c42958808d89aadd"
RAW_K40_SPECS = {
    "R1_CA": {
        "campaign": "k40fx_20260814T140232Z_eb85ca0c",
        "filename": "k40r1_flat_CA.m1440.snapshot.json",
        "replicate": "R1",
        "initial_pool": "artificial",
    },
    "R1_CS": {
        "campaign": "k40fx_20260814T140232Z_eb85ca0c",
        "filename": "k40r1_flat_CS.m1440.snapshot.json",
        "replicate": "R1",
        "initial_pool": "singletons",
    },
    "R2_CA": {
        "campaign": "k40fx_20260814T191933Z_eb85ca0c",
        "filename": "k40r1_flat_CA.m1440.snapshot.json",
        "replicate": "R2",
        "initial_pool": "artificial",
    },
    "R2_CS": {
        "campaign": "k40fx_20260814T191933Z_eb85ca0c",
        "filename": "k40r1_flat_CS.m1440.snapshot.json",
        "replicate": "R2",
        "initial_pool": "singletons",
    },
}


def _git(*args, binary=False):
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=not binary,
    )


def _unsafe_runtime_artifacts(root: Path) -> list[str]:
    return _preimport_runtime_artifacts(root)


def checkout_identity(*, require_detached: bool) -> dict:
    head = _git("rev-parse", "--verify", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    untracked = _git("status", "--porcelain", "--untracked-files=all")
    symbolic = _git("symbolic-ref", "-q", "HEAD")
    ancestor = _git(
        "merge-base", "--is-ancestor", REVIEWED_BASE, "HEAD"
    )
    commit = head.stdout.strip()
    if (
        head.returncode != 0
        or len(commit) != 40
        or status.returncode != 0
        or untracked.returncode != 0
        or status.stdout.strip()
        or ancestor.returncode != 0
    ):
        raise SystemExit(
            "checkout must be tracked-clean and descend from reviewed base"
        )
    untracked_imports = [
        line for line in untracked.stdout.splitlines()
        if line.startswith("?? ")
        and re.search(r"\.(?:py|pth|so|sub)$", line[3:])
    ]
    if untracked_imports:
        raise SystemExit("checkout contains untracked importable worker code")
    if symbolic.returncode == 0:
        detached = False
        branch = symbolic.stdout.strip()
    elif symbolic.returncode == 1:
        detached = True
        branch = ""
    else:
        raise SystemExit("cannot verify checkout branch state")
    if require_detached and not detached:
        raise SystemExit("submission requires a detached reviewed checkout")
    runtime_artifacts = (
        _unsafe_runtime_artifacts(REPO_ROOT) if detached else []
    )
    if runtime_artifacts:
        raise SystemExit(
            "checkout contains ignored/importable runtime artifacts; "
            "remove them and invoke Python with -B: "
            f"{runtime_artifacts[:10]}"
        )
    return {
        "expected_commit": commit,
        "reviewed_base_commit": REVIEWED_BASE,
        "detached": detached,
        "branch": branch,
        "tracked_clean": True,
        "runtime_artifacts_absent": True,
    }


def _write_new_atomic(path: Path, payload: bytes, *, executable=False) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.tmp.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if executable:
            temporary.chmod(0o500)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite {path}") from exc
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _replace_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        flush_and_fsync(handle)
    os.replace(temporary, path)


def _existing_execution_comments() -> set[str]:
    comments = set()
    for command in (
        ["squeue", "-h", "-o", "%k"],
        [
            "sacct", "-X", "-n", "-P", "--starttime", "2026-01-01",
            "--format=Comment",
        ],
    ):
        result = subprocess.run(
            command, text=True, capture_output=True, check=False
        )
        if result.returncode != 0:
            raise SystemExit(
                "cannot query Slurm execution-deduplication comments"
            )
        comments.update(
            line.strip().split("|", 1)[0]
            for line in result.stdout.splitlines() if line.strip()
        )
    return comments


def _reserve_execution_digests(plan: dict, plan_sha: str) -> list[Path]:
    root = Path(plan["shared_reservation_root"]).expanduser().resolve()
    if REPO_ROOT.resolve() == root or REPO_ROOT.resolve() in root.parents:
        raise SystemExit("execution reservation root must be cluster-shared")
    root.mkdir(parents=True, exist_ok=True)
    reservations = []
    try:
        for job in sorted(plan["jobs"], key=lambda item: item["cell_id"]):
            path = root / f"{job['execution_digest']}.json"
            payload = (json.dumps({
                "schema": "evsp-dr-mip-statistics-reservation-v1",
                "execution_digest": job["execution_digest"],
                "approved_plan_sha256": plan_sha,
                "cell_id": job["cell_id"],
                "campaign": plan["campaign"],
            }, sort_keys=True) + "\n").encode()
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(path, flags, 0o600)
            except FileExistsError as exc:
                raise SystemExit(
                    f"execution digest already reserved: "
                    f"{job['execution_digest']}"
                ) from exc
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            reservations.append(path)
    except Exception:
        for path in reservations:
            path.unlink(missing_ok=True)
        raise
    return reservations


def _canonical(payload: dict) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()


def _parse_assignments(values, *, value_type=Path) -> dict:
    parsed = {}
    for value in values or []:
        if "=" not in value:
            raise SystemExit(f"expected NAME=VALUE, found {value!r}")
        name, raw = value.split("=", 1)
        if not name or name in parsed:
            raise SystemExit(f"duplicate/empty assignment {name!r}")
        parsed[name] = value_type(raw)
    return parsed


def resolve_raw_k40_candidates(
    assignments: dict[str, Path],
    *,
    data_roots: list[Path],
    enforce_frozen_path: bool = True,
    expected_status_sha256: dict[str, str] | None = None,
    journal_assignments: dict[str, Path] | None = None,
    expected_journal_sha256: dict[str, str] | None = None,
    expected_labels: tuple[str, ...] | None = None,
) -> dict[str, dict]:
    """Validate the four explicit, non-GIRO k40 factorial snapshots."""

    expected_label_set = set(expected_labels or RAW_K40_SPECS)
    if not expected_label_set <= set(RAW_K40_SPECS):
        raise ValueError("raw-k40 expected labels are unsupported")
    if set(assignments) != expected_label_set:
        missing = sorted(expected_label_set - set(assignments))
        extra = sorted(set(assignments) - expected_label_set)
        raise ValueError(
            f"raw-k40 requires exactly {sorted(expected_label_set)}; "
            f"missing={missing}, extra={extra}"
        )
    for label, values in (
        ("status SHA-256", expected_status_sha256),
        ("journal paths", journal_assignments),
        ("journal SHA-256", expected_journal_sha256),
    ):
        if values is not None and set(values) != expected_label_set:
            raise ValueError(
                f"raw-k40 {label} require exactly "
                f"{sorted(expected_label_set)}"
            )
    if journal_assignments is not None:
        journal_paths = [
            path.expanduser().resolve()
            for path in journal_assignments.values()
        ]
        if len(journal_paths) != len(set(journal_paths)):
            raise ValueError("raw-k40 journal paths must be distinct")
    for values in (
        expected_status_sha256, expected_journal_sha256
    ):
        if values is not None and any(
            not re.fullmatch(r"[0-9a-f]{64}", digest)
            for digest in values.values()
        ):
            raise ValueError("raw-k40 expected SHA-256 is malformed")
    resolved_paths = [
        path.expanduser().resolve() for path in assignments.values()
    ]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("raw-k40 snapshot paths must be distinct")

    selected = {}
    for label in expected_labels or tuple(RAW_K40_SPECS):
        spec = RAW_K40_SPECS[label]
        path = assignments[label].expanduser().resolve()
        if enforce_frozen_path and (
            path.name != spec["filename"]
            or path.parent.name != spec["campaign"]
        ):
            raise ValueError(
                f"{label} path does not identify its frozen campaign cell"
            )
        candidate = validate_candidate(
            path,
            source_family="k40_factorial",
            data_roots=data_roots,
        )
        treatment = candidate.get("treatment") or {}
        physics = candidate.get("physics") or {}
        expected_physics = {
            "soc_step": 15.0,
            "block_min": 10.0,
            "g_kwh": 300.0,
            "charge_kw": 300.0,
            "min_soc_frac": 0.0,
        }
        if (
            candidate.get("scale") != 40
            or candidate.get("trip_count") != 947
            or candidate.get("instance_sha256")
            != RAW_K40_INSTANCE_SHA256
            or candidate.get("tariff_sha256")
            != RAW_K40_TARIFF_SHA256
            or candidate.get("source_commit") != RAW_K40_SOURCE_COMMIT
            or candidate.get("snapshot_mark_minutes") != 1440
            or not math.isclose(
                float(candidate.get("age_hours", math.nan)),
                24.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or candidate.get("stop_reason") != "snapshot_m1440"
            or treatment.get("master_sense") != "cover"
            or treatment.get("initial_pool") != spec["initial_pool"]
            or Path(str(candidate.get("prices_csv") or "")).name
            != "hourly_prices_flat.csv"
            or any(
                not math.isclose(
                    float(physics.get(key, math.nan)),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                for key, expected in expected_physics.items()
            )
        ):
            raise ValueError(f"{label} does not match the frozen raw-k40 cell")
        candidate = dict(candidate)
        if (
            expected_status_sha256 is not None
            and candidate.get("status_sha256")
            != expected_status_sha256.get(label)
        ):
            raise ValueError(f"{label} status SHA-256 mismatch")
        if journal_assignments is not None:
            expected_journal_path = (
                journal_assignments[label].expanduser().resolve()
            )
            if Path(candidate["journal_path"]).resolve() != expected_journal_path:
                raise ValueError(f"{label} journal path mismatch")
        if (
            expected_journal_sha256 is not None
            and candidate.get("journal_sha256")
            != expected_journal_sha256.get(label)
        ):
            raise ValueError(f"{label} journal SHA-256 mismatch")
        candidate["inventory_replicate"] = candidate.get("replicate")
        candidate["replicate"] = spec["replicate"]
        candidate["raw_k40_label"] = label
        selected[label] = candidate

    identity_fields = (
        "instance_sha256", "tariff_sha256", "trip_set_sha256", "trip_count",
        "csv", "prices_csv",
    )
    reference_candidate = selected[next(iter(selected))]
    if any(
        selected[label].get(field) != reference_candidate.get(field)
        for label in selected
        for field in identity_fields
    ):
        raise ValueError("raw-k40 snapshots do not share one frozen instance")
    return selected


def _python_identity(path: Path) -> dict:
    executable = path.expanduser().resolve()
    try:
        result = subprocess.run(
            [
                str(executable),
                str(REPO_ROOT / "src/mip_statistics_environment.py"),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        return {
            "available": False,
            "executable": str(executable),
            "reason": str(exc),
        }
    if result.returncode != 0 or not executable.is_file():
        return {
            "available": False,
            "executable": str(executable),
            "reason": result.stderr.strip() or "python unavailable",
        }
    if result.returncode != 0:
        return {
            "available": False,
            "executable": str(executable),
            "executable_sha256": (
                sha256_file(executable) if executable.is_file() else None
            ),
            "reason": result.stderr.strip() or result.stdout.strip(),
        }
    details = json.loads(result.stdout)
    return {"available": True, **details}


def _safe_export_value(label: str, value: str) -> str:
    if any(character in value for character in (",", "\n", "\r", "\0")):
        raise ValueError(f"{label} is unsafe for Slurm --export")
    return value


def _validated_start(path: Path, candidate: dict) -> dict:
    source = path.expanduser().resolve()
    raw = source.read_bytes()
    payload = json.loads(raw)
    routes = payload.get("routes") if isinstance(payload, dict) else None
    if (
        not isinstance(routes, list)
        or not routes
        or (
            payload.get("schema") != "evsp-dr-k40-giro40-partition-v1"
            and (
                payload.get("infeasible") != []
                or payload.get("source") != "rerealized"
            )
        )
    ):
        raise ValueError(f"GIRO start is missing/partial: {source}")
    status = json.loads(Path(candidate["status_path"]).read_text())
    physics = payload.get("physics")
    if not isinstance(physics, dict):
        raise ValueError(f"GIRO start lacks re-realized physics: {source}")
    for start_key, candidate_key in (
        ("g_kwh", "g_kwh"),
        ("charge_kw", "charge_kw"),
        ("reserve_frac", "min_soc_frac"),
    ):
        if not math.isclose(
            float(physics.get(start_key, math.nan)),
            float(candidate["physics"][candidate_key]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"GIRO start physics mismatch: {source}")
    if payload.get("schema") == "evsp-dr-k40-giro40-partition-v1":
        if (
            hashlib.sha256(raw).hexdigest()
            != GIRO40_PARTITION_FILE_SHA256
            or payload.get("source") != GIRO40_AUGMENTED
            or payload.get("route_count") != 40
            or len(routes) != 40
            or payload.get("partition_sha256")
            != GIRO40_PARTITION_SHA256
            or payload.get("route_set_sha256")
            != GIRO40_ROUTE_SET_SHA256
            or payload.get("continuous_cost_pricing_certified") is not False
            or payload.get("pricing_certificate_scope")
            != "none_for_augmented_routes"
        ):
            raise ValueError(f"GIRO40 partition metadata is invalid: {source}")
        for route in routes:
            physical = route.get("physical_realization") or {}
            if (
                not route.get("continuous_realized_charging_blocks")
                and sum(float(value) for value in (
                    (route.get("charging_stops") or {}).get("kwh") or []
                )) > 1e-9
            ):
                raise ValueError(
                    f"GIRO40 route lacks continuous blocks: {source}"
                )
            if (
                physical.get("status")
                != "validated_continuous_injection"
                or physical.get("continuous_cost_pricing_certified")
                is not False
            ):
                raise ValueError(
                    f"GIRO40 route physical metadata is invalid: {source}"
                )
    if (
        Path(str(payload.get("prices_csv") or "")).name
        != Path(str(candidate["prices_csv"])).name
    ):
        raise ValueError(f"GIRO start tariff mismatch: {source}")
    expected = set(status["trip_ids"])
    counts = {trip: 0 for trip in expected}
    for route in routes:
        nodes = route.get("route", route.get("route_nodes", []))
        trips = [
            node for node in nodes
            if isinstance(node, int) and not isinstance(node, bool)
        ]
        if not trips or len(trips) != len(set(trips)):
            raise ValueError(f"GIRO start has an invalid route: {source}")
        for trip in trips:
            if trip not in counts:
                raise ValueError(f"GIRO start has an unknown trip: {trip}")
            counts[trip] += 1
    if any(count != 1 for count in counts.values()):
        raise ValueError(f"GIRO start is not an exact partition: {source}")
    detail = _physical_start_validation(source, candidate)
    return {
        "path": str(source),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "route_count": len(routes),
        "trip_set_sha256": candidate["trip_set_sha256"],
        "physical_replay_validated": True,
        "validated_bus_count": detail["validated_bus_count"],
        "expected_full_objective": detail["expected_full_objective"],
    }


def _physical_start_validation(path: Path, candidate: dict) -> dict:
    relative = Path(candidate["csv"])
    instance = Path(candidate["instance_path"]).resolve()
    data_root = instance
    for _part in relative.parts:
        data_root = data_root.parent
    if (data_root / relative).resolve() != instance:
        raise ValueError("cannot establish GIRO validation data root")
    tariff = (data_root / candidate["prices_csv"]).resolve()
    if (
        not tariff.is_file()
        or sha256_file(tariff) != candidate["tariff_sha256"]
    ):
        raise ValueError("GIRO validation tariff differs from candidate")
    status, routes, trips = load_pool(Path(candidate["status_path"]))
    _merged, _indices, detail = merge_validated_partition_start(
        routes,
        trips,
        path,
        candidate["prices_csv"],
        status,
        data_dir=data_root,
        reference_data_dir=REPO_ROOT / "data",
    )
    return detail


def _rep_code(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", value).upper()
    return (cleaned or "X")[-2:].rjust(2, "0")


def _slurm_wall_time(budget_hours: float) -> str:
    total_seconds = int(round(float(budget_hours) * 3600.0)) + 600
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _job_name(job, campaign: str) -> str:
    if job.get("matrix") == K40_CS_OVERNIGHT_MODE:
        label = str(job["source"]["raw_k40_label"]).replace("_", "")
        treatment = "R8" if job["arm"] == "RAW" else "G2"
        nonce = hashlib.sha256(
            f"{campaign}|{job['cell_id']}".encode()
        ).hexdigest()[:2].upper()
        name = f"K40{label}{treatment}{nonce}"
        if len(name) > 15:
            raise ValueError(f"Slurm name exceeds 15 characters: {name}")
        return name
    if job.get("matrix") in RAW_K40_MODES:
        label = str(job["source"]["raw_k40_label"]).replace("_", "")
        nonce = hashlib.sha256(
            f"{campaign}|{job['cell_id']}".encode()
        ).hexdigest()[:2].upper()
        budget = (
            "30M" if job.get("matrix") == "raw_k40_smoke"
            else "8H"
        )
        name = f"RK40{label}{budget}{nonce}"
        if len(name) > 15:
            raise ValueError(f"Slurm name exceeds 15 characters: {name}")
        return name
    age = (
        f"A{int(round(job['age_hours'])):02d}"
        if job.get("matrix") == "secondary" else ""
    )
    arm = "R" if job["arm"] == "RAW" else "G"
    nonce = hashlib.sha256(
        f"{campaign}|{job['cell_id']}".encode()
    ).hexdigest()[:2].upper()
    name = (
        f"S{job['scale']:02d}{_rep_code(job['replicate'])}"
        f"{arm}{job['budget_hours']:02d}H{age}{nonce}"
    )
    if len(name) > 15:
        raise ValueError(f"Slurm name exceeds 15 characters: {name}")
    return name


def _fresh_preparation(scale: int) -> dict:
    filename = f"Practice_{scale}bus.csv"
    return {
        "scale": scale,
        "classification": "first-N stress instance",
        "verified_service_day_sample": False,
        "instance": filename,
        "prices_csv": "hourly_prices_flat.csv",
        "physics": {
            "g_kwh": 300,
            "charge_kw": 300,
            "min_soc_frac": 0.0,
            "soc_step": 15,
            "block_min": 10,
        },
        "column_discovery": {
            "master_sense": "cover",
            "initial_pool": "singletons",
        },
        "final_mip_partitioning": "strict_exact_once",
        "snapshot_minutes": [60, 180, 360, 600, 720, 900, 1440],
        "command": (
            "python -u src/exact_pricer_expanded.py "
            f"--csv {filename} --prices_csv hourly_prices_flat.csv "
            "--g-kwh 300 --charge-kw 300 --min-soc-frac 0 "
            "--soc-step 15 --block-min 10 --master-sense cover "
            "--initial-pool singletons --snapshot-at-minutes "
            "60,180,360,600,720,900,1440 --resume "
            f"--out results/mip_statistics_prep/k{scale}.json"
        ),
        "planned_only": True,
    }


def _job_from_candidate(
    candidate,
    *,
    arm,
    budget_hours,
    start_map,
    matrix,
    age_label=None,
):
    start = None
    blocked = []
    augmented = arm in {"GIRO", GIRO40_AUGMENTED}
    if augmented:
        start_path = (
            start_map.get(str(candidate.get("raw_k40_label")))
            or start_map.get(str(candidate["scale"]))
        )
        if start_path is None:
            blocked.append("validated_giro_start_missing")
        else:
            try:
                start = _validated_start(start_path, candidate)
            except (OSError, ValueError) as exc:
                blocked.append(f"validated_giro_start_invalid: {exc}")
    if matrix in EXPLICIT_K40_MODES:
        treatment = (
            "raw" if arm == "RAW" else "giro40_augmented"
        )
        cell = (
            f"k40_{candidate['raw_k40_label'].lower()}_{treatment}_m1440"
        )
    else:
        cell = (
            f"k{candidate['scale']}_{candidate['replicate']}_{arm.lower()}"
            + (f"_a{age_label}" if age_label is not None else "")
        )
    return {
        "cell_id": cell,
        "matrix": matrix,
        "scale": candidate["scale"],
        "replicate": candidate["replicate"],
        "arm": arm,
        "augmentation_changes_column_set": augmented,
        "partitioning": "strict_exact_once",
        "two_stage": True,
        "cost_stage_policy": (
            "disabled_for_mixed_augmented_cost_semantics"
            if arm == GIRO40_AUGMENTED
            else "run_only_after_finite_pool_fleet_proof"
        ),
        "budget_hours": budget_hours,
        "time_limit_s": int(budget_hours * 3600),
        "threads": 8,
        "mip_gap": 1e-4,
        "age_hours": candidate.get("age_hours"),
        "source": candidate,
        "validated_start": start,
        "blocked_reasons": blocked,
    }


def build_plan(
    inventory_payload,
    *,
    mode: str,
    campaign: str,
    start_map: dict,
    identity: dict,
    explicit_raw_candidates: dict[str, dict] | None = None,
    python_path: Path = Path(sys.executable),
    reservation_root: Path = Path(
        "/share/evsp-dr/mip-statistics-execution-reservations"
    ),
) -> dict:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,79}", campaign):
        raise ValueError("campaign name must be a safe relative identifier")
    if (
        mode == K40_CS_OVERNIGHT_MODE
        and _git(
            "merge-base", "--is-ancestor",
            K40_CS_PACKAGING_BASE_COMMIT,
            identity["expected_commit"],
        ).returncode != 0
    ):
        raise ValueError("overnight checkout does not descend from bf81adb7")
    selected = representative_candidates(inventory_payload)
    preparations = [_fresh_preparation(scale) for scale in (10, 20)]
    jobs = []
    if mode == "pilot":
        for scale, budget in PILOT_BUDGET_HOURS.items():
            candidate = selected.get(scale)
            if candidate is None:
                continue
            for arm in ("RAW", "GIRO"):
                jobs.append(_job_from_candidate(
                    candidate,
                    arm=arm,
                    budget_hours=budget,
                    start_map=start_map,
                    matrix="pilot",
                ))
    elif mode == "secondary":
        candidates = inventory_payload.get("candidates") or []
        for scale in SECONDARY_SCALES:
            for target in SECONDARY_AGES:
                candidate = select_age_candidate(
                    candidates, scale=scale, target=target
                )
                if candidate is None:
                    continue
                label = (
                    "-or-".join(str(value) for value in target)
                    if isinstance(target, tuple) else str(target)
                )
                jobs.append(_job_from_candidate(
                    candidate,
                    arm="GIRO",
                    budget_hours=2,
                    start_map=start_map,
                    matrix="secondary",
                    age_label=label,
                ))
    elif mode in RAW_K40_MODES:
        if set(explicit_raw_candidates or {}) != set(RAW_K40_SPECS):
            raise ValueError("raw_k40 mode requires four explicit candidates")
        preparations = []
        for label in RAW_K40_SPECS:
            candidate = explicit_raw_candidates[label]
            jobs.append(_job_from_candidate(
                candidate,
                arm="RAW",
                budget_hours=(
                    RAW_K40_SMOKE_BUDGET_HOURS
                    if mode == "raw_k40_smoke"
                    else RAW_K40_BUDGET_HOURS
                ),
                start_map={},
                matrix=mode,
            ))
    elif mode == K40_CS_OVERNIGHT_MODE:
        if set(explicit_raw_candidates or {}) != set(K40_CS_LABELS):
            raise ValueError(
                "k40_cs_overnight requires R1_CS and R2_CS candidates"
            )
        preparations = []
        for label in K40_CS_LABELS:
            candidate = explicit_raw_candidates[label]
            jobs.append(_job_from_candidate(
                candidate,
                arm="RAW",
                budget_hours=K40_CS_RAW_BUDGET_HOURS,
                start_map={},
                matrix=mode,
            ))
            jobs.append(_job_from_candidate(
                candidate,
                arm=GIRO40_AUGMENTED,
                budget_hours=K40_CS_GIRO40_BUDGET_HOURS,
                start_map=start_map,
                matrix=mode,
            ))
    else:
        raise ValueError(mode)
    campaign_root = (
        REPO_ROOT / "src/results/mip_statistics" / campaign
    ).resolve()
    log_root = (
        REPO_ROOT / "src/logs/mip_statistics" / campaign
    ).resolve()
    if (
        (REPO_ROOT / "src/results/mip_statistics").resolve()
        not in campaign_root.parents
        or (REPO_ROOT / "src/logs/mip_statistics").resolve()
        not in log_root.parents
    ):
        raise ValueError("campaign paths escape designated namespaces")
    for job in jobs:
        job["job_name"] = _job_name(job, campaign)
        cell_root = campaign_root / "input" / job["cell_id"]
        source_status_name = Path(job["source"]["status_path"]).name
        status_name = (
            source_status_name
            if source_status_name.endswith(".snapshot.json")
            else f"{Path(source_status_name).stem}.frozen.snapshot.json"
        )
        job.update({
            "staged_status": str(cell_root / status_name),
            "staged_journal": str(
                cell_root / f"{status_name}.columns.jsonl"
            ),
            "staged_instance": str(
                cell_root / "data" / job["source"]["csv"]
            ),
            "staged_tariff": str(
                cell_root / "data" / job["source"]["prices_csv"]
            ),
            "staged_start": (
                str(cell_root / "validated_start.json")
                if job["validated_start"] else None
            ),
            "output": str(
                campaign_root / "outputs" / f"{job['cell_id']}.json"
            ),
            "progress_dir": str(
                campaign_root / "progress" / job["cell_id"]
            ),
            "job_id": None,
            "submission_state": "planned",
        })
    inventory_sha = hashlib.sha256(
        _canonical(inventory_payload)
    ).hexdigest()
    worker_sha = (
        sha256_file(WORKER_PATH) if WORKER_PATH.is_file() else None
    )
    runner_sha = sha256_file(RUNNER_PATH)
    python_identity = _python_identity(python_path)
    environment = {
        "HOME": _safe_export_value(
            "HOME", os.environ.get("HOME", str(Path.home()))
        ),
        "USER": _safe_export_value(
            "USER", os.environ.get("USER", "")
        ),
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "EVSP_DR_ROOT": _safe_export_value(
            "EVSP_DR_ROOT", str(REPO_ROOT)
        ),
        "EVSP_EXPECTED_COMMIT": identity["expected_commit"],
        "EVSP_MIP_PYTHON": _safe_export_value(
            "EVSP_MIP_PYTHON", python_identity["executable"]
        ),
    }
    code_hashes = {
        path: sha256_file(REPO_ROOT / path) for path in CODE_PATHS
    }
    if mode == "raw_k40_smoke":
        observed_physical_hashes = {
            path: code_hashes[path]
            for path in RAW_K40_PHYSICAL_CODE_HASHES
        }
        if observed_physical_hashes != RAW_K40_PHYSICAL_CODE_HASHES:
            raise ValueError(
                "physical realization runtime differs from reviewed "
                f"commit {RAW_K40_PHYSICAL_COMMIT}"
            )
    for job in jobs:
        job["execution"] = {
            "cell_id": job["cell_id"],
            "arm": job["arm"],
            "matrix": job["matrix"],
            "source_label": job["source"].get("raw_k40_label"),
            "source_master_sense": (
                job["source"].get("treatment") or {}
            ).get("master_sense"),
            "source_initial_pool": (
                job["source"].get("treatment") or {}
            ).get("initial_pool"),
            "scale": job["scale"],
            "replicate": job["replicate"],
            "time_limit_s": job["time_limit_s"],
            "threads": 8,
            "mip_gap": job["mip_gap"],
            "status": job["staged_status"],
            "status_sha256": job["source"]["status_sha256"],
            "journal": job["staged_journal"],
            "journal_sha256": job["source"]["journal_sha256"],
            "instance": job["staged_instance"],
            "instance_sha256": job["source"]["instance_sha256"],
            "instance_relative": job["source"]["csv"],
            "tariff": job["staged_tariff"],
            "tariff_sha256": job["source"]["tariff_sha256"],
            "tariff_relative": job["source"]["prices_csv"],
            "validated_start": job["staged_start"],
            "validated_start_sha256": (
                job["validated_start"]["sha256"]
                if job["validated_start"] else None
            ),
            "output": job["output"],
            "progress_dir": job["progress_dir"],
        }
        job["execution_digest"] = hashlib.sha256(_canonical({
            "arm": job["arm"],
            "scale": job["scale"],
            "source_status_sha256": job["source"]["status_sha256"],
            "source_journal_sha256": job["source"]["journal_sha256"],
            "instance_sha256": job["source"]["instance_sha256"],
            "tariff_sha256": job["source"]["tariff_sha256"],
            "validated_start_sha256": (
                job["validated_start"]["sha256"]
                if job["validated_start"] else None
            ),
            "time_limit_s": job["time_limit_s"],
            "threads": 8,
            "mip_gap": job["mip_gap"],
            "code_hashes": code_hashes,
            "worker_sha256": worker_sha,
            "environment_identity_sha256": python_identity.get(
                "identity_sha256"
            ),
        })).hexdigest()
    missing_scales = (
        [scale for scale in PILOT_BUDGET_HOURS if scale not in selected]
        if mode == "pilot" else []
    )
    expected_secondary_cells = len(SECONDARY_SCALES) * len(SECONDARY_AGES)
    missing_matrix_cells = (
        expected_secondary_cells - len(jobs)
        if mode == "secondary" else 0
    )
    plan = {
        "schema": "evsp-dr-mip-statistics-approved-plan-v1",
        "campaign": campaign,
        "mode": mode,
        "campaign_root": str(campaign_root),
        "log_root": str(log_root),
        "checkout_identity": identity,
        "inventory_sha256": inventory_sha,
        "selection_rule": (
            "four explicit hash-validated raw k40 factorial m1440 cells"
            if mode in EXPLICIT_K40_MODES
            else inventory_payload["selection_rule"]
        ),
        "selected_candidates": (
            {
                label: candidate["candidate_id"]
                for label, candidate in (explicit_raw_candidates or {}).items()
            }
            if mode in EXPLICIT_K40_MODES
            else {
                str(scale): candidate["candidate_id"]
                for scale, candidate in selected.items()
            }
        ),
        "fresh_exact_cg_preparations": preparations,
        "resources": {
            "partition": "scaglione",
            "threads": 8,
            "requeue": False,
            "signal": "B:USR1@180",
            "submission_release": (
                "single_atomic_four_task_array_submission"
                if mode == K40_CS_OVERNIGHT_MODE
                else "held_jobs_released_after_submission"
            ),
            "array_tasks": (
                4 if mode == K40_CS_OVERNIGHT_MODE else None
            ),
            "array_slurm_wall_time": (
                _slurm_wall_time(K40_CS_RAW_BUDGET_HOURS)
                if mode == K40_CS_OVERNIGHT_MODE else None
            ),
        },
        "worker": str(WORKER_PATH),
        "worker_sha256": worker_sha,
        "runner": str(RUNNER_PATH),
        "runner_sha256": runner_sha,
        "code_hashes": code_hashes,
        "physical_realization_review": (
            ({
                "commit": RAW_K40_PHYSICAL_COMMIT,
                "code_hashes": RAW_K40_PHYSICAL_CODE_HASHES,
            } if mode == "raw_k40_smoke" else {
                "semantics_base_commit": RAW_K40_PHYSICAL_COMMIT,
                "packaging_base_commit": K40_CS_PACKAGING_BASE_COMMIT,
                "runtime_code_hashes": {
                    path: code_hashes[path]
                    for path in RAW_K40_PHYSICAL_CODE_HASHES
                },
            })
            if mode in {"raw_k40_smoke", K40_CS_OVERNIGHT_MODE} else None
        ),
        "python_identity": python_identity,
        "environment_whitelist": environment,
        "shared_reservation_root": str(
            reservation_root.expanduser().resolve()
        ),
        "jobs": jobs,
        "missing_scales": missing_scales,
        "missing_matrix_cells": missing_matrix_cells,
        "inventory_missing_roots": inventory_payload.get("missing_roots") or [],
        "inventory_missing_slots": inventory_payload.get("missing_slots") or [],
        "blocked": (
            any(job["blocked_reasons"] for job in jobs)
            or (bool(jobs) and not python_identity["available"])
            or (
                mode not in EXPLICIT_K40_MODES
                and bool(inventory_payload.get("missing_roots"))
            )
            or (
                mode not in EXPLICIT_K40_MODES
                and bool(inventory_payload.get("missing_slots"))
            )
            or (mode == "pilot" and bool(missing_scales))
            or (mode == "secondary" and missing_matrix_cells > 0)
        ),
        "raw_k40_guards": (
            {
                "giro_columns_allowed": False,
                "extra_routes_allowed": False,
                "initial_partition_allowed": False,
                "strict_partitioning": True,
                "budget_seconds": int((
                    RAW_K40_SMOKE_BUDGET_HOURS
                    if mode == "raw_k40_smoke"
                    else RAW_K40_BUDGET_HOURS
                ) * 3600),
                "expected_trip_count": 947,
                "expected_snapshot_minutes": 1440,
            }
            if mode in RAW_K40_MODES else None
        ),
        "k40_cs_overnight_guards": (
            {
                "raw_budget_seconds": int(
                    K40_CS_RAW_BUDGET_HOURS * 3600
                ),
                "giro40_augmented_budget_seconds": int(
                    K40_CS_GIRO40_BUDGET_HOURS * 3600
                ),
                "raw_external_routes_allowed": False,
                "giro40_partition_route_count": 40,
                "strict_partitioning": True,
                "expected_trip_count": 947,
                "expected_snapshot_minutes": 1440,
                "continuous_cost_pricing_certified": False,
                "ca_jobs_included": False,
            }
            if mode == K40_CS_OVERNIGHT_MODE else None
        ),
        "global_route_space_optimality_claimed": False,
    }
    return plan


def _stage_and_submit(plan: dict, plan_sha: str) -> dict:
    if not plan["jobs"]:
        raise SystemExit("approved plan contains no runnable jobs")
    if plan["blocked"]:
        raise SystemExit("approved plan contains blocked cells")
    root = Path(plan["campaign_root"])
    logs = Path(plan["log_root"])
    if root.exists() or logs.exists():
        raise SystemExit("campaign already exists; reruns need a new name")
    missing_commands = [
        command for command in (
            "sbatch", "scontrol", "scancel", "squeue", "sacct",
        )
        if shutil.which(command) is None
    ]
    if missing_commands:
        raise SystemExit(
            f"required Slurm commands are unavailable: {missing_commands}"
        )
    observed_identity = checkout_identity(require_detached=True)
    if observed_identity != plan["checkout_identity"]:
        raise SystemExit("submission checkout differs from approved plan")
    if plan["worker_sha256"] is None:
        raise SystemExit("worker is not committed/available")
    root.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    worker = root / "input/submit_mip_statistics.sub"
    runner = root / "input/run_exact_pool_mip.py"
    if (
        sha256_file(WORKER_PATH) != plan["worker_sha256"]
        or sha256_file(RUNNER_PATH) != plan["runner_sha256"]
    ):
        raise SystemExit("approved worker/runner changed before staging")
    _write_new_atomic(worker, WORKER_PATH.read_bytes(), executable=True)
    _write_new_atomic(runner, RUNNER_PATH.read_bytes())
    if (
        sha256_file(worker) != plan["worker_sha256"]
        or sha256_file(runner) != plan["runner_sha256"]
    ):
        raise SystemExit("staged worker/runner hash mismatch")
    reviewed_code = root / "input/reviewed_code"
    for relative, expected in plan["code_hashes"].items():
        source_path = REPO_ROOT / relative
        if sha256_file(source_path) != expected:
            raise SystemExit(f"reviewed code changed before staging: {relative}")
        staged_path = reviewed_code / relative
        _write_new_atomic(staged_path, source_path.read_bytes())
        if sha256_file(staged_path) != expected:
            raise SystemExit(f"staged code hash mismatch: {relative}")
    plan_path = root / "approved-plan.json"
    plan_raw = _canonical(plan)
    _write_new_atomic(plan_path, plan_raw)
    if hashlib.sha256(plan_raw).hexdigest() != plan_sha:
        raise SystemExit("staged approval plan hash mismatch")
    # Phase 1: stage and verify every cell before any call to sbatch.
    for job in plan["jobs"]:
        source = job["source"]
        copies = (
            ("status_path", "staged_status", "status_sha256"),
            ("journal_path", "staged_journal", "journal_sha256"),
            ("instance_path", "staged_instance", "instance_sha256"),
            ("tariff_path", "staged_tariff", "tariff_sha256"),
        )
        for source_key, staged_key, hash_key in copies:
            source_path = Path(source[source_key])
            if sha256_file(source_path) != source[hash_key]:
                raise SystemExit(f"{job['cell_id']}: source changed")
            _write_new_atomic(
                Path(job[staged_key]), source_path.read_bytes()
            )
            if sha256_file(Path(job[staged_key])) != source[hash_key]:
                raise SystemExit(f"{job['cell_id']}: staged hash mismatch")
        if job["validated_start"]:
            start = job["validated_start"]
            start_path = Path(start["path"])
            if sha256_file(start_path) != start["sha256"]:
                raise SystemExit(f"{job['cell_id']}: start changed")
            _write_new_atomic(
                Path(job["staged_start"]), start_path.read_bytes()
            )
            if sha256_file(Path(job["staged_start"])) != (
                    job["validated_start"]["sha256"]):
                raise SystemExit(f"{job['cell_id']}: staged start mismatch")
        execution = job["execution"]
        for path_key, hash_key in (
            ("status", "status_sha256"),
            ("journal", "journal_sha256"),
            ("instance", "instance_sha256"),
            ("tariff", "tariff_sha256"),
        ):
            if sha256_file(Path(execution[path_key])) != execution[hash_key]:
                raise SystemExit(
                    f"{job['cell_id']}: execution input changed"
                )
    if (
        sha256_file(worker) != plan["worker_sha256"]
        or sha256_file(RUNNER_PATH) != plan["runner_sha256"]
        or sha256_file(Path(plan["python_identity"]["executable"]))
        != plan["python_identity"]["executable_sha256"]
    ):
        raise SystemExit("worker/runner/Python changed before submission")
    manifest = json.loads(json.dumps(plan))
    manifest["approval_sha256"] = plan_sha
    manifest["submitted"] = False
    manifest["submission_atomicity"] = (
        "single_atomic_four_task_array_submission"
        if plan.get("mode") == K40_CS_OVERNIGHT_MODE
        else "all_cells_held_until_every_sbatch_is_accepted"
    )
    _replace_json(root / "campaign.json", manifest)

    # Phase 2: only now may cells be submitted.
    export_values = dict(plan["environment_whitelist"])
    export_values["EVSP_MIP_EXPECTED_WORKER_SHA256"] = plan[
        "worker_sha256"
    ]
    export = ",".join(
        f"{key}={_safe_export_value(key, str(value))}"
        for key, value in export_values.items()
    )
    existing_comments = _existing_execution_comments()
    if plan.get("mode") == K40_CS_OVERNIGHT_MODE:
        planned_comments = {f"MSTATARR:{plan_sha[:30]}"}
    else:
        planned_comments = {
            f"MSTAT:{job['execution_digest'][:32]}"
            for job in plan["jobs"]
        }
    if (
        plan.get("mode") != K40_CS_OVERNIGHT_MODE
        and len(planned_comments) != len(plan["jobs"])
    ):
        raise SystemExit("approved plan contains duplicate execution digests")
    if existing_comments & planned_comments:
        raise SystemExit(
            "an identical execution digest already exists in Slurm; reconcile "
            "that job instead of submitting a duplicate"
        )
    wall_times = [
        _slurm_wall_time(job["budget_hours"])
        for job in plan["jobs"]
    ]
    reservations = _reserve_execution_digests(plan, plan_sha)
    manifest["execution_reservations"] = [
        str(path) for path in reservations
    ]
    _replace_json(root / "campaign.json", manifest)
    if plan.get("mode") == K40_CS_OVERNIGHT_MODE:
        comment = next(iter(planned_comments))
        array_name = "K40R12RG82"
        command = [
            "sbatch", "--parsable", "--array=0-3",
            "--partition=scaglione", "--no-requeue",
            "--signal=B:USR1@180", "--nodes=1", "--ntasks=1",
            "--cpus-per-task=8", "--mem=64G",
            f"--time={max(wall_times)}",
            f"--job-name={array_name}",
            f"--comment={comment}",
            f"--output={logs}/%x_%A_%a.out",
            f"--error={logs}/%x_%A_%a.err",
            "--export=" + export,
            str(worker), str(plan_path), plan_sha, "__ARRAY__",
        ]
        for manifest_job in manifest["jobs"]:
            manifest_job["submission_state"] = "attempting_array"
            manifest_job["deduplication_comment"] = comment
        _replace_json(root / "campaign.json", manifest)
        completed = subprocess.run(
            command, cwd=REPO_ROOT, text=True,
            capture_output=True, check=False,
        )
        array_id = completed.stdout.strip().split(";", 1)[0]
        if completed.returncode != 0 or not array_id.isdigit():
            for manifest_job in manifest["jobs"]:
                manifest_job["submission_state"] = (
                    "array_submit_failed"
                    if completed.returncode != 0
                    else "orphaned_array_unparsed"
                )
                manifest_job["submission_error"] = (
                    completed.stderr or completed.stdout
                ).strip()
            _replace_json(root / "campaign.json", manifest)
            raise SystemExit(
                "atomic four-task array submission failed or returned an "
                "unparseable ID; reservations remain and the execution "
                "comment must be reconciled before any retry"
            )
        for index, manifest_job in enumerate(manifest["jobs"]):
            manifest_job["job_id"] = f"{array_id}_{index}"
            manifest_job["submission_state"] = "submitted_array"
            manifest_job["slurm_array_name"] = array_name
            manifest_job["slurm_array_task_id"] = index
            manifest_job["slurm_display_id"] = f"{array_name}_{index}"
        manifest["submitted"] = True
        _replace_json(root / "campaign.json", manifest)
        return manifest
    held_job_ids = []

    def cancel_known_held_jobs():
        if not held_job_ids:
            return True
        canceled = subprocess.run(
            ["scancel", *held_job_ids],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        return canceled.returncode == 0

    for job, manifest_job, wall_time in zip(
        plan["jobs"], manifest["jobs"], wall_times
    ):
        comment = f"MSTAT:{job['execution_digest'][:32]}"
        command = [
            "sbatch", "--parsable", "--hold", "--partition=scaglione",
            "--no-requeue", "--signal=B:USR1@180",
            "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=64G",
            f"--time={wall_time}",
            f"--job-name={job['job_name']}",
            f"--comment={comment}",
            f"--output={logs}/%x_%j.out",
            f"--error={logs}/%x_%j.err",
            "--export=" + export,
            str(worker), str(plan_path), plan_sha, job["cell_id"],
        ]
        manifest_job["submission_state"] = "attempting"
        manifest_job["deduplication_comment"] = comment
        _replace_json(root / "campaign.json", manifest)
        completed = subprocess.run(
            command, cwd=REPO_ROOT, text=True,
            capture_output=True, check=False,
        )
        if completed.returncode != 0:
            manifest_job["submission_state"] = "failed"
            manifest_job["submission_error"] = (
                completed.stderr or completed.stdout
            ).strip()
            canceled = cancel_known_held_jobs()
            for prior in manifest["jobs"]:
                if prior.get("job_id") in held_job_ids:
                    prior["submission_state"] = (
                        "canceled_before_release"
                        if canceled else "held_cancel_failed"
                    )
            if canceled:
                for reservation in reservations:
                    reservation.unlink(missing_ok=True)
            _replace_json(root / "campaign.json", manifest)
            raise SystemExit(
                f"{job['cell_id']}: held sbatch failed; no job released"
            )
        job_id = completed.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            manifest_job["submission_state"] = "orphaned_held_unparsed"
            canceled = cancel_known_held_jobs()
            for prior in manifest["jobs"]:
                if prior.get("job_id") in held_job_ids:
                    prior["submission_state"] = (
                        "canceled_before_release"
                        if canceled else "held_cancel_failed"
                    )
            _replace_json(root / "campaign.json", manifest)
            raise SystemExit(
                "sbatch returned an invalid job ID; any unknown job remains "
                "held and must be reconciled by its execution comment"
            )
        manifest_job["job_id"] = job_id
        manifest_job["submission_state"] = "held"
        held_job_ids.append(job_id)
        _replace_json(root / "campaign.json", manifest)
    released = subprocess.run(
        ["scontrol", "release", ",".join(held_job_ids)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if released.returncode != 0:
        for manifest_job in manifest["jobs"]:
            manifest_job["submission_state"] = "held_release_failed"
            manifest_job["release_error"] = (
                released.stderr or released.stdout
            ).strip()
        _replace_json(root / "campaign.json", manifest)
        raise SystemExit(
            "atomic job release failed; every campaign job remains held"
        )
    for manifest_job in manifest["jobs"]:
        manifest_job["submission_state"] = "released"
    manifest["submitted"] = True
    _replace_json(root / "campaign.json", manifest)
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=(
            "inventory", "pilot", "secondary",
            "raw_k40", "raw_k40_smoke", K40_CS_OVERNIGHT_MODE,
        ),
        required=True,
    )
    parser.add_argument("--campaign")
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--data-root", type=Path, action="append")
    parser.add_argument("--giro-start", action="append", default=[])
    parser.add_argument(
        "--raw-k40-status",
        action="append",
        default=[],
        help=(
            "Explicit LABEL=SNAPSHOT for R1_CA,R1_CS,R2_CA,R2_CS; "
            "required only in raw_k40 mode."
        ),
    )
    parser.add_argument(
        "--raw-k40-status-sha256", action="append", default=[],
        help="Repeatable LABEL=SHA256; required in raw_k40_smoke mode.",
    )
    parser.add_argument(
        "--raw-k40-journal", action="append", default=[],
        help="Repeatable LABEL=JOURNAL; required in raw_k40_smoke mode.",
    )
    parser.add_argument(
        "--raw-k40-journal-sha256", action="append", default=[],
        help="Repeatable LABEL=SHA256; required in raw_k40_smoke mode.",
    )
    parser.add_argument("--inventory-out", type=Path)
    parser.add_argument("--plan-out", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--reservation-root",
        type=Path,
        default=Path(
            "/share/evsp-dr/mip-statistics-execution-reservations"
        ),
    )
    parser.add_argument("--approved-plan-sha256")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    if args.mode != "inventory" and not args.campaign:
        parser.error("non-inventory modes require --campaign")
    expected_explicit_count = (
        2 if args.mode == K40_CS_OVERNIGHT_MODE
        else 4 if args.mode in RAW_K40_MODES
        else 0
    )
    if (
        args.mode in EXPLICIT_K40_MODES
        and len(args.raw_k40_status) != expected_explicit_count
    ):
        parser.error(
            f"{args.mode} requires {expected_explicit_count} "
            "--raw-k40-status values"
        )
    if args.mode not in EXPLICIT_K40_MODES and args.raw_k40_status:
        parser.error(
            "--raw-k40-status is valid only in explicit k40 modes"
        )
    smoke_counts = (
        len(args.raw_k40_status_sha256),
        len(args.raw_k40_journal),
        len(args.raw_k40_journal_sha256),
    )
    expected_hash_count = (
        4 if args.mode == "raw_k40_smoke"
        else 2 if args.mode == K40_CS_OVERNIGHT_MODE
        else 0
    )
    if (
        args.mode in {"raw_k40_smoke", K40_CS_OVERNIGHT_MODE}
        and smoke_counts
        != (expected_hash_count, expected_hash_count, expected_hash_count)
    ):
        parser.error(
            f"{args.mode} requires {expected_hash_count} status hashes, "
            "journal paths, and journal hashes"
        )
    if (
        args.mode not in {"raw_k40_smoke", K40_CS_OVERNIGHT_MODE}
        and any(smoke_counts)
    ):
        parser.error(
            "explicit RAW hash/journal flags are only valid in hash-bound "
            "k40 modes"
        )
    if (
        args.mode == K40_CS_OVERNIGHT_MODE
        and len(args.giro_start) != 2
    ):
        parser.error(
            "k40_cs_overnight requires R1_CS and R2_CS GIRO40 starts"
        )
    if args.submit and not args.approved_plan_sha256:
        parser.error("--submit requires --approved-plan-sha256")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    roots = dict(DEFAULT_ROOTS)
    roots.update(_parse_assignments(args.root))
    data_roots = args.data_root or [REPO_ROOT / "data"]
    raw_assignments = (
        _parse_assignments(args.raw_k40_status)
        if args.mode in EXPLICIT_K40_MODES else None
    )
    if raw_assignments is not None:
        # Explicit RAW mode validates only the four named immutable inputs.
        # A broad inventory would re-read unrelated large journals and could
        # accidentally make the approved plan depend on unrelated files.
        payload = {
            "schema": "evsp-dr-mip-statistics-explicit-raw-k40-v1",
            "candidates": [],
            "selection_rule": "explicit raw-k40 inputs only",
            "missing_roots": [],
            "missing_slots": [],
        }
    else:
        payload = inventory(roots, data_roots=data_roots)
    if args.inventory_out:
        _write_new_atomic(
            args.inventory_out,
            (json.dumps(payload, indent=2) + "\n").encode(),
        )
    if args.mode == "inventory":
        print(json.dumps(payload, indent=2))
        return 0
    identity = checkout_identity(require_detached=args.submit)
    start_map = _parse_assignments(args.giro_start)
    if args.mode in RAW_K40_MODES and start_map:
        raise SystemExit("raw_k40 mode forbids --giro-start")
    if (
        args.mode == K40_CS_OVERNIGHT_MODE
        and set(start_map) != set(K40_CS_LABELS)
    ):
        raise SystemExit(
            "k40_cs_overnight GIRO starts must be R1_CS and R2_CS"
        )
    explicit_raw_candidates = None
    if args.mode in EXPLICIT_K40_MODES:
        status_hashes = (
            _parse_assignments(
                args.raw_k40_status_sha256, value_type=str
            )
            if args.mode in {
                "raw_k40_smoke", K40_CS_OVERNIGHT_MODE
            } else None
        )
        journal_assignments = (
            _parse_assignments(args.raw_k40_journal)
            if args.mode in {
                "raw_k40_smoke", K40_CS_OVERNIGHT_MODE
            } else None
        )
        journal_hashes = (
            _parse_assignments(
                args.raw_k40_journal_sha256, value_type=str
            )
            if args.mode in {
                "raw_k40_smoke", K40_CS_OVERNIGHT_MODE
            } else None
        )
        explicit_raw_candidates = resolve_raw_k40_candidates(
            raw_assignments,
            data_roots=data_roots,
            enforce_frozen_path=args.mode == "raw_k40",
            expected_status_sha256=status_hashes,
            journal_assignments=journal_assignments,
            expected_journal_sha256=journal_hashes,
            expected_labels=(
                K40_CS_LABELS
                if args.mode == K40_CS_OVERNIGHT_MODE else None
            ),
        )
        payload = dict(payload)
        payload["explicit_raw_k40_candidates"] = {
            label: candidate["candidate_id"]
            for label, candidate in explicit_raw_candidates.items()
        }
    plan = build_plan(
        payload,
        mode=args.mode,
        campaign=args.campaign,
        start_map=start_map,
        identity=identity,
        explicit_raw_candidates=explicit_raw_candidates,
        python_path=args.python,
        reservation_root=args.reservation_root,
    )
    plan_raw = _canonical(plan)
    plan_sha = hashlib.sha256(plan_raw).hexdigest()
    print(json.dumps(plan, indent=2))
    print(f"[approval-sha256] {plan_sha}")
    if args.plan_out:
        _write_new_atomic(args.plan_out, plan_raw)
    if not args.submit:
        print("[dry-run] no Slurm jobs submitted")
        return 0
    if args.approved_plan_sha256 != plan_sha:
        raise SystemExit("current plan differs from approved SHA-256")
    _stage_and_submit(plan, plan_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
