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
import subprocess
import sys
import tempfile
from pathlib import Path

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
)
DEFAULT_ROOTS = {
    "repool_small": REPO_ROOT / "results/repool_small",
    "exact_big": REPO_ROOT / "results/exact_big",
    "k40_factorial": REPO_ROOT / "results/k40_factorial",
    "bigtar_snapshots": REPO_ROOT / "results/bigtar_snapshots",
    "fresh_preparation": REPO_ROOT / "results/mip_statistics_prep",
}

RAW_K40_BUDGET_HOURS = 8
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
    return {
        "expected_commit": commit,
        "reviewed_base_commit": REVIEWED_BASE,
        "detached": detached,
        "branch": branch,
        "tracked_clean": True,
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
) -> dict[str, dict]:
    """Validate the four explicit, non-GIRO k40 factorial snapshots."""

    expected_labels = set(RAW_K40_SPECS)
    if set(assignments) != expected_labels:
        missing = sorted(expected_labels - set(assignments))
        extra = sorted(set(assignments) - expected_labels)
        raise ValueError(
            "raw-k40 requires exactly R1_CA,R1_CS,R2_CA,R2_CS; "
            f"missing={missing}, extra={extra}"
        )
    resolved_paths = [
        path.expanduser().resolve() for path in assignments.values()
    ]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("raw-k40 snapshot paths must be distinct")

    selected = {}
    for label, spec in RAW_K40_SPECS.items():
        path = assignments[label].expanduser().resolve()
        if (
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
        candidate["inventory_replicate"] = candidate.get("replicate")
        candidate["replicate"] = spec["replicate"]
        candidate["raw_k40_label"] = label
        selected[label] = candidate

    identity_fields = (
        "instance_sha256", "tariff_sha256", "trip_set_sha256", "trip_count",
        "csv", "prices_csv",
    )
    if any(
        selected[label].get(field) != selected["R1_CA"].get(field)
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
        or payload.get("infeasible") != []
        or payload.get("source") != "rerealized"
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
    )
    return detail


def _rep_code(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", value).upper()
    return (cleaned or "X")[-2:].rjust(2, "0")


def _job_name(job, campaign: str) -> str:
    if job.get("matrix") == "raw_k40":
        label = str(job["source"]["raw_k40_label"]).replace("_", "")
        nonce = hashlib.sha256(
            f"{campaign}|{job['cell_id']}".encode()
        ).hexdigest()[:2].upper()
        name = f"RK40{label}8H{nonce}"
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
    if arm == "GIRO":
        start_path = start_map.get(str(candidate["scale"]))
        if start_path is None:
            blocked.append("validated_giro_start_missing")
        else:
            try:
                start = _validated_start(start_path, candidate)
            except (OSError, ValueError) as exc:
                blocked.append(f"validated_giro_start_invalid: {exc}")
    if matrix == "raw_k40":
        cell = (
            f"k40_{candidate['raw_k40_label'].lower()}_raw_m1440"
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
        "augmentation_changes_column_set": arm == "GIRO",
        "partitioning": "strict_exact_once",
        "two_stage": True,
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
    elif mode == "raw_k40":
        if set(explicit_raw_candidates or {}) != set(RAW_K40_SPECS):
            raise ValueError("raw_k40 mode requires four explicit candidates")
        preparations = []
        for label in RAW_K40_SPECS:
            candidate = explicit_raw_candidates[label]
            jobs.append(_job_from_candidate(
                candidate,
                arm="RAW",
                budget_hours=RAW_K40_BUDGET_HOURS,
                start_map={},
                matrix="raw_k40",
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
            if mode == "raw_k40"
            else inventory_payload["selection_rule"]
        ),
        "selected_candidates": (
            {
                label: candidate["candidate_id"]
                for label, candidate in (explicit_raw_candidates or {}).items()
            }
            if mode == "raw_k40"
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
        },
        "worker": str(WORKER_PATH),
        "worker_sha256": worker_sha,
        "runner": str(RUNNER_PATH),
        "runner_sha256": runner_sha,
        "code_hashes": code_hashes,
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
                mode != "raw_k40"
                and bool(inventory_payload.get("missing_roots"))
            )
            or (
                mode != "raw_k40"
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
                "budget_seconds": RAW_K40_BUDGET_HOURS * 3600,
                "expected_trip_count": 947,
                "expected_snapshot_minutes": 1440,
            }
            if mode == "raw_k40" else None
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
    planned_comments = {
        f"MSTAT:{job['execution_digest'][:32]}"
        for job in plan["jobs"]
    }
    if len(planned_comments) != len(plan["jobs"]):
        raise SystemExit("approved plan contains duplicate execution digests")
    if existing_comments & planned_comments:
        raise SystemExit(
            "an identical execution digest already exists in Slurm; reconcile "
            "that job instead of submitting a duplicate"
        )
    reservations = _reserve_execution_digests(plan, plan_sha)
    manifest["execution_reservations"] = [
        str(path) for path in reservations
    ]
    _replace_json(root / "campaign.json", manifest)
    for job, manifest_job in zip(plan["jobs"], manifest["jobs"]):
        wall_hours = job["budget_hours"]
        comment = f"MSTAT:{job['execution_digest'][:32]}"
        command = [
            "sbatch", "--parsable", "--partition=scaglione",
            "--no-requeue", "--signal=B:USR1@180",
            "--nodes=1", "--ntasks=1", "--cpus-per-task=8", "--mem=64G",
            f"--time={wall_hours:02d}:10:00",
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
            _replace_json(root / "campaign.json", manifest)
            raise SystemExit(f"{job['cell_id']}: sbatch failed")
        job_id = completed.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise SystemExit("sbatch returned an invalid job ID")
        manifest_job["job_id"] = job_id
        manifest_job["submission_state"] = "submitted"
        _replace_json(root / "campaign.json", manifest)
    manifest["submitted"] = True
    _replace_json(root / "campaign.json", manifest)
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("inventory", "pilot", "secondary", "raw_k40"),
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
    if args.mode == "raw_k40" and len(args.raw_k40_status) != 4:
        parser.error("raw_k40 mode requires four --raw-k40-status values")
    if args.mode != "raw_k40" and args.raw_k40_status:
        parser.error("--raw-k40-status is valid only in raw_k40 mode")
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
        if args.mode == "raw_k40" else None
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
    if args.mode == "raw_k40" and start_map:
        raise SystemExit("raw_k40 mode forbids --giro-start")
    explicit_raw_candidates = None
    if args.mode == "raw_k40":
        explicit_raw_candidates = resolve_raw_k40_candidates(
            raw_assignments,
            data_roots=data_roots,
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
