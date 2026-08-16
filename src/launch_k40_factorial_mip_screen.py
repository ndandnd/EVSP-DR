#!/usr/bin/env python3
"""Dry-run/approval-gated strict-MIP screen for completed k40 factorial pools."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from durable_io import flush_and_fsync
from k40_factorial_artifacts import (
    INSTANCE_SHA256,
    PRICES_SHA256,
    validate_campaign,
)
from launch_exact_cg_profile_campaign import (
    _safe_relative,
    _write_new,
    reviewed_checkout_identity,
    sha256_bytes,
    sha256_file,
    validated_python,
)
from run_exact_pool_mip import resolve_pool_journal


REPO_ROOT = Path(__file__).resolve().parents[1]
MIP_CORE_COMMIT = "ae736fbc9c5fef71f39d7d758b7062355c485313"
WORKER_GIT_PATH = "src/submit_k40_factorial_mip_screen.sub"
SCREEN_MARKS = (360, 720, 1440)
TREATMENTS = ("CA", "CS")
MIP_GAP = 1e-4


def _git(*args: str, binary: bool = False):
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=not binary,
    )


def _mip_identity() -> dict:
    identity = reviewed_checkout_identity()
    if _git(
        "merge-base", "--is-ancestor", MIP_CORE_COMMIT, "HEAD"
    ).returncode != 0:
        raise SystemExit("reviewed MIP core is not an ancestor")
    core_paths = (
        "src/run_exact_pool_mip.py",
        "src/master_lp_scipy.py",
        "src/config.py",
        "src/audit_giro_known_columns.py",
        "src/utils_v2.py",
        "src/durable_io.py",
    )
    if _git("diff", "--quiet", MIP_CORE_COMMIT, "--", *core_paths).returncode:
        raise SystemExit("MIP core differs from reviewed commit")
    identity["mip_core_commit"] = MIP_CORE_COMMIT
    return identity


def _mip_python(path: Path) -> dict:
    identity = validated_python(path)
    result = subprocess.run(
        [
            identity["python_executable"], "-c",
            "import gurobipy as gp; "
            "print('.'.join(map(str, gp.gurobi.version())))",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit("MIP Python environment lacks gurobipy")
    identity["gurobi_version"] = result.stdout.strip()
    encoded = (
        identity["identity_sha256"] + "|" + identity["gurobi_version"]
    ).encode()
    identity["mip_identity_sha256"] = hashlib.sha256(encoded).hexdigest()
    return identity


def _worker_bytes(commit: str) -> bytes:
    result = _git("show", f"{commit}:{WORKER_GIT_PATH}", binary=True)
    if result.returncode != 0 or not result.stdout:
        raise SystemExit("cannot read reviewed MIP worker blob")
    return result.stdout


def _parse_replicates(values: list[str]) -> dict[str, Path]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise SystemExit("--replicate must be R1=/path or R2=/path")
        label, path = value.split("=", 1)
        label = label.upper()
        if label not in {"R1", "R2"} or label in parsed:
            raise SystemExit("replicates must be distinct R1 and R2")
        parsed[label] = Path(path).expanduser().resolve()
    if set(parsed) != {"R1", "R2"}:
        raise SystemExit("exactly R1 and R2 replicate paths are required")
    return parsed


def _find_input(snapshot: Path, relative: Path) -> Path:
    candidates = [REPO_ROOT / "data" / relative]
    candidates.extend(
        parent / "data" / relative for parent in snapshot.parents
    )
    for candidate in dict.fromkeys(path.resolve() for path in candidates):
        if candidate.is_file():
            return candidate
    raise SystemExit(f"cannot locate data/{relative} for {snapshot}")


def _validate_start(
    path: Path,
    trip_ids: list[int],
    *,
    require_attestation: bool = True,
) -> dict:
    path = path.expanduser().resolve()
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise SystemExit("validated start is not a JSON object")
    routes = payload.get("routes")
    if not isinstance(routes, list) or not routes:
        raise SystemExit("validated start has no routes")
    if payload.get("infeasible") not in (None, []):
        raise SystemExit("validated start contains infeasible/partial routes")
    if payload.get("source") != "rerealized":
        raise SystemExit("validated start is not marked re-realized")
    counts = {trip: 0 for trip in trip_ids}
    for ordinal, route in enumerate(routes, start=1):
        if not isinstance(route, dict):
            raise SystemExit(f"start route {ordinal} is not an object")
        nodes = route.get("route", route.get("route_nodes"))
        if not isinstance(nodes, list):
            raise SystemExit(f"start route {ordinal} has no route nodes")
        trips = [
            node for node in nodes
            if isinstance(node, int) and not isinstance(node, bool)
        ]
        if not trips or len(trips) != len(set(trips)):
            raise SystemExit(f"start route {ordinal} has invalid trips")
        for trip in trips:
            if trip not in counts:
                raise SystemExit(f"start route {ordinal} has unknown trip {trip}")
            counts[trip] += 1
    bad = {trip: count for trip, count in counts.items() if count != 1}
    if bad:
        raise SystemExit(
            f"validated start is not an exact partition: "
            f"{list(bad.items())[:15]}"
        )
    physics = payload.get("physics") or {}
    if (float(physics.get("g_kwh", math.nan)) != 300.0
            or float(physics.get("charge_kw", math.nan)) != 300.0
            or float(physics.get("reserve_frac", math.nan)) != 0.0):
        raise SystemExit("validated start physics mismatch")
    if Path(str(payload.get("prices_csv"))).name != "hourly_prices_flat.csv":
        raise SystemExit("validated start tariff mismatch")
    attestation = payload.get("_factorial_start_provenance")
    if require_attestation:
        if not isinstance(attestation, dict):
            raise SystemExit("validated start lacks preparation attestation")
        if (
            attestation.get("schema")
            != "evsp-dr-k40-factorial-giro-start-v1"
            or attestation.get("mip_core_commit") != MIP_CORE_COMMIT
            or attestation.get("instance_sha256") != INSTANCE_SHA256
            or attestation.get("prices_sha256") != PRICES_SHA256
            or int(attestation.get("bus_count", -1)) != len(routes)
            or int(attestation.get("trip_count", -1)) != len(trip_ids)
        ):
            raise SystemExit("validated start attestation mismatch")
        for key in ("snapshot_sha256", "journal_sha256"):
            value = str(attestation.get(key) or "")
            if (
                len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise SystemExit(
                    f"validated start has invalid attested {key}"
                )
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bus_count": len(routes),
        "trip_count": len(trip_ids),
        "attestation": attestation,
        "raw": raw,
    }


def _cell_name(rep: str, arm: str, mark: int, budget: int) -> str:
    age = {360: "06", 720: "12", 1440: "24"}[mark]
    budget_tag = "M30" if budget == 1800 else "H02"
    name = f"K{rep[-1]}{arm}{age}{budget_tag}"
    if len(name) > 15:
        raise SystemExit(f"MIP job name exceeds 15 characters: {name}")
    return name


def _selected_cells(args, campaigns: dict) -> list[tuple[str, str, int]]:
    all_cells = [
        (rep, arm, mark)
        for rep in ("R1", "R2")
        for arm in TREATMENTS
        for mark in SCREEN_MARKS
    ]
    if args.mode == "screen":
        if args.cell:
            raise SystemExit("--cell is escalation-only")
        return all_cells
    if not args.cell:
        raise SystemExit("escalation mode requires at least one --cell")
    selected = []
    for raw in args.cell:
        try:
            rep, arm, mark_text = raw.upper().split(":")
            mark = int(mark_text.lstrip("M"))
        except (ValueError, TypeError) as exc:
            raise SystemExit(
                "--cell must be R1:CA:M360 (or another primary cell)"
            ) from exc
        cell = (rep, arm, mark)
        if cell not in all_cells or cell in selected:
            raise SystemExit(f"invalid/duplicate escalation cell {raw}")
        selected.append(cell)
    return selected


def _canonical_plan(payload: dict) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()


def launch(args) -> dict:
    identity = _mip_identity()
    python_identity = _mip_python(args.python)
    replicates = _parse_replicates(args.replicate)
    if len(set(replicates.values())) != 2:
        raise SystemExit("replicate campaign directories must be distinct")
    validated = {
        rep: validate_campaign(path, replicate=rep)
        for rep, path in replicates.items()
    }
    first_rows = validated["R1"]["rows"]
    trip_ids = json.loads(
        Path(first_rows[0]["status_path"]).read_text()
    )["trip_ids"]
    start = _validate_start(args.validated_start, trip_ids)
    if (
        start["attestation"].get("reviewed_checkout_commit")
        != identity["expected_commit"]
    ):
        raise SystemExit(
            "validated start was prepared by a different checkout commit"
        )
    if start["bus_count"] != 40:
        raise SystemExit(
            f"validated GIRO start must contain 40 buses, found "
            f"{start['bus_count']}"
        )
    campaign = args.campaign or (
        "k40_factorial_mip_"
        + dt.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    )
    if campaign in {".", ".."} or not all(
            char.isalnum() or char in "_.-" for char in campaign):
        raise SystemExit("invalid MIP campaign name")
    root = REPO_ROOT / "src/results/k40_factorial_mip" / campaign
    logs = REPO_ROOT / "src/logs/k40_factorial_mip" / campaign
    if root.exists() or logs.exists():
        raise SystemExit("MIP campaign exists; retries need a fresh name")
    budget = 1800 if args.mode == "screen" else 7200
    cells = _selected_cells(args, validated)
    worker = root / "input/submit_k40_factorial_mip_screen.sub"
    worker_bytes = _worker_bytes(identity["expected_commit"])
    worker_sha = sha256_bytes(worker_bytes)
    for label, value in (
        ("repository path", str(REPO_ROOT)),
        ("Python path", python_identity["python_executable"]),
    ):
        if "," in value:
            raise SystemExit(
                f"{label} contains a comma and is unsafe for Slurm export"
            )
    shared_start = root / "input/common/validated_start.json"
    plans = []
    for rep, arm, mark in cells:
        row = next(
            row for row in validated[rep]["rows"]
            if row["arm"] == arm and row["checkpoint"] == f"m{mark}"
        )
        source_result = Path(row["status_path"])
        status_raw = source_result.read_bytes()
        status = json.loads(status_raw)
        if status.get("trip_ids") != trip_ids:
            raise SystemExit(
                f"{rep}/{arm}/m{mark}: trip set differs across cells"
            )
        source_journal = resolve_pool_journal(source_result, status).resolve()
        csv_relative = _safe_relative(status["csv"], "csv")
        prices_relative = _safe_relative(status["prices_csv"], "prices")
        source_instance = _find_input(source_result, csv_relative)
        source_prices = _find_input(source_result, prices_relative)
        hashes = {
            "result": sha256_bytes(status_raw),
            "journal": sha256_file(source_journal),
            "instance": sha256_file(source_instance),
            "prices": sha256_file(source_prices),
        }
        if (hashes["instance"] != INSTANCE_SHA256
                or hashes["prices"] != PRICES_SHA256):
            raise SystemExit(f"{rep}/{arm}/m{mark}: identity hash mismatch")
        label = f"{rep}_{arm}_m{mark}"
        cell_root = root / "input" / label
        staged_result = cell_root / source_result.name
        staged_journal = cell_root / source_journal.name
        staged_instance = cell_root / "data" / csv_relative
        staged_prices = cell_root / "data" / prices_relative
        output = root / "outputs" / f"{label}.mip.json"
        # Snapshot bytes remain exact. resolve_pool_journal intentionally
        # prefers the staged sibling for immutable *.snapshot.json files.
        staged_status_raw = status_raw
        spec = {
            "label": label,
            "replicate": rep,
            "treatment": arm,
            "snapshot_mark_minutes": mark,
            "staged_result": str(staged_result),
            "staged_result_sha256": hashes["result"],
            "staged_journal": str(staged_journal),
            "staged_journal_sha256": hashes["journal"],
            "staged_instance": str(staged_instance),
            "staged_instance_sha256": hashes["instance"],
            "csv": str(csv_relative),
            "staged_prices": str(staged_prices),
            "staged_prices_sha256": hashes["prices"],
            "prices_csv": str(prices_relative),
            "staged_start": str(shared_start),
            "staged_start_sha256": start["sha256"],
            "output": str(output),
            "time_limit_s": budget,
            "threads": 8,
            "mip_gap": MIP_GAP,
            "expected_commit": identity["expected_commit"],
            "mip_core_commit": MIP_CORE_COMMIT,
        }
        spec_raw = (json.dumps(spec, indent=2) + "\n").encode()
        spec_path = cell_root / "job.json"
        job_name = _cell_name(rep, arm, mark, budget)
        comment = (
            f"EVSPK40MIP:{campaign}:{label}:{hashes['result'][:12]}"
        )
        export = (
            "HOME=" + os.environ.get("HOME", str(Path.home()))
            + ",USER=" + os.environ.get("USER", "")
            + ",PATH=/usr/local/bin:/usr/bin:/bin"
            + ",EVSP_DR_ROOT=" + str(REPO_ROOT)
            + ",EVSP_EXPECTED_COMMIT=" + identity["expected_commit"]
            + ",EVSP_MIP_PYTHON=" + python_identity["python_executable"]
            + ",EVSP_MIP_ENV_SHA256="
            + python_identity["mip_identity_sha256"]
            + ",EVSP_MIP_EXPECTED_WORKER_SHA256=" + worker_sha
        )
        command = [
            "sbatch", "--parsable", "--partition=scaglione",
            "--no-requeue", "--nodes=1", "--ntasks=1",
            "--cpus-per-task=8", "--mem=40G",
            f"--time={'00:40:00' if budget == 1800 else '02:10:00'}",
            f"--job-name={job_name}", f"--comment={comment}",
            f"--output={logs}/%x_%j.out",
            f"--error={logs}/%x_%j.err",
            "--export=" + export,
            str(worker), str(spec_path), sha256_bytes(spec_raw),
        ]
        plans.append({
            "label": label,
            "replicate": rep,
            "treatment": arm,
            "snapshot_mark_minutes": mark,
            "source_result": str(source_result),
            "source_journal": str(source_journal),
            "source_instance": str(source_instance),
            "source_prices": str(source_prices),
            "source_hashes": hashes,
            "staged_result": str(staged_result),
            "staged_journal": str(staged_journal),
            "staged_instance": str(staged_instance),
            "staged_prices": str(staged_prices),
            "staged_result_bytes": staged_status_raw,
            "spec": spec,
            "spec_path": str(spec_path),
            "spec_bytes": spec_raw,
            "spec_sha256": sha256_bytes(spec_raw),
            "output": str(output),
            "job_name": job_name,
            "slurm_comment": comment,
            "command": command,
            "job_id": None,
            "submission_state": "planned",
        })
    manifest = {
        "schema": "evsp-dr-k40-factorial-mip-campaign-v1",
        "campaign": campaign,
        "mode": args.mode,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "submitted": False,
        "checkout_identity": identity,
        "mip_core_commit": MIP_CORE_COMMIT,
        "python": python_identity,
        "worker": str(worker),
        "worker_sha256": worker_sha,
        "validated_start": {
            key: value for key, value in start.items() if key != "raw"
        },
        "budget_seconds": budget,
        "jobs": plans,
    }
    approved = {
        **{key: value for key, value in manifest.items()
           if key not in {"created_at", "submitted", "jobs"}},
        "jobs": [{
            key: value for key, value in plan.items()
            if key not in {
                "staged_result_bytes", "spec_bytes",
                "job_id", "submission_state", "submission_error",
                "reconciled_slurm_state",
            }
        } for plan in plans],
    }
    approval_raw = _canonical_plan(approved)
    approval_sha = hashlib.sha256(approval_raw).hexdigest()
    manifest["approval_sha256"] = approval_sha
    for plan in plans:
        print("[k40-mip-plan]", " ".join(plan["command"]))
    if not args.submit:
        print("[approval-plan]")
        print(json.dumps(approved, indent=2))
        print(f"[approval-sha256] {approval_sha}")
        if args.plan_out:
            _write_new(args.plan_out.expanduser().resolve(), approval_raw)
        print("[dry-run] no MIP jobs submitted")
        return manifest
    if args.approved_plan_sha256 != approval_sha:
        raise SystemExit("current MIP plan differs from approved SHA-256")

    root.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    _write_new(worker, worker_bytes, executable=True)
    _write_new(shared_start, start["raw"])
    if sha256_file(shared_start) != start["sha256"]:
        raise SystemExit("staged validated start hash mismatch")
    for plan in plans:
        _write_new(Path(plan["staged_result"]), plan["staged_result_bytes"])
        _write_new(
            Path(plan["staged_journal"]),
            Path(plan["source_journal"]).read_bytes(),
        )
        _write_new(
            Path(plan["staged_instance"]),
            Path(plan["source_instance"]).read_bytes(),
        )
        _write_new(
            Path(plan["staged_prices"]),
            Path(plan["source_prices"]).read_bytes(),
        )
        _write_new(Path(plan["spec_path"]), plan["spec_bytes"])
        for source_key, staged_key, hash_key in (
            ("source_journal", "staged_journal", "journal"),
            ("source_instance", "staged_instance", "instance"),
            ("source_prices", "staged_prices", "prices"),
        ):
            if sha256_file(Path(plan[staged_key])) != plan["source_hashes"][hash_key]:
                raise SystemExit(f"{plan['label']}: staged hash mismatch")
            if sha256_file(Path(plan[source_key])) != plan["source_hashes"][hash_key]:
                raise SystemExit(f"{plan['label']}: source changed")
        if sha256_file(Path(plan["source_result"])) != plan["source_hashes"]["result"]:
            raise SystemExit(f"{plan['label']}: source result changed")
        if (sha256_file(Path(plan["staged_result"]))
                != plan["spec"]["staged_result_sha256"]):
            raise SystemExit(f"{plan['label']}: staged result hash mismatch")
        if sha256_file(Path(plan["spec_path"])) != plan["spec_sha256"]:
            raise SystemExit(f"{plan['label']}: staged spec hash mismatch")
        plan.pop("staged_result_bytes")
        plan.pop("spec_bytes")
    manifest_path = root / "campaign.json"
    _write_new(
        manifest_path, (json.dumps(manifest, indent=2) + "\n").encode()
    )
    for plan in plans:
        plan["submission_state"] = "attempting"
        _replace_manifest(manifest_path, manifest)
        _write_new(
            root / f".{plan['label']}.attempt.json",
            (json.dumps({
                "label": plan["label"],
                "job_name": plan["job_name"],
                "slurm_comment": plan["slurm_comment"],
                "command": plan["command"],
            }, indent=2) + "\n").encode(),
        )
        pre_submit_identity = _mip_identity()
        staged_inputs_clean = (
            sha256_file(Path(plan["staged_result"]))
            == plan["spec"]["staged_result_sha256"]
            and sha256_file(Path(plan["staged_journal"]))
            == plan["source_hashes"]["journal"]
            and sha256_file(Path(plan["staged_instance"]))
            == plan["source_hashes"]["instance"]
            and sha256_file(Path(plan["staged_prices"]))
            == plan["source_hashes"]["prices"]
        )
        if (
            pre_submit_identity["expected_commit"]
            != identity["expected_commit"]
            or sha256_file(worker) != worker_sha
            or sha256_file(Path(plan["spec_path"])) != plan["spec_sha256"]
            or sha256_file(shared_start) != start["sha256"]
            or not staged_inputs_clean
        ):
            plan["submission_state"] = "failed"
            plan["submission_error"] = (
                "checkout/worker/spec/start changed before sbatch"
            )
            _replace_manifest(manifest_path, manifest)
            raise SystemExit(plan["submission_error"])
        plan["pre_submission_observed_git_commit"] = (
            pre_submit_identity["observed_commit"]
        )
        completed = subprocess.run(
            plan["command"], text=True, capture_output=True, check=False,
            cwd=REPO_ROOT,
        )
        if completed.returncode != 0:
            plan["submission_state"] = "failed"
            plan["submission_error"] = (
                completed.stderr or completed.stdout
            ).strip()
            _replace_manifest(manifest_path, manifest)
            raise SystemExit(f"{plan['label']}: sbatch failed")
        job_id = completed.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            plan["submission_state"] = "failed"
            plan["submission_error"] = "sbatch returned an invalid job ID"
            _replace_manifest(manifest_path, manifest)
            raise SystemExit(plan["submission_error"])
        plan["job_id"] = job_id
        plan["submission_state"] = "submitted"
        _replace_manifest(manifest_path, manifest)
    manifest["submitted"] = True
    _replace_manifest(manifest_path, manifest)
    print(f"[submitted] {manifest_path}")
    return manifest


def _replace_manifest(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        flush_and_fsync(handle)
    os.replace(temporary, path)


def _parse_sha(value: str) -> str:
    value = value.lower()
    if (len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise argparse.ArgumentTypeError("expected 64-character SHA-256")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replicate", action="append", required=True,
        help="R1=/campaign/dir and R2=/campaign/dir",
    )
    parser.add_argument("--validated-start", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("screen", "escalation"), default="screen"
    )
    parser.add_argument(
        "--cell", action="append",
        help="Escalation cell R1:CA:M360 (repeatable).",
    )
    parser.add_argument("--campaign")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--plan-out", type=Path)
    parser.add_argument("--approved-plan-sha256", type=_parse_sha)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    if args.submit and not args.approved_plan_sha256:
        parser.error("--submit requires --approved-plan-sha256")
    return args


def main(argv=None) -> int:
    launch(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
