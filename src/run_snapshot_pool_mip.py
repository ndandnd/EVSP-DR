"""Run one matched two-stage MIP on an immutable exact-CG snapshot.

The worker builds a GIRO partition seed for the snapshot's exact trip set,
re-realizes that seed under the snapshot physics/tariff, and then solves the
strict partition MIP.  Every snapshot gets private seed files, avoiding cache
and concurrent-write ambiguity in a many-job stopping-rule campaign.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from durable_io import atomic_write_json, valid_json_object
from run_exact_pool_mip import file_sha256, git_value, resolve_pool_journal

SRC = Path(__file__).resolve().parent
DATA_DIR = SRC.parent / "data"


def run(command):
    print("[MIP-SNAPSHOT] " + " ".join(str(value) for value in command),
          flush=True)
    return subprocess.run(command, cwd=SRC).returncode


def _load_route_cache(path: Path, expected: dict) -> dict | None:
    if not valid_json_object(path, ("routes", "_snapshot_worker_provenance")):
        return None
    with open(path) as fh:
        payload = json.load(fh)
    if not isinstance(payload.get("routes"), list):
        return None
    if payload.get("infeasible"):
        return None
    if payload.get("_snapshot_worker_provenance") != expected:
        return None
    return payload


def _generate_route_cache(command, temporary: Path, destination: Path,
                          expected: dict, accepted_codes=(0,)) -> int:
    rc = run([*command, "--out", str(temporary)])
    if rc not in accepted_codes or not valid_json_object(temporary, ("routes",)):
        raise SystemExit(f"route-cache generation failed with rc={rc}")
    with open(temporary) as fh:
        payload = json.load(fh)
    if not isinstance(payload.get("routes"), list):
        raise SystemExit("route-cache generator returned a non-list routes field")
    if payload.get("infeasible"):
        raise SystemExit(
            "route-cache generator returned an incomplete route set; "
            f"{len(payload['infeasible'])} GIRO route(s) were infeasible"
        )
    payload["_snapshot_worker_provenance"] = expected
    atomic_write_json(destination, payload)
    temporary.unlink(missing_ok=True)
    return rc


def _verify_snapshot_problem_inputs(status: dict, data_dir: Path | None = None):
    """Require the working data bytes recorded by the immutable snapshot."""

    data_dir = DATA_DIR if data_dir is None else Path(data_dir)
    provenance = status.get("provenance") or {}
    checks = (
        ("csv", "instance_sha256"),
        ("prices_csv", "prices_sha256"),
    )
    verified = {}
    for status_key, hash_key in checks:
        recorded_path = status.get(status_key)
        expected_sha = provenance.get(hash_key)
        if not recorded_path or not expected_sha:
            raise SystemExit(
                f"snapshot lacks required {status_key}/{hash_key} provenance"
            )
        path = (data_dir / str(recorded_path)).resolve()
        try:
            path.relative_to(data_dir.resolve())
        except ValueError as exc:
            raise SystemExit(
                f"snapshot input is outside the repository data directory: "
                f"{recorded_path}"
            ) from exc
        if not path.is_file():
            raise SystemExit(f"snapshot input is unavailable: {path}")
        actual_sha = file_sha256(path)
        if actual_sha != expected_sha:
            raise SystemExit(
                f"snapshot input hash mismatch for {path}: expected "
                f"{expected_sha}, found {actual_sha}"
            )
        verified[hash_key] = actual_sha
    return verified


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timelimit", type=int, default=3600)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)

    snapshot_sha = file_sha256(args.snapshot)
    with open(args.snapshot) as fh:
        status = json.load(fh)
    source_journal = resolve_pool_journal(args.snapshot, status)
    source_journal_sha = file_sha256(source_journal)
    if valid_json_object(
            args.out, ("status_name", "buses", "source_result_sha256",
                       "source_journal_sha256")):
        with open(args.out) as fh:
            existing_result = json.load(fh)
        existing_provenance = existing_result.get("mip_provenance") or {}
        if (existing_result.get("source_result_sha256") == snapshot_sha
                and existing_result.get("source_journal_sha256")
                == source_journal_sha
                and existing_provenance.get("git_commit")
                == git_value("rev-parse", "HEAD")):
            print(f"[MIP-SNAPSHOT] {args.out} is a complete result for this "
                  "snapshot — keeping it")
            return 0
    instance = status.get("csv")
    prices = status.get("prices_csv")
    if not instance or not prices:
        raise SystemExit("snapshot lacks csv/prices_csv provenance")
    input_hashes = _verify_snapshot_problem_inputs(status)

    inputs = args.out.parent / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    seed = inputs / f"{args.out.stem}.giro_seed.json"
    realized = inputs / f"{args.out.stem}.giro_seed_rrz.json"
    seed_expected = {
        "kind": "giro_seed",
        "worker_commit": git_value("rev-parse", "HEAD"),
        "snapshot_sha256": snapshot_sha,
        "source_journal_sha256": source_journal_sha,
        **input_hashes,
        "instance": str(instance),
    }
    realized_expected = {
        "kind": "rerealized_giro_seed",
        "worker_commit": git_value("rev-parse", "HEAD"),
        "snapshot_sha256": snapshot_sha,
        "source_journal_sha256": source_journal_sha,
        **input_hashes,
        "instance": str(instance),
        "prices": str(prices),
        "g_kwh": status.get("g_kwh", 300.0),
        "charge_kw": status.get("charge_kw", 300.0),
        "min_soc_frac": status.get("min_soc_frac", 0.0),
    }

    if _load_route_cache(seed, seed_expected) is None:
        _generate_route_cache([
            sys.executable, "-u", str(SRC / "make_giro_seed_routes.py"),
            "--instance", str(instance),
        ], Path(str(seed) + ".generator_tmp"), seed, seed_expected)
    if _load_route_cache(realized, realized_expected) is None:
        _generate_route_cache([
            sys.executable, "-u", str(SRC / "rerealize_routes.py"),
            "--routes", str(seed),
            "--physics-from", str(args.snapshot),
            "--instance", str(instance),
            "--prices", str(prices),
        ], Path(str(realized) + ".generator_tmp"), realized,
            realized_expected)

    if (file_sha256(args.snapshot) != snapshot_sha
            or file_sha256(source_journal) != source_journal_sha
            or _verify_snapshot_problem_inputs(status) != input_hashes):
        raise SystemExit(
            "snapshot, journal, or problem inputs changed while preparing "
            "the common GIRO seed; retry from immutable campaign inputs"
        )

    return run([
        sys.executable, "-u", str(SRC / "run_exact_pool_mip.py"),
        "--result", str(args.snapshot),
        "--extra-routes", str(realized),
        "--two-stage",
        "--timelimit", str(args.timelimit),
        "--mipgap", "0.000001",
        "--threads", str(args.threads),
        "--out", str(args.out),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
