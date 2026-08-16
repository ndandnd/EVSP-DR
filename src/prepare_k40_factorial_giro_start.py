#!/usr/bin/env python3
"""Build one no-clobber, re-realized, physically validated k40 GIRO partition."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from k40_factorial_artifacts import (
    INSTANCE_SHA256,
    PRICES_SHA256,
    sha256_file,
)
from launch_k40_factorial_mip_screen import _mip_identity, _validate_start
from install_exact_cg_profile_input import install
from run_exact_pool_mip import resolve_pool_journal
from run_snapshot_pool_mip import _verify_snapshot_problem_inputs


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, environment=None) -> None:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"GIRO start preparation command failed ({completed.returncode}): "
            + " ".join(command)
        )


def prepare(snapshot: Path, output: Path, python: Path) -> dict:
    identity = _mip_identity()
    snapshot = snapshot.expanduser().resolve()
    output = output.expanduser().resolve()
    if not snapshot.name.endswith(".snapshot.json"):
        raise SystemExit("GIRO start source must be an immutable snapshot")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite GIRO start: {output}")
    status_raw = snapshot.read_bytes()
    status = json.loads(status_raw)
    source_journal = resolve_pool_journal(snapshot, status).resolve()
    expected_journal = Path(str(snapshot) + ".columns.jsonl").resolve()
    if source_journal != expected_journal:
        raise SystemExit("snapshot/journal pairing mismatch")
    source_hashes = {
        "snapshot_sha256": sha256_file(snapshot),
        "journal_sha256": sha256_file(source_journal),
    }
    data_root = REPO_ROOT / "data"
    relative_instance = Path(str(status.get("csv") or ""))
    relative_prices = Path(str(status.get("prices_csv") or ""))
    if (
        relative_instance.is_absolute()
        or ".." in relative_instance.parts
        or relative_prices.is_absolute()
        or ".." in relative_prices.parts
    ):
        raise SystemExit("snapshot contains unsafe data paths")

    def locate(relative: Path) -> Path:
        for parent in snapshot.parents:
            candidate = parent / "data" / relative
            if candidate.is_file():
                return candidate.resolve()
        raise SystemExit(f"cannot locate source data/{relative}")

    source_instance = locate(relative_instance)
    source_prices = locate(relative_prices)
    if (
        sha256_file(source_instance) != INSTANCE_SHA256
        or sha256_file(source_prices) != PRICES_SHA256
    ):
        raise SystemExit("source snapshot data hashes mismatch")
    install(
        source_instance, data_root, relative_instance, INSTANCE_SHA256
    )
    install(
        source_prices, data_root, relative_prices, PRICES_SHA256
    )
    input_hashes = _verify_snapshot_problem_inputs(
        status, data_dir=data_root
    )
    if (
        input_hashes.get("instance_sha256") != INSTANCE_SHA256
        or input_hashes.get("prices_sha256") != PRICES_SHA256
    ):
        raise SystemExit("snapshot is not the intended k40-r2/flat problem")
    if not isinstance(status.get("trip_ids"), list):
        raise SystemExit("snapshot has no trip IDs")
    python = python.expanduser().resolve()
    if not python.is_file():
        raise SystemExit(f"Python executable is unavailable: {python}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            dir=output.parent, prefix=f".{output.name}.prepare.") as tmp_text:
        temporary = Path(tmp_text)
        seed = temporary / "giro_seed.json"
        realized = temporary / "giro_seed_rerealized.json"
        candidate = temporary / "validated_giro_start.json"
        _run([
            str(python), "-u",
            str(REPO_ROOT / "src/make_giro_seed_routes.py"),
            "--instance", str(status["csv"]),
            "--out", str(seed),
        ])
        _run([
            str(python), "-u",
            str(REPO_ROOT / "src/rerealize_routes.py"),
            "--routes", str(seed),
            "--physics-from", str(snapshot),
            "--instance", str(status["csv"]),
            "--prices", str(status["prices_csv"]),
            "--out", str(realized),
        ])
        payload = json.loads(realized.read_text())
        validation = _validate_start(
            realized, status["trip_ids"], require_attestation=False
        )
        if validation["bus_count"] != 40:
            raise SystemExit(
                f"GIRO start has {validation['bus_count']} buses, expected 40"
            )
        payload["_factorial_start_provenance"] = {
            "schema": "evsp-dr-k40-factorial-giro-start-v1",
            "reviewed_checkout_commit": identity["expected_commit"],
            "mip_core_commit": identity["mip_core_commit"],
            "runner_sha256": identity["run_exact_pool_mip_sha256"],
            **source_hashes,
            **input_hashes,
            "source_snapshot": str(snapshot),
            "source_journal": str(source_journal),
            "bus_count": 40,
            "trip_count": len(status["trip_ids"]),
        }
        candidate.write_text(json.dumps(payload, indent=2) + "\n")
        candidate_sha = sha256_file(candidate)
        environment = dict(os.environ)
        environment.update({
            "EVSP_EXPECTED_COMMIT": identity["expected_commit"],
            "EVSP_REQUIRE_DETACHED": "1",
            "EVSP_MIP_EXPECTED_RESULT_SHA256": source_hashes[
                "snapshot_sha256"
            ],
            "EVSP_MIP_EXPECTED_JOURNAL_SHA256": source_hashes[
                "journal_sha256"
            ],
            "EVSP_MIP_EXPECTED_INITIAL_PARTITION_SHA256": candidate_sha,
        })
        _run([
            str(python), "-u",
            str(REPO_ROOT / "src/run_exact_pool_mip.py"),
            "--result", str(snapshot),
            "--initial-partition-routes", str(candidate),
            "--two-stage",
            "--validate-only",
        ], environment=environment)
        if (
            sha256_file(snapshot) != source_hashes["snapshot_sha256"]
            or sha256_file(source_journal) != source_hashes["journal_sha256"]
            or _verify_snapshot_problem_inputs(
                status, data_dir=data_root
            ) != input_hashes
            or sha256_file(source_instance) != INSTANCE_SHA256
            or sha256_file(source_prices) != PRICES_SHA256
        ):
            raise SystemExit("source changed while preparing GIRO start")
        try:
            os.link(candidate, output)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite GIRO start: {output}"
            ) from exc
    result = {
        "output": str(output),
        "output_sha256": sha256_file(output),
        "bus_count": 40,
        "trip_count": len(status["trip_ids"]),
        **source_hashes,
        **input_hashes,
    }
    print(json.dumps(result, indent=2))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args(argv)
    prepare(args.snapshot, args.out, args.python)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
