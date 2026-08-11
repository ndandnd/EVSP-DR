"""Validate campaign sources and decide whether an output is truly complete."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from durable_io import read_jsonl_records, valid_json_object
from run_cg_snapshot_control import control_result_complete
from run_exact_pool_mip import file_sha256, resolve_pool_journal


TERMINAL_POOL_STOPS = {
    "certified", "wall_limit", "max_iters", "no_path",
    "degenerate_stall", "stalled_marginal_returns", "master_failed",
}


def current_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def source_identity(source: Path, *, require_terminal: bool = False) -> dict:
    with open(source) as fh:
        status = json.load(fh)
    if not isinstance(status, dict):
        raise ValueError(f"{source} is not a JSON object")
    stop = str(status.get("stop_reason", ""))
    is_snapshot = source.name.endswith(".snapshot.json")
    if require_terminal and not (
            is_snapshot or stop in TERMINAL_POOL_STOPS):
        raise ValueError(
            f"{source} is not immutable/terminal (stop_reason={stop!r})"
        )
    journal = resolve_pool_journal(source, status)
    read_jsonl_records(journal, repair_trailing=False, collect=False)
    return {
        "result": str(source),
        "result_sha256": file_sha256(source),
        "journal": str(journal),
        "journal_sha256": file_sha256(journal),
        "stop_reason": status.get("stop_reason"),
    }


def output_complete(family: str, source: Path, output: Path,
                    *, total_wall_s: float = 172800.0,
                    expected_commit: str | None = None) -> bool:
    try:
        identity = source_identity(source, require_terminal=True)
    except (OSError, ValueError, SystemExit):
        return False
    expected_commit = expected_commit or current_commit()
    if family == "LA":
        if not valid_json_object(
                output, ("any_method_succeeded",
                         "original_source_result_sha256",
                         "original_source_journal_sha256",
                         "audit_provenance")):
            return False
        with open(output) as fh:
            payload = json.load(fh)
        provenance = payload.get("audit_provenance") or {}
        result_hash_field = "original_source_result_sha256"
        journal_hash_field = "original_source_journal_sha256"
    elif family == "MC":
        if not valid_json_object(
                output, ("status_name", "buses", "source_result_sha256",
                         "source_journal_sha256", "mip_provenance")):
            return False
        with open(output) as fh:
            payload = json.load(fh)
        provenance = payload.get("mip_provenance") or {}
        result_hash_field = "source_result_sha256"
        journal_hash_field = "source_journal_sha256"
    elif family == "CC":
        if not control_result_complete(output, total_wall_s):
            return False
        with open(output) as fh:
            payload = json.load(fh)
        parent = payload.get("resume_parent") or {}
        provenance = payload.get("provenance") or {}
        if (parent.get("snapshot_sha256") != identity["result_sha256"]
                or parent.get("journal_sha256")
                != identity["journal_sha256"]):
            return False
        return (expected_commit is None
                or provenance.get("git_commit") == expected_commit)
    else:
        raise ValueError(f"unknown campaign family {family!r}")
    return (
        payload.get(result_hash_field) == identity["result_sha256"]
        and payload.get(journal_hash_field) == identity["journal_sha256"]
        and (expected_commit is None
             or provenance.get("git_commit") == expected_commit)
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--family", choices=("LA", "MC", "CC"))
    parser.add_argument("--total-wall-s", type=float, default=172800.0)
    parser.add_argument("--expected-commit")
    parser.add_argument("--print-source", action="store_true")
    parser.add_argument("--require-terminal", action="store_true")
    args = parser.parse_args(argv)

    try:
        identity = source_identity(
            args.source, require_terminal=args.require_terminal
        )
    except (OSError, ValueError, SystemExit) as exc:
        print(f"INVALID\t{exc}")
        return 2
    if args.print_source:
        print("\t".join(str(identity[key]) for key in (
            "result_sha256", "journal", "journal_sha256", "stop_reason"
        )))
        return 0
    if args.output is None or args.family is None:
        parser.error("--output and --family are required unless --print-source")
    return 0 if output_complete(
        args.family, args.source, args.output,
        total_wall_s=args.total_wall_s,
        expected_commit=args.expected_commit,
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
