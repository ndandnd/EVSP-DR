#!/usr/bin/env python3
"""Fail-closed repair for source telemetry copied to a new resume identity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_live_identity(path: Path, output: Path, commit: str) -> None:
    with path.open() as handle:
        first = next((line for line in handle if line.strip()), "")
    try:
        record = json.loads(first)
    except Exception as exc:
        raise SystemExit(f"malformed live resume telemetry: {path}") from exc
    identity = record.get("identity") or {}
    if (
        record.get("record_type") != "session_start"
        or identity.get("output") != str(output.resolve())
        or identity.get("git_commit") != commit
    ):
        raise SystemExit(f"live resume telemetry identity mismatch: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.resume_root.resolve()
    plan = json.loads((root / "execution_plan.json").read_text())
    solver_commit = plan["solver_commit"]
    report = []
    with (root / "matrix.tsv").open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            source_status = Path(row["source_status"])
            source_telemetry = Path(
                str(source_status) + ".phase-telemetry.jsonl"
            )
            resume_status = Path(row["resume_status"])
            resume_journal = Path(row["resume_journal"])
            live_telemetry = Path(
                str(resume_status) + ".phase-telemetry.jsonl"
            )
            archived_telemetry = Path(
                str(resume_status) + ".source-phase-telemetry.jsonl"
            )
            staged_state_unchanged = (
                sha256(resume_status) == row["staged_status_sha256"]
                and sha256(resume_journal) == row["staged_journal_sha256"]
            )
            if not source_telemetry.is_file():
                if archived_telemetry.exists():
                    raise SystemExit(
                        f"{row['local_index']}: orphan staged telemetry"
                    )
                if live_telemetry.exists():
                    validate_live_identity(
                        live_telemetry, resume_status, solver_commit
                    )
                    action = "resume_telemetry_active_no_source_archive"
                else:
                    action = "source_telemetry_absent"
                digest = ""
            else:
                digest = sha256(source_telemetry)
                if live_telemetry.exists() and archived_telemetry.exists():
                    if sha256(archived_telemetry) != digest:
                        raise SystemExit(
                            f"{row['local_index']}: archived source "
                            "telemetry hash mismatch"
                        )
                    validate_live_identity(
                        live_telemetry, resume_status, solver_commit
                    )
                    action = "resume_telemetry_active"
                if live_telemetry.exists():
                    if archived_telemetry.exists():
                        pass
                    elif sha256(live_telemetry) != digest:
                        raise SystemExit(
                            f"{row['local_index']}: source archive is absent "
                            "and live telemetry is not the source copy"
                        )
                    else:
                        if not staged_state_unchanged:
                            raise SystemExit(
                                f"{row['local_index']}: scientific state "
                                "changed; refusing telemetry-only repair"
                            )
                        live_telemetry.rename(archived_telemetry)
                        action = "archived_misplaced_source_copy"
                elif archived_telemetry.exists():
                    if sha256(archived_telemetry) != digest:
                        raise SystemExit(
                            f"{row['local_index']}: archived source "
                            "telemetry hash mismatch"
                        )
                    if not staged_state_unchanged:
                        raise SystemExit(
                            f"{row['local_index']}: resume state changed "
                            "without a live telemetry stream"
                        )
                    action = "already_archived"
                else:
                    raise SystemExit(
                        f"{row['local_index']}: source telemetry was not "
                        "preserved in the staged resume"
                    )
            report.append({
                "local_index": row["local_index"],
                "cell": row["cell"],
                "representation_id": row["representation_id"],
                "source_telemetry": str(source_telemetry),
                "source_telemetry_sha256": digest,
                "archived_telemetry": str(archived_telemetry),
                "action": action,
                "resume_status_sha256": sha256(resume_status),
                "resume_journal_sha256": sha256(resume_journal),
            })
    output = root / "telemetry_repair.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    counts = {}
    for row in report:
        counts[row["action"]] = counts.get(row["action"], 0) + 1
    print("CG telemetry repair:", dict(sorted(counts.items())))
    print(f"CSV: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
