#!/usr/bin/env python3
"""Hash immutable exact-CG inputs before submitted finite-pool solves."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--panel", choices=("A", "B"), required=True)
    parser.add_argument("--source-dir", choices=("cg", "frozen"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--wrapper-commit")
    parser.add_argument("--solver-commit")
    args = parser.parse_args()
    root = args.root.resolve()
    rows = list(csv.reader((root / "matrix.tsv").open(), delimiter="\t"))
    manifest = []
    for fields in rows:
        if args.panel == "A":
            index, cell, target, instance_csv, rep = fields[:5]
        else:
            index, cell, target, instance_csv, rep = fields[:5]
        source = root / args.source_dir / f"{args.panel}__{cell}__{rep}.json"
        if not source.is_file():
            raise SystemExit(f"missing source status: {source}")
        status = json.loads(source.read_text())
        journal = Path(status["columns_journal"]).resolve(strict=True)
        manifest.append({
            "index": index,
            "cell": cell,
            "target_fleet": target,
            "representation_id": rep,
            "source_status": str(source),
            "source_status_sha256": sha256(source),
            "source_journal": str(journal),
            "source_journal_sha256": sha256(journal),
        })
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest)
    if args.provenance:
        args.provenance.write_text(json.dumps({
            "schema": "evsp-dr-integer-input-manifest-v1",
            "panel": args.panel,
            "wrapper_commit": args.wrapper_commit,
            "solver_commit": args.solver_commit,
            "source_rows": len(manifest),
            "corrections": [
                "precomputed immutable status and journal hashes",
                "canonical Unicorn Gurobi license path",
            ],
        }, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(manifest)} immutable input rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
