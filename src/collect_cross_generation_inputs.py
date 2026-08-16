#!/usr/bin/env python3
"""Read-only collector that resolves explicit Unicorn evidence requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assignments(values):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"root assignment must be NAME=PATH: {value}")
        name, raw = value.split("=", 1)
        if not name or name in result:
            raise ValueError(f"duplicate/empty root alias: {name}")
        result[name] = Path(raw).expanduser().absolute()
    return result


def collect(template_path: Path, roots: dict[str, Path]) -> dict:
    template = json.loads(template_path.read_text())
    if template.get("schema") != "evsp-dr-cross-generation-input-manifest-v1":
        raise ValueError("unexpected template schema")
    artifacts = list(template.get("artifacts") or [])
    collection = []
    for request in template.get("collection_requests") or []:
        alias = request.get("root_alias")
        root = roots.get(alias)
        if root is None:
            collection.append({
                "request_id": request.get("request_id"),
                "status": "root_not_supplied",
                "root_alias": alias,
            })
            continue
        if not root.is_dir():
            collection.append({
                "request_id": request.get("request_id"),
                "status": "root_missing",
                "path": str(root),
            })
            continue
        matches = sorted(
            path for path in root.glob(request["glob"])
            if path.is_file() and not path.is_symlink()
        )
        if not matches:
            collection.append({
                "request_id": request.get("request_id"),
                "status": "no_matches",
                "path": str(root),
                "glob": request["glob"],
            })
            continue
        run_pattern = (
            re.compile(request["run_id_regex"])
            if request.get("run_id_regex") else None
        )
        for path in matches:
            relative = str(path.relative_to(root))
            if run_pattern:
                match = run_pattern.search(relative)
                if match is None or "run_id" not in match.groupdict():
                    collection.append({
                        "request_id": request["request_id"],
                        "status": "run_id_regex_mismatch",
                        "path": str(path),
                    })
                    continue
                run_id = (
                    f"{request.get('run_id_namespace', request['request_id'])}:"
                    f"{match.group('run_id')}"
                )
            else:
                run_id = (
                    f"{request.get('run_id_namespace', request['request_id'])}:"
                    f"{path.stem}"
                )
            artifact_id = (
                f"{request['request_id']}-"
                + hashlib.sha256(relative.encode()).hexdigest()[:16]
            )
            metadata = json.loads(json.dumps(request.get("metadata") or {}))
            artifacts.append({
                "artifact_id": artifact_id,
                "run_id": run_id,
                "artifact_role": relative,
                "path": str(path),
                "artifact_type": request["artifact_type"],
                "expected_sha256": _sha(path),
                "required": request.get("required", False),
                "metadata": metadata,
            })
            collection.append({
                "request_id": request["request_id"],
                "status": "collected",
                "artifact_id": artifact_id,
                "path": str(path),
                "sha256": artifacts[-1]["expected_sha256"],
            })
    return {
        **{
            key: value for key, value in template.items()
            if key not in {"collection_requests", "artifacts"}
        },
        "artifacts": sorted(artifacts, key=lambda value: value["artifact_id"]),
        "collection_report": collection,
        "resolved_roots": {
            name: str(path) for name, path in sorted(roots.items())
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--out-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    result = collect(args.template, _assignments(args.root))
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(args.out_manifest, flags, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({
        "out_manifest": str(args.out_manifest),
        "artifacts": len(result["artifacts"]),
        "collected": sum(
            row["status"] == "collected"
            for row in result["collection_report"]
        ),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
