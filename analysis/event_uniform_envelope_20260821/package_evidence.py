#!/usr/bin/env python3
"""Create the deterministic compact evidence bundle for this analysis."""

from __future__ import annotations

import argparse
import gzip
import tarfile
from pathlib import Path


def selected_artifacts(root: Path) -> list[Path]:
    selected = []
    for path in root.rglob("*.json"):
        relative = path.relative_to(root).as_posix()
        scoped = f"/{relative}"
        if (
            relative.endswith("/cg.json")
            or "/mip_native/" in scoped
            or "/mip_native_v2/" in scoped
            or "/fleet_phase2.json" in relative
            or "/model_witness/" in scoped
            or "/arcflow/" in scoped
            or "/target/" in scoped
            or "/target_v2/" in scoped
            or "/snapshots/" in scoped
            or "/snapshots_v2/" in scoped
        ):
            selected.append(path)
    return sorted(
        selected, key=lambda path: path.relative_to(root).as_posix()
    )


def package(execution_root: Path, bundle_out: Path) -> int:
    root = execution_root.expanduser().resolve(strict=True)
    output = bundle_out.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    artifacts = selected_artifacts(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0
        ) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in artifacts:
                    relative = path.relative_to(root).as_posix()
                    info = archive.gettarinfo(
                        str(path), arcname=relative
                    )
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    with path.open("rb") as source:
                        archive.addfile(info, source)
    return len(artifacts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        required=True,
        help="Bound for command readability; bundle must be inside it.",
    )
    parser.add_argument("--bundle-out", type=Path, required=True)
    args = parser.parse_args(argv)
    analysis_dir = args.analysis_dir.expanduser().resolve(strict=True)
    bundle_out = args.bundle_out.expanduser().resolve()
    bundle_out.relative_to(analysis_dir)
    count = package(args.execution_root, bundle_out)
    print(f"packaged {count} compact evidence artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
