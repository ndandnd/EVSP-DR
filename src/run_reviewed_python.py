#!/usr/bin/env python3
"""Isolated bootstrap for reviewed EVSP-DR operational entry points."""

from __future__ import annotations

import os
import pathlib
import re
import runpy
import subprocess
import sys


ALLOWED_TARGETS = {
    "launch_mip_statistics_campaign.py",
    "validate_k40_cs_overnight_plan.py",
    "summarize_mip_statistics.py",
    "launch_tariff_response_pilot.py",
    "assemble_tariff_response_campaign.py",
    "validate_tariff_response_archive.py",
    "reconcile_tariff_response_gate.py",
    "launch_scale_ladder.py",
    "summarize_scale_ladder.py",
    "reconcile_scale_ladder_gate.py",
    "audit_legacy_scale_ladder_campaign.py",
}


def unsafe_runtime_artifacts(root: pathlib.Path) -> list[str]:
    tracked_result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if tracked_result.returncode != 0:
        return ["<cannot-enumerate-tracked-files>"]
    tracked = {
        value.decode()
        for value in tracked_result.stdout.split(b"\0") if value
    }
    unsafe = []
    for scan_root in (root, root / "src"):
        for current, dirs, files in os.walk(scan_root, followlinks=False):
            current_path = pathlib.Path(current)
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
                if (
                    path.is_symlink()
                    or suffix in {".pyc", ".pyo", ".so", ".pth"}
                    or (suffix == ".py" and relative not in tracked)
                ):
                    unsafe.append(relative)
    return sorted(set(unsafe))


def main() -> int:
    if (
        len(sys.argv) < 3
        or re.fullmatch(r"[0-9a-f]{40}", sys.argv[1]) is None
        or sys.argv[2] not in ALLOWED_TARGETS
    ):
        raise SystemExit(
            "usage: run_reviewed_python.py EXPECTED_COMMIT "
            "{launch_mip_statistics_campaign.py|"
            "validate_k40_cs_overnight_plan.py|"
            "summarize_mip_statistics.py|"
            "launch_tariff_response_pilot.py|"
            "assemble_tariff_response_campaign.py|"
            "validate_tariff_response_archive.py|"
            "reconcile_tariff_response_gate.py|"
            "launch_scale_ladder.py|summarize_scale_ladder.py|"
            "reconcile_scale_ladder_gate.py|"
            "audit_legacy_scale_ladder_campaign.py} [ARG ...]"
        )
    root = pathlib.Path(__file__).resolve().parents[1]
    expected_commit = sys.argv[1]
    head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if (
        head.returncode != 0
        or head.stdout.strip() != expected_commit
        or status.returncode != 0
        or status.stdout.strip()
    ):
        raise SystemExit(
            "bootstrap checkout is not the exact tracked-clean reviewed commit"
        )
    unsafe = unsafe_runtime_artifacts(root)
    if unsafe:
        raise SystemExit(
            f"checkout contains unreviewed runtime artifacts: {unsafe[:10]}"
        )
    target = root / "src" / sys.argv[2]
    arguments = [str(target), *sys.argv[3:]]
    sys.path.insert(0, str(root / "src"))
    sys.argv = arguments
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
