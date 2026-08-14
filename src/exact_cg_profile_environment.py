#!/usr/bin/env python3
"""Print a canonical identity for the exact-CG profiling Python environment."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy
import pandas
import scipy


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_identity() -> dict:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python 3.12 required, found {sys.version}")
    try:
        import highspy
        highs_version = highspy.Highs().version()
    except (ImportError, AttributeError):
        highs_version = f"scipy-bundled-with-{scipy.__version__}"
    executable = Path(sys.executable).resolve()
    payload = {
        "python_executable": str(executable),
        "python_executable_sha256": sha256_file(executable),
        "python_version": platform.python_version(),
        "numpy_version": numpy.__version__,
        "pandas_version": pandas.__version__,
        "scipy_version": scipy.__version__,
        "highs_version": str(highs_version),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()
    payload["identity_sha256"] = hashlib.sha256(encoded).hexdigest()
    payload["observed_platform"] = platform.platform()
    return payload


if __name__ == "__main__":
    print(json.dumps(environment_identity(), sort_keys=True))
