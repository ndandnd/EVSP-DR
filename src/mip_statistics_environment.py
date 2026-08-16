#!/usr/bin/env python3
"""Canonical content identity for the MIP-statistics Python environment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_sha(name: str) -> str:
    distribution = importlib.metadata.distribution(name)
    files = distribution.files
    if files is None:
        raise RuntimeError(f"{name} distribution has no file inventory")
    digest = hashlib.sha256()
    count = 0
    for relative in sorted(files, key=str):
        text = str(relative)
        if "__pycache__" in text or text.endswith(".pyc"):
            continue
        path = Path(distribution.locate_file(relative))
        if not path.is_file():
            continue
        digest.update(text.encode())
        digest.update(b"\0")
        digest.update(_sha(path).encode())
        digest.update(b"\n")
        count += 1
    if count == 0:
        raise RuntimeError(f"{name} distribution has no hashable files")
    return digest.hexdigest()


def identity() -> dict:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python 3.12 required, found {sys.version}")
    import gurobipy
    import numpy
    import pandas
    import scipy

    executable = Path(sys.executable).resolve()
    payload = {
        "executable": str(executable),
        "executable_sha256": _sha(executable),
        "version": platform.python_version(),
        "gurobi_version": ".".join(
            map(str, gurobipy.gurobi.version())
        ),
        "numpy_version": numpy.__version__,
        "pandas_version": pandas.__version__,
        "scipy_version": scipy.__version__,
        "numpy_distribution_sha256": _distribution_sha("numpy"),
        "pandas_distribution_sha256": _distribution_sha("pandas"),
        "scipy_distribution_sha256": _distribution_sha("scipy"),
        "gurobipy_distribution_sha256": _distribution_sha("gurobipy"),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()
    payload["identity_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


if __name__ == "__main__":
    print(json.dumps(identity(), sort_keys=True))
