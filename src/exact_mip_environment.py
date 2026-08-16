#!/usr/bin/env python3
"""Print a content-bound identity for the exact-pool MIP environment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_sha256(name: str) -> str:
    distribution = importlib.metadata.distribution(name)
    digest = hashlib.sha256()
    files = distribution.files
    if files is None:
        raise RuntimeError(f"distribution {name} has no file manifest")
    included = 0
    for relative in sorted(files, key=str):
        relative_text = str(relative)
        if "__pycache__" in relative_text or relative_text.endswith(".pyc"):
            continue
        path = Path(distribution.locate_file(relative))
        if not path.is_file():
            continue
        digest.update(relative_text.encode())
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode())
        digest.update(b"\n")
        included += 1
    if included == 0:
        raise RuntimeError(f"distribution {name} has no hashable files")
    return digest.hexdigest()


def environment_identity() -> dict:
    import gurobipy
    import numpy
    import pandas
    import scipy

    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Python 3.12 required, found {sys.version}")
    executable = Path(sys.executable).resolve()
    payload = {
        "python_executable": str(executable),
        "python_executable_sha256": _sha256_file(executable),
        "python_version": platform.python_version(),
        "numpy_version": numpy.__version__,
        "numpy_distribution_sha256": _distribution_sha256("numpy"),
        "pandas_version": pandas.__version__,
        "pandas_distribution_sha256": _distribution_sha256("pandas"),
        "scipy_version": scipy.__version__,
        "scipy_distribution_sha256": _distribution_sha256("scipy"),
        "gurobi_version": ".".join(
            map(str, gurobipy.gurobi.version())
        ),
        "gurobipy_distribution_sha256": _distribution_sha256("gurobipy"),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()
    payload["identity_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


if __name__ == "__main__":
    print(json.dumps(environment_identity(), sort_keys=True))
