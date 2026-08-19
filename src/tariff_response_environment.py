#!/usr/bin/env python3
"""Emit the approved Python/native solver environment identity."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path


PORTABLE_FIELDS = (
    "executable", "executable_sha256", "python",
    "numpy", "pandas", "scipy", "matplotlib", "gurobi",
    "numpy_distribution_sha256", "pandas_distribution_sha256",
    "scipy_distribution_sha256", "matplotlib_distribution_sha256",
    "gurobipy_distribution_sha256", "numpy_build_identity_schema",
    "numpy_build", "scipy_build",
    "highs_version", "machine",
)
CG_PORTABLE_FIELDS = tuple(
    field for field in PORTABLE_FIELDS
    if field not in {"gurobi", "gurobipy_distribution_sha256"}
)


_RUNTIME_SIMD_FIELDS = frozenset({"found", "not found"})
NUMPY_BUILD_IDENTITY_SCHEMA = (
    "numpy-config-v2-runtime-simd-separated"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_sha(name: str) -> str:
    distribution = importlib.metadata.distribution(name)
    if distribution.files is None:
        raise RuntimeError(f"{name} distribution has no file inventory")
    digest = hashlib.sha256()
    count = 0
    for relative in sorted(distribution.files, key=str):
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


def _simd_features(value, field):
    if not isinstance(value, list) or any(
        not isinstance(item, str) for item in value
    ):
        raise ValueError(f"SIMD Extensions.{field} must be a string list")
    return sorted(set(value))


def _stable_build_config(config):
    """Return canonical build identity with runtime SIMD partition removed.

    NumPy's ``CONFIG`` report combines immutable wheel build information with
    the CPU features detected on the importing host.  The latter legitimately
    differs across Unicorn nodes, so it is provenance rather than portable
    package identity.  The compiled SIMD baseline and dispatch set remain part
    of the build identity; only the host-dependent partition of the dispatch
    set into ``found`` and ``not found`` is removed.
    """
    if not isinstance(config, dict):
        return json.dumps(config, sort_keys=True, separators=(",", ":"))
    stable = copy.deepcopy(config)
    simd = stable.get("SIMD Extensions")
    if isinstance(simd, dict):
        baseline = _simd_features(simd.get("baseline"), "baseline")
        # NumPy's generated config removes falsey values, so either runtime
        # partition key may be absent when its canonical value is an empty
        # list. A present malformed value still fails closed.
        found = _simd_features(simd.get("found", []), "found")
        not_found = _simd_features(
            simd.get("not found", []), "not found"
        )
        if set(found) & set(not_found):
            raise ValueError(
                "SIMD Extensions found/not found sets overlap"
            )
        stable["SIMD Extensions"] = {
            key: value
            for key, value in simd.items()
            if key not in _RUNTIME_SIMD_FIELDS
        }
        stable["SIMD Extensions"]["baseline"] = baseline
        stable["SIMD Extensions"]["dispatch"] = sorted(
            set(found) | set(not_found)
        )
    return json.dumps(stable, sort_keys=True, separators=(",", ":"))


def _runtime_simd_metadata(config):
    """Return host SIMD observations excluded from portable identity."""
    if not isinstance(config, dict):
        return None
    simd = config.get("SIMD Extensions")
    if not isinstance(simd, dict):
        return None
    observed = {}
    for key in ("found", "not found"):
        observed[key] = _simd_features(simd.get(key, []), key)
    return observed


def identity(profile="full"):
    import matplotlib
    import numpy
    import pandas
    import scipy

    executable = Path(sys.executable).resolve()
    numpy_config = getattr(numpy.__config__, "CONFIG", None)
    scipy_config = getattr(scipy.__config__, "CONFIG", None)
    try:
        import highspy
        highs_version = str(highspy.Highs().version())
    except (ImportError, AttributeError):
        highs_version = f"scipy-bundled-with-{scipy.__version__}"
    portable = {
        "python": platform.python_version(),
        "executable": str(executable),
        "executable_sha256": _sha(executable),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "machine": platform.machine().strip().lower(),
        "numpy_build_identity_schema": NUMPY_BUILD_IDENTITY_SCHEMA,
        "numpy_build": _stable_build_config(numpy_config),
        "scipy_build": _stable_build_config(scipy_config),
        "highs_version": highs_version,
        "numpy_distribution_sha256": _distribution_sha("numpy"),
        "pandas_distribution_sha256": _distribution_sha("pandas"),
        "scipy_distribution_sha256": _distribution_sha("scipy"),
        "matplotlib_distribution_sha256": _distribution_sha("matplotlib"),
    }
    if profile == "full":
        import gurobipy
        portable.update({
            "gurobi": ".".join(
                str(value) for value in gurobipy.gurobi.version()
            ),
            "gurobipy_distribution_sha256":
                _distribution_sha("gurobipy"),
        })
    elif profile != "cg":
        raise ValueError(f"unknown portable identity profile: {profile}")
    encoded = json.dumps(
        portable, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "schema": "evsp-dr-portable-environment-v1",
        "portable_profile": profile,
        "portable": portable,
        "portable_identity_sha256": hashlib.sha256(encoded).hexdigest(),
        "node_metadata": {
            "platform": platform.platform(),
            "hostname": platform.node(),
            "system": platform.system(),
            "kernel_release": platform.release(),
            "os_version": platform.version(),
            "pythonpath_observed": os.environ.get("PYTHONPATH"),
            "numpy_runtime_simd": _runtime_simd_metadata(numpy_config),
            "scipy_runtime_simd": _runtime_simd_metadata(scipy_config),
        },
    }


def compare_portable(planned: dict, observed: dict) -> list[dict]:
    differences = []
    planned_portable = planned.get("portable")
    observed_portable = observed.get("portable")
    if not isinstance(planned_portable, dict):
        return [{
            "field": "portable",
            "planned": planned_portable,
            "observed": observed_portable,
            "reason": "missing_planned_required_object",
        }]
    if not isinstance(observed_portable, dict):
        return [{
            "field": "portable",
            "planned": planned_portable,
            "observed": observed_portable,
            "reason": "missing_observed_required_object",
        }]
    profile = planned.get("portable_profile", "full")
    if observed.get("portable_profile", "full") != profile:
        differences.append({
            "field": "portable_profile",
            "planned": profile,
            "observed": observed.get("portable_profile"),
            "reason": "value_mismatch",
        })
    fields = PORTABLE_FIELDS if profile == "full" else CG_PORTABLE_FIELDS
    for field in fields:
        planned_value = planned_portable.get(field)
        observed_value = observed_portable.get(field)
        if field not in planned_portable or field not in observed_portable:
            reason = "missing_required_field"
        elif planned_value != observed_value:
            reason = "value_mismatch"
        else:
            continue
        differences.append({
            "field": f"portable.{field}",
            "planned": planned_value,
            "observed": observed_value,
            "reason": reason,
        })
    planned_sha = planned.get("portable_identity_sha256")
    observed_sha = observed.get("portable_identity_sha256")
    if planned_sha != observed_sha and not differences:
        differences.append({
            "field": "portable_identity_sha256",
            "planned": planned_sha,
            "observed": observed_sha,
            "reason": "value_mismatch",
        })
    return differences


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=("full", "cg"), default="full"
    )
    parser.add_argument("--compare-plan", type=Path)
    parser.add_argument("--plan-field", default="python_identity")
    args = parser.parse_args(argv)
    observed = identity(args.profile)
    if args.compare_plan is None:
        print(json.dumps(observed, sort_keys=True))
        return 0
    plan = json.loads(args.compare_plan.read_text())
    planned = plan.get(args.plan_field)
    differences = compare_portable(
        planned if isinstance(planned, dict) else {},
        observed,
    )
    print(json.dumps({
        "compatible": not differences,
        "differences": differences,
        "planned_portable_identity_sha256": (
            planned or {}
        ).get("portable_identity_sha256"),
        "observed_portable_identity_sha256":
            observed["portable_identity_sha256"],
        "observed_node_metadata": observed["node_metadata"],
    }, sort_keys=True))
    return 0 if not differences else 3


if __name__ == "__main__":
    raise SystemExit(main())
