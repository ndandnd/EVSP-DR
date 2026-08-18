#!/usr/bin/env python3
"""Emit the approved Python/native solver environment identity."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path


def identity():
    import gurobipy
    import matplotlib
    import numpy
    import pandas
    import scipy

    executable = Path(sys.executable).resolve()
    return {
        "python": platform.python_version(),
        "executable": str(executable),
        "executable_sha256": hashlib.sha256(
            executable.read_bytes()
        ).hexdigest(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "gurobi": ".".join(
            str(value) for value in gurobipy.gurobi.version()
        ),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "numpy_build": getattr(numpy.__config__, "CONFIG", None),
    }


if __name__ == "__main__":
    print(json.dumps(identity(), sort_keys=True))
