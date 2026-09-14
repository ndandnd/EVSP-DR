#!/usr/bin/env python3
"""Run the shared strict-capacity verifier for this pilot-format root."""

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_adapter(path: Path):
    spec = importlib.util.spec_from_file_location("strict_capacity_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--python", default="/home/nc437/evsp_env/bin/python")
    args = parser.parse_args()
    result = load_adapter(args.adapter).collect_strict_capacity(
        args.root, "pilot", args.python,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
