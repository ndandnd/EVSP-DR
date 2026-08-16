#!/usr/bin/env python3
"""Tiny compute-node probe for portable completion-marker publication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portable_bundle import atomic_write_new_file, capability_probe


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = capability_probe(args.directory)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.out:
        if not args.out.parent.is_dir():
            raise SystemExit("probe --out parent must already exist")
        atomic_write_new_file(args.out, encoded.encode())
    print(encoded, end="")
    return 0 if result["ready_for_recovery_probe_only"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
