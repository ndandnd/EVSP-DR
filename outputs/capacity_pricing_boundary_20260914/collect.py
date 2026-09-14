#!/usr/bin/env python3
"""Emit the strict-capacity campaign's compact native collection schema."""

import argparse
import json
from pathlib import Path

import campaign


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=campaign.MANIFEST_PATH)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    payload = campaign.collect_campaign(
        campaign.load_manifest(manifest_path), args.campaign_root.resolve(),
        manifest_path=manifest_path,
    )
    if args.out is not None:
        if args.out.exists():
            raise FileExistsError(f"refusing to overwrite collection: {args.out}")
        campaign.atomic_json(args.out.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
