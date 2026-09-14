#!/usr/bin/env python3
"""Emit compact native records for the matched one-hour MIP follow-up."""

import argparse
import importlib.util
import json
from pathlib import Path


SOURCE_ROOT = Path("/home/nc437/ladder-lite/strict_capacity_parallel_20260914")
HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, default=HERE)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("strict_capacity_campaign", SOURCE_ROOT / "campaign.py")
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    manifest_path = args.campaign_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["reporting_contract"] = manifest["proof_scope"]
    payload = campaign.collect_campaign(
        manifest, args.campaign_root.resolve(), manifest_path=manifest_path.resolve(),
    )
    if args.out:
        if args.out.exists():
            raise FileExistsError(args.out)
        campaign.atomic_json(args.out.resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
