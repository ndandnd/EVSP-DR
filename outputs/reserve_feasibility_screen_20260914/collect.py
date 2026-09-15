#!/usr/bin/env python3
"""Emit the strict-capacity pilot adapter contract on stdout."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import campaign

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path)
p.add_argument('--campaign-root',type=Path)
p.add_argument('--manifest',type=Path)
a=p.parse_args()
root=(a.campaign_root or a.root)
if root is None: p.error('--root or --campaign-root is required')
root=root.resolve(); manifest_path=(a.manifest.resolve() if a.manifest else root/'manifest.json')
manifest=campaign.load_manifest(manifest_path)
campaign.validate_tooling(manifest,root)
v=campaign.collect_campaign(manifest,root,manifest_path=manifest_path)
v['errors']=[]
print(json.dumps(v,sort_keys=True,allow_nan=False))
