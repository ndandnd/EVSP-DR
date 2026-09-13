#!/usr/bin/env python3
"""Freeze the controlled comparison contract; never submit jobs."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    base = Path('/home/nc437/ladder-lite')
    root = base / 'controlled_comparison_20260913'
    root.mkdir(exist_ok=True)
    assert not (root / 'manifest.json').exists(), 'Preserve existing campaign; do not prepare twice'
    recovery = base / 'full_pool_recovery_20260912'
    source_manifest = json.loads((recovery / 'manifest.json').read_text())
    code = recovery / 'code'
    mip_code = base / 'execution/871d057e1067411f09581e37d78f7c1ca43f68bb'
    cg_commit = 'e091a4dba549510238507ef5e5367abea958bd30'
    mip_commit = '871d057e1067411f09581e37d78f7c1ca43f68bb'
    for checkout, pin in [(code, cg_commit), (mip_code, mip_commit)]:
        assert subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip() == pin
        assert not subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    assets = {}
    frozen = []
    for case_id in ['w1_k08', 'w4_k10', 'w3_k15']:
        original = source_manifest['cases'][case_id]
        destination = root / 'inputs' / case_id
        destination.mkdir(parents=True, exist_ok=False)
        parent_source = Path(original['parent_status'])
        parent = json.loads(parent_source.read_text())
        assert parent['certified_rc_optimal'] and parent['final']['artificials'] == 0 and parent['final']['iter'] > 0
        assert sha(code / 'data' / original['csv']) == original['input_sha256']
        assert sha(code / 'data' / parent['csv']) == parent['provenance']['instance_sha256']
        assert sha(original['cache']) == original['cache_sha256']
        assert sha(original['cache'] + '.manifest.json') == original['cache_manifest_sha256']
        parent_raw = destination / 'parent_original.json'
        shutil.copyfile(parent_source, parent_raw)
        journal_source = Path(parent['columns_journal'])
        journal = destination / 'parent_columns.jsonl'
        try:
            os.link(journal_source, journal)
            storage = 'hard link to completed immutable source journal; do not mutate either path'
        except OSError:
            shutil.copyfile(journal_source, journal)
            storage = 'independent copy'
        journal_hash = sha(journal)
        parent['columns_journal'] = str(journal)
        descriptor = destination / 'parent_descriptor.json'
        write(descriptor, parent)
        entry = dict(csv=original['csv'], input_sha256=original['input_sha256'],
                     cache=original['cache'], cache_sha256=original['cache_sha256'],
                     cache_manifest_sha256=original['cache_manifest_sha256'],
                     parent_descriptor=str(descriptor), parent_descriptor_sha256=sha(descriptor),
                     parent_journal=str(journal), parent_journal_sha256=journal_hash,
                     parent_csv=parent['csv'], parent_csv_sha256=parent['provenance']['instance_sha256'],
                     target_k=original['target_duties'], parent_source=str(parent_source),
                     parent_source_sha256=sha(parent_source), parent_journal_source=str(journal_source),
                     parent_storage=storage, parent_certificate=True,
                     cache_storage='reuse authenticated existing graph; construction excluded from timed arms')
        assets[case_id] = entry
        for p in [parent_raw, descriptor, journal]:
            frozen.append(dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size))
    static = []
    for p in [code / 'data/Ref_dict.csv', code / 'data/par_ref_dhd.csv', code / 'data/hourly_prices_flat.csv', base / 'SCAGLIONE_RESOURCE_POLICY.md']:
        static.append(dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size))
    arms = {
        'A': dict(fixed_sequence_index=False, inherit_max_columns=512, skip_gurobi_incidence=False),
        'B': dict(fixed_sequence_index=True, inherit_max_columns=512, skip_gurobi_incidence=False),
        'C': dict(fixed_sequence_index=True, inherit_max_columns=0, skip_gurobi_incidence=False),
        'D': dict(fixed_sequence_index=True, inherit_max_columns=0, skip_gurobi_incidence=True),
        'E': dict(fixed_sequence_index=False, inherit_max_columns=0, skip_gurobi_incidence=False),
    }
    contrasts = {'index': ['A', 'B'], 'pool': ['B', 'C'], 'master': ['C', 'D'], 'full_index': ['E', 'C']}
    pairs = []
    for case in assets:
        for contrast, order in contrasts.items():
            for repetition in [1, 2]:
                pairs.append(dict(id=f'{case}_{contrast}_r{repetition}', case_id=case,
                                  contrast=contrast, repetition=repetition,
                                  order=order if repetition == 1 else list(reversed(order)),
                                  cpus=8, mem='96G', slurm_time='07:00:00'))
    manifest = dict(schema='evsp-controlled-comparison-contract-v1', root=str(root),
                    prepared_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    python='/home/nc437/evsp_env/bin/python', code=str(code), cg_commit=cg_commit,
                    mip_code=str(mip_code), mip_commit=mip_commit,
                    common=dict(cg_seconds=7200, mip_seconds=3600, stage1_seconds=1800, threads=8,
                                master_sense='cover', battery_kwh=240, charge_kw=240, soc_step_kwh=2.5,
                                block_minutes=5, reserve_kwh=0, return_soc_floor=None,
                                shared_station_capacity=False, tariff='flat',
                                objective='100000 + electricity + 5 * charge_starts',
                                columns_per_iter=30, rc_epsilon=0.0001, inherit_workers=8,
                                inherit_time_limit_s=0, mip_stage2_fleet='<= validated stage1 incumbent'),
                    arms=arms, contrasts=contrasts, inputs=assets, pairs=pairs, static_files=static,
                    frozen_files=frozen, source_manifest=dict(path=str(recovery/'manifest.json'), sha256=sha(recovery/'manifest.json')),
                    resources=dict(partition='default_partition', exclude='scaglione-compute-01',
                                   independent_allocations=24, concurrency=24, requeue=False,
                                   reason='All 24 independent comparisons eligible immediately. Each arm receives the same 8 CPU/96 GiB allocation used by successful full-pool chains. Sequential arms share a node. Seven hours covers two 2-hour CG and 1-hour MIP budgets plus startup and validation.'),
                    interpretation=[
                        'Primary units are three deliberately selected cases, not 18 independent datasets.',
                        'Reverse-order repetition controls order effects partially; two repetitions do not establish population significance.',
                        'All arms share the repaired e091 execution code. Only the listed contrast flag changes.',
                        'No import-specific deadline: both 512 arms attempt every selected sequence. This differs from historical 512/900-second runs.',
                        'e091 uses ordered imap when import_time_limit_s=0, even with eight workers; verify imported order and content hashes.',
                        'CG uses a shared frozen parent and cache, never another comparison arm as its parent.',
                        'A timeout is censored; equal time limits alone do not establish a speedup.',
                        'Preserve weighted LP, route weight, pricing certificate, pool fleet proof and physical replay as separate outputs.',
                        'Input authentication precedes paired process timings; graph loading and runtime index setup remain inside CG time.',
                        'Final MIP may improve the charging objective after fleet proof; report both stages separately.',
                        'No new graph-builder or capacity-pricing benchmark is included in this baseline campaign.',
                    ])
    write(root / 'manifest.json', manifest)
    print(json.dumps(dict(root=str(root), manifest_sha256=sha(root/'manifest.json'), pairs=len(pairs), inputs=list(assets))))


if __name__ == '__main__':
    main()
