#!/usr/bin/env python3
"""Freeze fee-comparison inputs. Does not submit or alter source artifacts."""
import argparse, datetime as dt, hashlib, json, os, shutil, subprocess
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,data):
    Path(path).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--code',required=True,type=Path);p.add_argument('--commit',required=True);a=p.parse_args()
    base=Path('/home/nc437/ladder-lite');root=base/'zero_charge_start_fee_20260913';root.mkdir(exist_ok=True)
    assert not (root/'manifest.json').exists(),'Preserve the frozen manifest'
    source=base/'full_pool_recovery_20260912';source_code=source/'code';old= json.loads((source/'manifest.json').read_text())
    catalog=json.loads((root/'source_catalog.json').read_text())
    for cmd in [['rev-parse','HEAD'],['status','--porcelain','--untracked-files=no']]:
        text=subprocess.check_output(['git','-C',str(a.code),*cmd],text=True).strip()
        assert text==(a.commit if cmd[0]=='rev-parse' else ''),text
    assets={};frozen=[]
    for chain in range(1,7):
        for k in [5,10,15]:
            cid=f'w{chain}_k{k:02d}';entry=catalog[cid]
            status_path=Path(entry['status']);status=json.loads(status_path.read_text())
            if not status.get('certified_rc_optimal') and cid=='w1_k15':
                status_path=source/'cases/w1_k14/cg.json';status=json.loads(status_path.read_text())
            assert status.get('certified_rc_optimal') and status['final']['artificials']==0 and status['final']['iter']>0,cid
            csv=entry['csv'];csv_source=source_code/'data'/csv
            parent_csv=status['csv'];parent_csv_source=source_code/'data'/parent_csv
            assert sha(parent_csv_source)==status['provenance']['instance_sha256'],cid
            for rel in [csv,parent_csv]:
                dest=a.code/'data'/rel;origin=source_code/'data'/rel
                if dest.exists(): assert sha(dest)==sha(origin),rel
                else:dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(origin,dest)
            if cid in old['cases']:
                o=old['cases'][cid];cache=Path(o['cache']);assert sha(cache)==o['cache_sha256'];assert sha(str(cache)+'.manifest.json')==o['cache_manifest_sha256']
            else:cache=Path(entry['legacy_cache'])
            cache_manifest=Path(str(cache)+'.manifest.json');assert cache.is_file() and cache_manifest.is_file()
            cache_metadata=json.loads(cache_manifest.read_text());cache_source_commit=cache_metadata['identity']['git_commit']
            assert isinstance(cache_source_commit,str) and len(cache_source_commit)==40
            dest=root/'inputs'/cid;dest.mkdir(parents=True,exist_ok=False)
            raw=dest/'parent_original.json';shutil.copy2(status_path,raw)
            parent_journal_source=Path(status['columns_journal']);journal=dest/'parent_columns.jsonl'
            try:os.link(parent_journal_source,journal);storage='hardlink_to_completed_immutable_journal'
            except OSError:shutil.copy2(parent_journal_source,journal);storage='copy'
            status['columns_journal']=str(journal);descriptor=dest/'parent_descriptor.json';write(descriptor,status)
            assets[cid]=dict(csv=csv,input_sha256=sha(csv_source),target_k=k,chain=chain,
                cache=str(cache),cache_sha256=sha(cache),cache_manifest_sha256=sha(cache_manifest),cache_source_commit=cache_source_commit,
                parent_descriptor=str(descriptor),parent_descriptor_sha256=sha(descriptor),
                parent_journal=str(journal),parent_journal_sha256=sha(journal),
                parent_csv=parent_csv,parent_csv_sha256=sha(parent_csv_source),
                parent_source=str(status_path),parent_source_sha256=sha(status_path),
                parent_journal_source=str(parent_journal_source),parent_storage=storage,
                parent_certificate=True,same_k_initial_pool=parent_csv==csv,
                source_parent_fee=5.0,parent_usage='trip sequences replayed and costed under the destination fee; not reused route costs',
                cache_usage='source cache immutable; active process fee applied using audited cache-cost handling')
            frozen.extend(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in [raw,descriptor,journal])
    static=[]
    for rel in ['Ref_dict.csv','par_ref_dhd.csv','hourly_prices_flat.csv']:
        dest=a.code/'data'/rel;origin=source_code/'data'/rel
        if dest.exists():assert sha(dest)==sha(origin)
        else:shutil.copy2(origin,dest)
        static.append(dict(path=str(dest),sha256=sha(dest),bytes=dest.stat().st_size))
    policy=base/'SCAGLIONE_RESOURCE_POLICY.md';static.append(dict(path=str(policy),sha256=sha(policy),bytes=policy.stat().st_size))
    arms={f'fee{fee}':dict(charge_start_cost=float(fee),fixed_sequence_index=True,inherit_max_columns=0,skip_gurobi_incidence=False) for fee in [5,0]}
    runs=[]
    # Alternate order in the launch list; every run is independent, both fees use the same frozen inputs.
    for i,cid in enumerate(assets):
        for fee in ([5,0] if i%2==0 else [0,5]):
            arm=f'fee{fee}';runs.append(dict(id=cid+'_'+arm,case_id=cid,comparison_group=cid,contrast='charge_start_fee',repetition=1,order=[arm],cpus=8,mem='96G',slurm_time='04:00:00'))
    manifest=dict(schema='evsp-zero-charge-start-fee-contract-v1',root=str(root),prepared_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        python='/home/nc437/evsp_env/bin/python',code=str(a.code),cg_commit=a.commit,mip_code=str(a.code),mip_commit=a.commit,
        common=dict(cg_seconds=7200,mip_seconds=3600,stage1_seconds=1800,threads=8,master_sense='cover',battery_kwh=240,charge_kw=240,soc_step_kwh=2.5,block_minutes=5,min_soc_frac=0,return_soc_floor=None,shared_station_capacity=False,tariff='flat',objective='100000 + electricity + treatment_fee * charge_starts',columns_per_iter=30,rc_epsilon=0.0001,inherit_workers=8,inherit_time_limit_s=0,mip_stage2_fleet='<= validated stage1 incumbent'),
        arms=arms,inputs=assets,pairs=runs,static_files=static,frozen_files=frozen,
        source_manifest=dict(path=str(source/'manifest.json'),sha256=sha(source/'manifest.json')),
        resources=dict(partition='default_partition',exclude='scaglione-compute-01',independent_runs=len(runs),concurrency=min(50,len(runs)),requeue=False,reason='All36 independent treatments eligible immediately; same8CPU96GiB per run as validated full-pool CG. Four-hour allocation covers2hCG+1hMIP and startup/validation.'),
        interpretation=['18 matched input cases,36runs,not36 independent random samples.','No restart fromk2: both fee arms replay the same frozen completed target pool; C1k15 can use k14 if its target pool is not yet certified.','Historical fee5 results are context; new fee5 controls use the same updated code as fee0.','Initial sequences come from fee5 pools; converged LPs still require destination-fee pricing certificates, and integer pools can remain biased by initialization.','Electricity cost, start fees, charging activities, energy and terminal energy are reported separately.','Baseline chains have no shared capacity or equal-return-energy constraint; GIRO fair-terminal tariff comparison is a separate cohort.','No universal runtime or target-attainment guarantee; interrupted CG remains uncertified.','Existing graph construction excluded; authentication and graph loading recorded separately.'])
    write(root/'manifest.json',manifest);print(json.dumps(dict(root=str(root),manifest_sha256=sha(root/'manifest.json'),inputs=len(assets),runs=len(runs))))

if __name__=='__main__':main()
