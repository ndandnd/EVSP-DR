#!/usr/bin/env python3
"""Prepare and execute the four-cell, hash-bound GREEDY event campaign."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,platform,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SCHEMA='evsp-dr-greedy-event-campaign-v1'
TARIFF='hourly_prices_flat.csv'
CODE=('scripts/event_uniform_envelope/greedy_campaign.py','scripts/event_uniform_envelope/greedy_campaign.sub',
      'src/prepare_event_greedy_seed.py','src/exact_pricer_expanded.py','src/event_pricer_network.py',
      'src/freeze_terminal_exact_cg_pool.py','src/run_exact_pool_mip.py')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def publish(path,value):
    path=Path(path);temporary=path.with_name('.'+path.name+'.tmp.'+str(os.getpid()))
    try:
        with temporary.open('x') as f:
            json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.link(temporary,path)
    finally:temporary.unlink(missing_ok=True)

def identity(commit):
    def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    if git('rev-parse','HEAD')!=commit or git('status','--porcelain','--untracked-files=no'):
        raise ValueError('execution commit or tracked contents mismatch')
    if subprocess.run(['git','-C',str(ROOT),'symbolic-ref','-q','HEAD'],capture_output=True).returncode!=1:
        raise ValueError('detached execution checkout required')

def prepare(args):
    identity(args.commit)
    if args.root.exists():raise FileExistsError(args.root)
    cells=[]; manifests={}
    for family,names in [('nested_probability_k2_15_20260908',('k03_p2','k05_p2','k06_p1')),
                         ('easy_trip_nested_20260908',('easy_k10',))]:
        source=ROOT/'data/scale_ladder/instances'/family/'selection_manifest.csv'
        with source.open() as f:rows={r['cell_id']:r for r in csv.DictReader(f)}
        manifests[str(source.relative_to(ROOT))]=sha(source)
        for name in names:
            row=rows[name];instance=ROOT/row['relative_path']
            if sha(instance)!=row['instance_file_sha256']:raise ValueError('instance manifest hash mismatch')
            with instance.open() as f:trips=sum(1 for _ in csv.DictReader(f))
            if trips!=int(row['trip_count']):raise ValueError('trip count mismatch')
            scale=int(row['scale']);small=scale<=5
            cells.append(dict(index=len(cells),cell=name,scale=scale,trips=trips,
                instance=str(instance),instance_relative_to_data=str(instance.relative_to(ROOT/'data')),
                instance_sha256=sha(instance),source_manifest=str(source.relative_to(ROOT)),
                cg_seconds=28800,mip_seconds=28800 if small else 3600,
                fleet_timelimit=None if small else 1800,cost_timelimit=None if small else 1800,
                allow_unproven_fleet_cost=not small,
                mip_policy='proof_first_8h' if small else '30min_fleet_plus_30min_conditional_cost'))
    if len(cells)!=4:raise ValueError('expected exactly four cells')
    # Ensure the selected checkout implements both explicit partition preservation and stage caps.
    solver=(ROOT/'src/run_exact_pool_mip.py').read_text()
    for token in ('--allow-unproven-fleet-cost','--fleet-timelimit','--cost-timelimit','--verified-expanded-initial-partition'):
        if token not in solver:raise ValueError('execution MIP interface incomplete: '+token)
    if 'candidate["expanded_grid_charging_stops"]' not in solver:
        raise ValueError('execution MIP lacks expanded partition-stop preservation')
    plan=dict(schema=SCHEMA,execution_commit=args.commit,cells=cells,column_pool_treatment='GREEDY',
        tariff=TARIFF,tariff_sha256=sha(ROOT/'data'/TARIFF),selection_manifest_sha256=manifests,
        reference_sha256={name:sha(ROOT/'data'/name) for name in ('Ref_dict.csv','par_ref_dhd.csv')},
        code_sha256={name:sha(ROOT/name) for name in CODE},
        physics=dict(g_kwh=240,charge_kw=240,reserve_kwh=0,soc_step=2.5,block_min=5,time_model='event',event_arc_mode='lazy'),
        cg=dict(rc_eps=.0001,master_sense='partition',master_backend='gurobi',initial_pool='singletons',columns_per_iter=30,column_selection='reduced_cost',column_diversity_weight=0.,column_candidate_multiplier=4),
        seed_semantics='GREEDY proposes trip sequences; event graph reoptimizes charging and splits unrepresentable sequences into feasible prefixes; no minimum-fleet certificate',
        mip_start='explicit physically replayed seed partition preserving expanded-grid costs',
        strict_tariff_coverage=False,tariff_extension='existing production flat last-hour extension',
        resources=dict(cache=dict(partition='default_partition',cpus=1,mem='96G',allocation='04:00:00'),cg=dict(partition='default_partition',cpus=1,mem='96G',allocation='08:30:00'),mip=dict(partition='scaglione',cpus=8,mem='48G',exclude=['scaglione-compute-01','scaglione-cpu-04'],allocation='10:00:00')),
        mip_concurrency=2,requeue=False)
    args.root.mkdir(parents=True);(args.root/'logs').mkdir();publish(args.root/'plan.json',plan)
    print(json.dumps(dict(root=str(args.root),plan_sha256=sha(args.root/'plan.json'),cells=len(cells))))

def checked_record(path,expected,artifacts):
    record=json.loads(Path(path).read_text())
    if any(record.get(k)!=v for k,v in expected.items()):raise ValueError('stage record identity mismatch')
    for file,key in artifacts:
        if sha(file)!=record[key]:raise ValueError('stage artifact changed: '+str(file))
    return record

def worker(args):
    plan_path=args.root/'plan.json'
    if sha(plan_path)!=os.environ['EVSP_PLAN_SHA256']:raise ValueError('plan hash mismatch')
    plan=json.loads(plan_path.read_text());identity(plan['execution_commit'])
    if plan['schema']!=SCHEMA or os.environ.get('SLURM_RESTART_COUNT','0')!='0':raise ValueError('invalid campaign or attempted restart')
    for name,digest in plan['code_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('code hash mismatch: '+name)
    for name,digest in plan['reference_sha256'].items():
        if sha(ROOT/'data'/name)!=digest:raise ValueError('reference hash mismatch')
    if sha(ROOT/'data'/TARIFF)!=plan['tariff_sha256']:raise ValueError('tariff hash mismatch')
    cell=plan['cells'][args.index]
    if cell['index']!=args.index or sha(cell['instance'])!=cell['instance_sha256']:raise ValueError('cell identity mismatch')
    folder=args.root/cell['cell']
    if args.stage=='cache':folder.mkdir()
    if not folder.is_dir():raise ValueError('cache stage directory missing')
    publish(folder/(args.stage+'.allocation.json'),dict(host=platform.node(),platform=platform.platform(),job_id=os.environ.get('SLURM_JOB_ID'),array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),cpuinfo=Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else None))
    def call(script,*argv):
        started=time.monotonic();success=False
        try:
            subprocess.run([sys.executable,'-u',str(ROOT/'src'/script),*map(str,argv)],check=True,cwd=ROOT);success=True
        finally:publish(folder/(args.stage+'.'+script+'.timing.json'),dict(elapsed_s=time.monotonic()-started,success=success))
    cache=folder/'network.pkl';manifest=Path(str(cache)+'.manifest.json');seed=folder/'seed.json';cg=folder/'cg.json';snapshot=folder/'snapshot.json';freeze_record=folder/'freeze.json';out=folder/'mip.json'
    expected=dict(cell=cell['cell'],commit=plan['execution_commit'],plan_sha256=sha(plan_path))
    graph_args=['--csv',cell['instance'],'--prices_csv',TARIFF,'--time-model','event','--event-arc-mode','lazy','--event-network-cache',cache,'--soc-step',2.5,'--block-min',5,'--g-kwh',240,'--charge-kw',240,'--min-soc-frac',0]
    if args.stage=='cache':
        call('exact_pricer_expanded.py',*graph_args,'--event-network-cache-only')
        call('prepare_event_greedy_seed.py','--repo',ROOT,'--data-dir',ROOT/'data','--csv',cell['instance_relative_to_data'],'--prices-csv',TARIFF,'--event-network-cache',cache,'--g-kwh',240,'--charge-kw',240,'--reserve-kwh',0,'--soc-step',2.5,'--block-min',5,'--out',seed)
        payload=json.loads(seed.read_text())
        if payload.get('schema')!='evsp-dr-event-greedy-partition-v1' or payload.get('source')!='GREEDY' or payload.get('exact_trip_partition') is not True or payload['input_hashes']['instance_sha256']!=cell['instance_sha256']:raise ValueError('unexpected seed identity')
        publish(folder/'CACHE_READY.json',dict(**expected,cache_sha256=sha(cache),cache_manifest_sha256=sha(manifest),seed_sha256=sha(seed),greedy_seed_routes=payload['route_count']))
        return
    ready=checked_record(folder/'CACHE_READY.json',expected,[(seed,'seed_sha256'),(manifest,'cache_manifest_sha256')])
    if args.stage=='cg':
        if sha(cache)!=ready['cache_sha256']:raise ValueError('network cache changed')
        call('exact_pricer_expanded.py',*graph_args,'--event-network-cache-mode','require','--max-iters',50000,'--columns_per_iter',30,'--column-selection','reduced_cost','--column-diversity-weight',0,'--column-candidate-multiplier',4,'--rc-eps',.0001,'--master-sense','partition','--master-backend','gurobi','--initial-pool','singletons','--wall-limit-s',cell['cg_seconds'],'--checkpoint-every',25,'--validated-seed-routes',seed,'--augmentation-label','GREEDY','--phase-telemetry',folder/'cg.phase-telemetry.jsonl','--gurobi-log',folder/'cg.gurobi.log','--out',cg)
        if sha(seed)!=ready['seed_sha256']:raise ValueError('seed changed during CG')
        call('freeze_terminal_exact_cg_pool.py','--base-status',cg,'--resume-status',cg,'--cell',cell['cell'],'--instance-relative-to-data',cell['instance_relative_to_data'],'--instance-sha256',cell['instance_sha256'],'--source-solver-commit',plan['execution_commit'],'--expected-column-pool-treatment','GREEDY','--expected-seed-sha256',ready['seed_sha256'],'--out',snapshot,'--record',freeze_record)
        frozen=json.loads(freeze_record.read_text())
        publish(folder/'FROZEN_READY.json',dict(**expected,seed_sha256=ready['seed_sha256'],snapshot_sha256=frozen['snapshot_sha256'],journal_sha256=frozen['journal_sha256'],freeze_record_sha256=sha(freeze_record)))
        return
    frozen=checked_record(folder/'FROZEN_READY.json',expected,[(seed,'seed_sha256'),(snapshot,'snapshot_sha256'),(Path(str(snapshot)+'.columns.jsonl'),'journal_sha256'),(freeze_record,'freeze_record_sha256')])
    os.environ.update(EVSP_MIP_EXPECTED_RESULT_SHA256=frozen['snapshot_sha256'],EVSP_MIP_EXPECTED_JOURNAL_SHA256=frozen['journal_sha256'],EVSP_MIP_EXPECTED_INITIAL_PARTITION_SHA256=frozen['seed_sha256'])
    stage_flags=[] if cell['fleet_timelimit'] is None else ['--fleet-timelimit',cell['fleet_timelimit'],'--cost-timelimit',cell['cost_timelimit'],'--allow-unproven-fleet-cost']
    call('run_exact_pool_mip.py','--result',snapshot,'--data-dir',ROOT/'data','--reference-data-dir',ROOT/'data','--initial-partition-routes',seed,'--verified-expanded-initial-partition','--two-stage','--timelimit',cell['mip_seconds'],*stage_flags,'--threads',8,'--mipgap',.0001,'--progress-dir',folder/'mip_progress','--gurobi-log',folder/'mip.gurobi.log','--out',out)
    result=json.loads(out.read_text())
    if result.get('mip_start',{}).get('source_sha256')!=frozen['seed_sha256'] or result.get('mip_start',{}).get('kind')!='validated_exact_partition':raise ValueError('explicit GREEDY MIP start not preserved')
    if result.get('buses') is None or result['buses']>ready['greedy_seed_routes']:
        raise ValueError('final incumbent is worse than validated GREEDY fleet')
    publish(folder/'COMPLETE.json',dict(**expected,mip_sha256=sha(out),seed_sha256=frozen['seed_sha256'],column_pool_treatment='GREEDY',buses=result.get('buses'),fleet_proven=result.get('fleet_proven'),optimal_scope=result.get('optimal_scope'),stage_policy=cell['mip_policy']))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    x=sub.add_parser('prepare');x.add_argument('--root',type=Path,required=True);x.add_argument('--commit',required=True)
    x=sub.add_parser('worker');x.add_argument('--root',type=Path,required=True);x.add_argument('--index',type=int,required=True);x.add_argument('--stage',choices=('cache','cg','mip'),required=True)
    a=p.parse_args();a.root=a.root.expanduser().resolve();(prepare if a.mode=='prepare' else worker)(a)
if __name__=='__main__':main()
