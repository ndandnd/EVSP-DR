#!/usr/bin/env python3
"""Prepare or execute eight pinned, matched-fleet event tariff pilot cells."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, platform, shutil, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from compare_original_giro_charging import sha
from freeze_terminal_exact_cg_pool import TERMINAL_STOPS, terminal_artificials

def write_json(path,value):
    path=Path(path); temporary=path.with_name('.'+path.name+'.tmp.'+str(os.getpid()))
    try:
        with temporary.open('x') as f:
            json.dump(value,f,sort_keys=True,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.link(temporary,path)
    finally: temporary.unlink(missing_ok=True)

def git(*args): return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
def check_identity(commit):
    if git('rev-parse','HEAD')!=commit or git('status','--porcelain','--untracked-files=no'):
        raise ValueError('execution commit or tracked contents mismatch')
    sym=subprocess.run(['git','-C',str(ROOT),'symbolic-ref','-q','HEAD'],capture_output=True)
    if sym.returncode!=1: raise ValueError('clean detached checkout required')

def prepare(args):
    check_identity(args.commit)
    if args.cg_seconds<=0 or args.mip_seconds<=0 or args.cg_seconds+args.mip_seconds>28800:
        raise ValueError('positive CG/MIP budgets totaling at most eight hours required')
    if args.root.exists(): raise FileExistsError(args.root)
    from make_duty_pair_instances import _peak_concurrency
    import pandas as pd
    tariff_ids=getattr(args,'tariffs',('flat','peak08'))
    with (ROOT/'data/tariff_response/tariff_manifest.csv').open() as handle:
        manifest={row['tariff_id']:row for row in csv.DictReader(handle)}
    for tariff in tariff_ids:
        if sha(ROOT/manifest[tariff]['relative_path'])!=manifest[tariff]['sha256']:
            raise ValueError('tracked tariff manifest hash mismatch')
    cells=[]
    for label,k,power,sub in [('easy',k,240,'easy_trip_nested_20260908') for k in (5,8,10)]+[('original_eligible',5,350,'original_replay_eligible_20260908')]:
        instance=ROOT/f'data/scale_ladder/instances/{sub}/Practice_Custom_DutyUnion_{label}_k{k:02d}_20260908.csv'
        frame=pd.read_csv(instance); peak=_peak_concurrency(frame)
        if peak!=k: raise ValueError('peak fleet lower bound is not the declared matched fleet')
        for tariff in tariff_ids:
            prices=ROOT/f'data/tariff_response/{tariff}_h26.csv'
            cells.append(dict(index=len(cells),cell=f'{label}_k{k:02d}_g240_p{power}_{tariff}',
                fleet=k,trips=len(frame),peak_concurrency=peak,g_kwh=240,charge_kw=power,
                instance=str(instance),instance_sha256=sha(instance),tariff=str(prices),tariff_sha256=sha(prices),
                literal_original_required=label=='original_eligible'))
    plan=dict(schema='evsp-dr-matched-tariff-pilot-v1',commit=args.commit,cells=cells,
        cg_seconds=args.cg_seconds,mip_seconds=args.mip_seconds,
        master_sha256=sha(ROOT/'data/Par_VehicleDetails_Updated.csv'),
        reference_sha256={name:sha(ROOT/'data'/name) for name in ('Ref_dict.csv','par_ref_dhd.csv')},
        semantics='GIRO-AUGMENTED event pool; k proved by overlap bound plus validated k-duty witness; charging objective conservative expanded grid; physical realized invoices reported separately',
        terminal_policy='full initial240; returnSOC>=0; actual surplus reported; no forced restoration',
        power_scenario='350kW is a declared comparison scenario, not a claim about actual rated hardware; shared station capacity is not modeled',
        requeue=False,threads=8,memory='48G' if getattr(args,'split_stages',False) else '96G',
        layout='split_cg_mip' if getattr(args,'split_stages',False) else 'sequential',
        tariff_manifest_sha256=sha(ROOT/'data/tariff_response/tariff_manifest.csv'),tariff_ids=list(tariff_ids),
        stage_resources={'cg':{'partition':'default_partition','cpus':1,'memory':'16G'},
                         'mip':{'partition':'scaglione','cpus':8,'memory':'48G','exclude':['scaglione-compute-01','scaglione-cpu-04']}},
        network_build_excluded_from_cg_budget=True)
    args.root.mkdir(parents=True); (args.root/'logs').mkdir(); write_json(args.root/'plan.json',plan)
    print(json.dumps({'root':str(args.root),'cells':len(cells),'plan_sha256':sha(args.root/'plan.json')}))

def freeze(source,destination,cell,plan,seed):
    before=sha(source); status=json.loads(source.read_text()); p=status.get('provenance') or {}
    expected={'time_model':'event','soc_step':2.5,'block_min':5,'g_kwh':240.,'charge_kw':float(cell['charge_kw']),
        'min_soc_frac':0.,'master_sense':'partition','column_pool_treatment':'GIRO-AUGMENTED',
        'strict_tariff_coverage':True,'validated_seed_routes_sha256':sha(seed)}
    if any(status.get(k)!=v for k,v in expected.items()): raise ValueError('terminal CG configuration mismatch')
    if p.get('git_commit')!=plan['commit'] or p.get('instance_sha256')!=cell['instance_sha256'] or p.get('prices_sha256')!=cell['tariff_sha256']:
        raise ValueError('terminal CG identity mismatch')
    if status.get('stop_reason') not in TERMINAL_STOPS: raise ValueError('CG not terminal (external signal is censored)')
    if status.get('stop_reason')=='certified' and status.get('certified_rc_optimal') is not True: raise ValueError('missing RC certificate')
    artificial,_=terminal_artificials(status)
    if not math.isfinite(artificial) or not 0<=artificial<=1e-7: raise ValueError('terminal CG has artificials')
    journal=Path(str(source)+'.columns.jsonl'); journal_sha=sha(journal)
    frozen_journal=Path(str(destination)+'.columns.jsonl'); temporary=frozen_journal.with_suffix('.tmp')
    unique=set(); records=0; previous=-1
    with journal.open('rb') as source_file, temporary.open('xb') as target:
        for line in source_file:
            if not line.endswith(b'\n'): raise ValueError('incomplete journal')
            r=json.loads(line); trips=r['trips']; found=r.get('found_iter',0)
            if not trips or len(trips)!=len(set(trips)) or any(type(t) is not int for t in trips) or type(found) is not int or found<previous or not math.isfinite(float(r['cost'])): raise ValueError('invalid journal record')
            previous=found; unique.add(frozenset(trips)); records+=1; target.write(line)
        target.flush(); os.fsync(target.fileno())
    if len(unique)!=status['columns'] or before!=sha(source) or journal_sha!=sha(journal) or journal_sha!=sha(temporary): raise ValueError('CG source changed or pool count mismatch')
    os.link(temporary,frozen_journal); temporary.unlink()
    status['columns_journal']=str(frozen_journal)
    status['terminal_tariff_pool_snapshot']={'source_status_sha256':before,'source_journal_sha256':journal_sha,'journal_records':records,'cell':cell['cell']}
    write_json(destination,status)
    return sha(destination),journal_sha

def route_accounting(routes,problem,power):
    from config import CHARGE_START_COST
    arc={(u,v):e for u,arcs in problem.adjacency.items() for v,t,e,kind in arcs}
    charged=sum(sum(r['charging_stops'].get('kwh',[])) for r in routes)
    deadhead=sum(sum(arc[(u,v)] for u,v in zip(r['route_nodes'],r['route_nodes'][1:]) if u!=v) for r in routes)
    service=sum(problem.trip_energy[t] for r in routes for t in r['trips'])
    fees=sum(len(r['charging_stops'].get('stations',[])) for r in routes)*CHARGE_START_COST
    cost=sum(r['continuous_realized_cost'] for r in routes)-100000.*len(routes)
    return dict(fleet=len(routes),physical_charging_cost=cost,charge_start_fees=fees,energy_invoice=cost-fees,
        charged_kwh=charged,production_energy_consumed_kwh=deadhead+service,
        terminal_surplus_kwh=240.*len(routes)+charged-deadhead-service,
        conservative_grid_charging_cost=sum(r['expanded_grid_cost'] for r in routes)-100000.*len(routes))

def worker(args):
    plan_path=args.root/'plan.json'
    if sha(plan_path)!=os.environ['EVSP_PLAN_SHA256']: raise ValueError('plan hash mismatch')
    plan=json.loads(plan_path.read_text()); check_identity(plan['commit'])
    if os.environ.get('SLURM_RESTART_COUNT','0')!='0': raise ValueError('MIP tree cannot resume; allocation is censored')
    stage=getattr(args,'stage','all')
    if plan.get('layout')=='split_cg_mip' and stage=='all':
        raise ValueError('split plan requires an explicit CG or MIP worker')
    cell=plan['cells'][args.index]; folder=args.root/cell['cell']
    if stage in ('all','cg'): folder.mkdir()
    elif not folder.is_dir(): raise ValueError('CG cell directory missing')
    write_json(folder/('allocation.json' if stage=='all' else stage+'.allocation.json'),dict(hostname=platform.node(),platform=platform.platform(),
        cpu_count=os.cpu_count(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),slurm_array_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
        cpuinfo=Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else None))
    for key in ('instance','tariff'):
        if sha(cell[key])!=cell[key+'_sha256']: raise ValueError('input hash mismatch')
    for name,digest in plan['reference_sha256'].items():
        if sha(ROOT/'data'/name)!=digest: raise ValueError('reference hash mismatch')
    if sha(ROOT/'data/Par_VehicleDetails_Updated.csv')!=plan['master_sha256']: raise ValueError('master hash mismatch')
    def call(script,*argv):
        started=time.monotonic(); success=False
        try:
            subprocess.run([sys.executable,'-u',str(ROOT/'src'/script),*map(str,argv)],check=True,cwd=ROOT)
            success=True
        finally:
            write_json(folder/(script+'.timing.json'),dict(elapsed_s=time.monotonic()-started,success=success))
    shared=['--instance',cell['instance'],'--instance-sha256',cell['instance_sha256'],'--master-sha256',plan['master_sha256'],'--fleet',cell['fleet'],'--g-kwh',240,'--charge-kw',cell['charge_kw']]
    original=folder/'original.json'
    seed=folder/'seed.json'; cache=folder/'network.pkl'; cg=folder/'cg.json'; snapshot=folder/'snapshot.json'; mip=folder/'mip.json'
    frozen_record=folder/'FROZEN.json'
    if stage in ('all','cg'):
        call('compare_original_giro_charging.py',*shared,'--tariffs',cell['tariff'],'--terminal-energy-price',0.1,'--out',original)
        original_summary=json.loads(original.read_text())['summary'][0]
        if cell['literal_original_required'] and not original_summary['matched_physics_comparator_eligible']: raise ValueError('original comparator failed combined-instance replay')
        call('prepare_event_giro_seed.py',*shared,'--tariff',cell['tariff'],'--tariff-sha256',cell['tariff_sha256'],'--network-cache',cache,'--out',seed)
        call('exact_pricer_expanded.py','--csv',cell['instance'],'--prices_csv',cell['tariff'],'--time-model','event','--event-arc-mode','lazy','--event-network-cache',cache,'--event-network-cache-mode','require','--soc-step',2.5,'--block-min',5,'--max-iters',50000,'--columns_per_iter',30,'--column-selection','reduced_cost','--column-diversity-weight',0.,'--column-candidate-multiplier',4,'--rc-eps',0.0001,'--master-sense','partition','--master-backend','gurobi','--initial-pool','singletons','--wall-limit-s',plan['cg_seconds'],'--checkpoint-every',25,'--g-kwh',240,'--charge-kw',cell['charge_kw'],'--min-soc-frac',0,'--validated-seed-routes',seed,'--augmentation-label','GIRO-AUGMENTED','--strict-tariff-coverage','--phase-telemetry',folder/'cg.phase-telemetry.jsonl','--gurobi-log',folder/'cg.gurobi.log','--out',cg)
        freeze_started=time.monotonic()
        snapshot_sha,journal_sha=freeze(cg,snapshot,cell,plan,seed)
        write_json(folder/'freeze.timing.json',dict(elapsed_s=time.monotonic()-freeze_started,success=True))
        write_json(frozen_record,dict(cell=cell['cell'],commit=plan['commit'],
            plan_sha256=sha(plan_path),snapshot_sha256=snapshot_sha,journal_sha256=journal_sha,
            seed_sha256=sha(seed),original_sha256=sha(original)))
        if stage=='cg': return
    else:
        record=json.loads(frozen_record.read_text())
        if record['cell']!=cell['cell'] or record['commit']!=plan['commit'] or record['plan_sha256']!=sha(plan_path):
            raise ValueError('frozen stage identity mismatch')
        for path,key in ((snapshot,'snapshot_sha256'),(Path(str(snapshot)+'.columns.jsonl'),'journal_sha256'),
                         (seed,'seed_sha256'),(original,'original_sha256')):
            if sha(path)!=record[key]: raise ValueError('frozen stage artifact hash mismatch')
        snapshot_sha=record['snapshot_sha256'];journal_sha=record['journal_sha256']
        original_summary=json.loads(original.read_text())['summary'][0]
    os.environ.update(EVSP_MIP_EXPECTED_RESULT_SHA256=snapshot_sha,EVSP_MIP_EXPECTED_JOURNAL_SHA256=journal_sha,EVSP_MIP_EXPECTED_INITIAL_PARTITION_SHA256=sha(seed))
    call('run_exact_pool_mip.py','--result',snapshot,'--data-dir',ROOT/'data','--reference-data-dir',ROOT/'data','--initial-partition-routes',seed,'--verified-expanded-initial-partition','--two-stage','--timelimit',plan['mip_seconds'],'--mipgap',0.0001,'--threads',8,'--progress-dir',folder/'mip_progress','--gurobi-log',folder/'mip.gurobi.log','--out',mip)
    result=json.loads(mip.read_text())
    if result.get('buses')!=cell['fleet'] or result.get('fleet_proven') is not True: raise ValueError('matched fleet not proved by MIP')
    from audit_giro_known_columns import build_problem,HORIZON_MIN
    path=Path(cell['instance']); problem=build_problem(path.parent,path.name,max_station_to_trip_wait_min=HORIZON_MIN,reference_data_dir=ROOT/'data')
    fixed=route_accounting(json.loads(seed.read_text())['routes'],problem,cell['charge_kw'])
    joint=route_accounting(result['selected_routes'],problem,cell['charge_kw'])
    from config import CHARGING_STATIONS
    from utils_v2 import load_station_hourly_prices
    price_values={float(value) for curve in load_station_hourly_prices(cell['tariff'],CHARGING_STATIONS).values() for value in curve.values()}
    flat_price=next(iter(price_values)) if len(price_values)==1 else None
    lower=original_summary['charging_cost_lower']
    summary=dict(cell=cell,fixed_duties_reoptimized=fixed,joint=joint,original=original_summary,
        beats_original_robustly=bool(original_summary['matched_physics_comparator_eligible'] and lower is not None and joint['physical_charging_cost']<lower-1e-7),
        fixed_minus_joint_charging_cost=fixed['physical_charging_cost']-joint['physical_charging_cost'],
        fixed_minus_joint_terminal_surplus_kwh=fixed['terminal_surplus_kwh']-joint['terminal_surplus_kwh'],
        fixed_minus_joint_after_terminal_credit_0p1=(fixed['physical_charging_cost']-.1*fixed['terminal_surplus_kwh'])-(joint['physical_charging_cost']-.1*joint['terminal_surplus_kwh']),
        actual_flat_price=flat_price,
        original_minus_joint_after_flat_terminal_credit=(lower-joint['physical_charging_cost']-flat_price*(original_summary['terminal_surplus_total_kwh']-joint['terminal_surplus_kwh']) if flat_price is not None and lower is not None and original_summary['terminal_surplus_total_kwh'] is not None else None),
        fixed_minus_joint_after_flat_terminal_credit=(fixed['physical_charging_cost']-joint['physical_charging_cost']-flat_price*(fixed['terminal_surplus_kwh']-joint['terminal_surplus_kwh']) if flat_price is not None else None),
        fee_scope='Five currency units per charging start is a modeled penalty, not a verified original invoice charge',
        cost_optimal_scope=result.get('optimal_scope'),absolute_cost_gap=result.get('absolute_cost_gap'),
        limitation='Finite augmented pool; expanded-grid objective. Realized invoices are replayed outcomes, not a continuous-cost optimality certificate. Terminal-credit adjustment is accounting sensitivity only.')
    write_json(folder/'comparison.json',summary)
    write_json(folder/'COMPLETE.json',dict(mip_sha256=sha(mip),comparison_sha256=sha(folder/'comparison.json')))

def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--root',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--cg-seconds',type=int,default=14400);p.add_argument('--mip-seconds',type=int,default=14400);p.add_argument('--tariffs',nargs='+',choices=('flat','peak08','peak12','peak18'),default=['flat','peak08']);p.add_argument('--split-stages',action='store_true')
    p=sub.add_parser('worker');p.add_argument('--root',type=Path,required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--stage',choices=('all','cg','mip'),default='all')
    args=parser.parse_args();args.root=args.root.expanduser().resolve()
    (prepare if args.mode=='prepare' else worker)(args)
if __name__=='__main__': main()
