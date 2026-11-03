"""Preregister and execute the matched C2/C3/C6 k8 SOC-grid sensitivity on Unicorn."""
import argparse,csv,datetime,hashlib,json,os,shlex,subprocess,sys
from pathlib import Path

ROOT=Path.home()/'ladder-lite'; OWNER=ROOT/'capacity_v15_20261002'; REPO=OWNER/'code'; SRC=REPO/'src'
HERE=Path(__file__).resolve().parent; BASE=HERE.parent
PY=Path.home()/'evsp_env/bin/python'; SLURM=Path('/usr/local/slurm/slurm-25.05.5/bin');COMMIT='3a0ab6a8ab50e4af7d5a7be7688d5ffa1fbbf78c'
CASES=[('c2_k08',7,920637),('c3_k08',11,920641),('c6_k08',23,920653)]
POLICY='f9c5982718da6ab758019ceb85e31b539aea5416adc51816bc4bae220728e40b'; TASKS='16e56812dfefae666c33a90845d0850bda9b9083fc087045fa91034c6e1d2d32';BATCH='bd726a5423082793fbb6d12a8455a09d6cbc6dcd57d538e6cc62bc66925469f2'

def check(c,msg):
    if not c:raise ValueError(msg)

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def save(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2,sort_keys=True);f.write('\n')

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def cmd(*args):return subprocess.check_output(list(map(str,args)),text=True).strip()

def bind():
    check(cmd('git','-C',REPO,'rev-parse','HEAD')==COMMIT,'wrong execution commit');subprocess.run(['git','-C',REPO,'diff','--quiet','HEAD','--','src'],check=True)
    # Python imports can also be shadowed by an untracked file in src.
    check(not cmd('git','-C',REPO,'ls-files','--others','--exclude-standard','--','src/*.py'),'untracked source overlay')
    check(digest(OWNER/'tasks.tsv')==TASKS and digest(OWNER/'run.sbatch')==BATCH,'owner campaign changed')
    check(digest(ROOT/'SCAGLIONE_RESOURCE_POLICY.md')==POLICY,'policy changed')

def row(case,index):
    with (OWNER/'tasks.tsv').open() as f: rows=list(csv.DictReader(f,delimiter='\t'))
    r=rows[index];check((r['idx'],r['case'],r['arm'],r['family'],r['script'])==(str(index),case,'parx60','chain','run_capacity_terminal_cg.py'),'incorrect frozen row')
    args=shlex.split(r['args']); expected={'--target':'0','--fleet-cap':'12','--cg-seconds':'10800','--dive-seconds':'5400','--mip-seconds':'3600','--charge-kw':'240','--battery-kwh':'240','--start-fee':'5','--station-kw':'PARX=60'}
    flags=[v for v in args if v.startswith('--')];check(len(set(flags))==len(flags) and set(flags)==set(expected)|{'--data-dir','--csv','--prices','--non-strict-tariff'},'duplicate or unexpected frozen flag')
    check(all(args[args.index(k)+1]==v for k,v in expected.items()),'wrong frozen settings')
    return args

def check_inputs(summary):
    hashes={Path(k).name:digest(Path(k)) for k in summary['input_hashes']};check(hashes=={Path(k).name:h for k,h in summary['input_hashes'].items()},'owner inputs mismatched')
    return hashes

def replay(path,step,hash_file,receipt):
    env=dict(os.environ,PYTHONPATH=str(SRC),PYTHONDONTWRITEBYTECODE='1');subprocess.run([PY,HERE/'replay.py','--run',path,'--output',receipt,'--soc-step',str(step),'--start-fee','5','--expected-input-hashes',hash_file],env=env,check=True)

def register():
    bind(); rows=[]; reg=BASE/'registration'; reg.mkdir(exist_ok=False)
    for case,index,job in CASES:
        args=row(case,index);path=OWNER/'results/chain'/case/'parx60'/f'job{job}_r0';s=json.loads((path/'summary.json').read_text());hashes=check_inputs(s)
        check(s['dive_settings']=={'seconds':5400.,'theta':.7} and s['smoothing']==0 and not s['seed_routes'] and 'integer_pricing_settings' not in s,'control search not matched')
        src_inst=Path(args[args.index('--data-dir')+1])/args[args.index('--csv')+1];check(src_inst==Path(s['inputs']['instance']) and Path(args[args.index('--prices')+1])==Path(s['inputs']['tariff']),'archived inputs mismatch')
        sa=cmd(SLURM/'sacct','-XnP','-S','2026-10-02','-E','2026-10-05','-j','920164','--format=JobIDRaw,JobID,State,ExitCode,Partition,AllocCPUS,Elapsed'); hits=[r.split('|') for r in sa.splitlines() if r.startswith(str(job)+'|')];check(len(hits)==1 and hits[0][1:6]==[f'920164_{index}','COMPLETED','0:0','default_partition','8'],'control scheduler accounting mismatch')
        hashes_path=reg/f'{case}_input_hashes.json';save(hashes_path,hashes);receipt=reg/f'{case}_control_replay.json';replay(path,2.5,hashes_path,receipt)
        rec=json.loads(receipt.read_text());rows.append(dict(case=case,owner_array_task=f'920164_{index}',owner_raw_job=job,owner_attempt=0,control_sacct=hits[0][2:],input_hashes=hashes,replay_receipt_sha256=digest(receipt),summary_sha256=digest(path/'summary.json'),control_root_sha256=rec['root_prefix_sha256'],control_post_dive_pool_sha256=rec['final_pool_sha256'],owner_instance='Practice_Custom_DutyUnion_k08_p'+case[1].zfill(2)+'_20260908.csv',control_grid=2.5,treatment_grid=5.,new_runs=1))
    files={f.name:digest(f) for f in sorted(HERE.iterdir()) if f.is_file()};report={'schema':'c236_grid_protocol_v1','registered_utc':utc(),'execution_commit':COMMIT,'owner_tasks_sha256':TASKS,'owner_batch_sha256':BATCH,'policy_sha256':POLICY,'protocol_files_sha256':files,'cases':rows,'design':'preselected C2,C3,C6 k8; archived default-partition controls; fresh singleton start, no external routes, caches, predecessors or deck/holdout gates; one new arm per case; covering trip master, stage1 fleet then stage2 charging cost incl start fees; clocks and host allocation can affect time-limited dives','sole_scientific_switch':'--soc-step 5.0 versus frozen default 2.5','resources':{'partition':'default_partition','exclude':'scaglione-compute-01','nice':0,'cpus':8,'ram_gib':48,'hours':20,'concurrency':3,'requeue':True,'budget_seconds':{'cg':10800,'simple_dive':5400,'pool_mip':3600}},'initialization':'new process, fresh in-memory graph and internal singleton columns for every new run. No persistent graph cache and no previous-k dependency','outcomes':'Report separately graph wall/size, root weighted conservative LP pricing certificate and route-weight; finite-pool fleet, bound and economic cost; independent selected-route replay and exact-once and modeled plug validation. Compare charging at equal fleet only. Do not claim SOC is continuous, target-blind full-model optimality, or runtime significance from historical single attempts. Missing certificates and no selection are outcomes. No optimistic grid bound is requested. No changes to held jobs or future k10 decisions.'};save(reg/'manifest.json',report);print(json.dumps({'registered':True,'manifest_sha256':digest(reg/'manifest.json'),'cases':len(rows),'controls_replayed':True}))

def execute(task,expected_manifest):
    bind();reg=BASE/'registration'; mpath=reg/'manifest.json';check(digest(mpath)==expected_manifest,'registration hash mismatch');m=json.loads(mpath.read_text());check(all(digest(HERE/k)==v for k,v in m['protocol_files_sha256'].items()),'protocol file drift')
    case,index,_=CASES[task];a=row(case,index);info=m['cases'][task];check(info['case']==case and int(os.environ['SLURM_CPUS_PER_TASK'])==8,'worker or case mismatch');src_data=Path(a[a.index('--data-dir')+1]);files=[src_data/a[a.index('--csv')+1],Path(a[a.index('--prices')+1]),src_data/'Ref_dict.csv',src_data/'par_ref_dhd.csv'];check({p.name:digest(p) for p in files}==info['input_hashes'],'scientific inputs drift')
    dest=BASE/'results'/case/'soc_5'/('job'+os.environ['SLURM_JOB_ID']+'_r'+os.environ.get('SLURM_RESTART_COUNT','0'));dest.mkdir(parents=True,exist_ok=False); result=dest/'solve';argv=[str(PY),'-u','run_capacity_terminal_cg.py',*a,'--soc-step','5.0','--out',str(result)];save(dest/'start.json',{'utc':utc(),'manifest_sha256':expected_manifest,'case':case,'grid_kwh':5.,'argv':argv,'job':os.environ['SLURM_JOB_ID'],'restart':os.environ.get('SLURM_RESTART_COUNT','0')})
    proc=subprocess.run(argv,cwd=SRC);rc=proc.returncode
    try:
        check(rc in (0,3),'solver crashed'); replay(result,5.,reg/f'{case}_input_hashes.json',dest/'replay.json');audit=json.loads((dest/'replay.json').read_text());check((rc==0)==(audit['selection_state']=='selected'),'wrapper exit mismatches selection');final={'solver_rc':rc,'validated':True,'replay_sha256':digest(dest/'replay.json'),'summary_sha256':audit['summary_sha256'],'state':audit['selection_state']}
    except Exception:
        save(dest/'exit.json',{'solver_rc':rc,'validated':False,'utc':utc()});raise
    save(dest/'exit.json',dict(final,utc=utc()));return 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','worker']);p.add_argument('--task',type=int,choices=range(3));p.add_argument('--manifest-sha256');a=p.parse_args()
    if a.mode=='register':register()
    else:execute(a.task,a.manifest_sha256)
