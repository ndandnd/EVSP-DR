"""Native prelaunch smoke; validation files are not research observations."""
from pathlib import Path
import csv,json,os,shutil,sys
from unittest import mock
import campaign as c
root=c.B;v=c.read(root/'manifest.json');w=root/'validation_v2';w.mkdir(exist_ok=True);checks=[]
c.code_check()
for p,h in v['tooling_sha256'].items():assert c.sha(root/p)==h
sys.path.insert(0,str(c.CODE/'src'));import exact_pricer_expanded as exact
for cid,case in v['cases'].items():
 for budget in [case['fresh_primary_budget_s'],case['fresh_graph_sensitivity_budget_s']]:
  with mock.patch.object(exact,'run_cg',return_value={}) as run:
   assert exact.main(c.cg_args(case,w/'parser.json',budget)[2:])==0;run.assert_called_once()
   a=run.call_args.args[0];assert a.initial_pool=='singletons' and not a.inherit_event_pool_from and not a.validated_seed_routes
checks.append('48 budget command lines parse as fresh singleton starts without inherited/seed columns')
import gurobipy as gp
m=gp.Model('validation_2001');m.Params.OutputFlag=0;m.Params.Threads=1;x=m.addVars(2001,lb=0,ub=1,obj=1);m.addConstr(x.sum()>=1);m.optimize();assert m.Status==gp.GRB.OPTIMAL;m.dispose()
checks.append('native unrestricted Gurobi license')
# Authenticate an actual legacy graph using the consumer compatibility attestation.
case=v['cases']['c6_k05'];args=c.cg_args(case,w/'cache_probe.json',300)
for key in ['--out','--phase-telemetry','--gurobi-log']:
 i=args.index(key);del args[i:i+2]
args+=['--event-network-cache-only'];c.run_process(args,w/'real_cache_check',300)
checks.append('actual old-source k5 pickle hash, consumer identity and metrics pass the native loader')
with (c.CODE/'data'/case['csv']).open() as f:r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)[:4]
rel='scale_ladder/instances/cumulative_budget_validation_20260913.csv'
with (c.CODE/'data'/rel).open('w') as f:t=csv.DictWriter(f,fieldnames=fields);t.writeheader();t.writerows(rows)
tiny=dict(case,csv=rel,input_sha256=c.sha(c.CODE/'data'/rel),cache=str(w/'network.pkl'),fresh_primary_budget_s=120,fresh_graph_sensitivity_budget_s=240)
args=c.cg_args(tiny,w/'build.json',300)
for key in ['--out','--phase-telemetry','--gurobi-log']:
 i=args.index(key);del args[i:i+2]
args[args.index('--event-network-cache-mode')+1]='build-or-load';args+=['--event-network-cache-only'];c.run_process(args,w/'tiny_cache_build',300)
tiny['consumer_cache_manifest_sha256']=c.sha(tiny['cache']+'.manifest.json')
smoke=dict(v,cases={'smoke':tiny})
for name in ['campaign.py','worker.sub']:shutil.copy2(root/name,w/name)
c.save(w/'manifest.json',smoke);c.B=w;c.register_mip=lambda *args:None
for mode in ['base','mip_base','extra','mip_extra']:c.worker(mode,'smoke')
p=c.read(w/'cases/smoke/base/completion.json');assert p['certified'] and p['usable']
assert c.read(w/'cases/smoke/mip_base/completion.json')['status']=='finished'
assert c.read(w/'cases/smoke/extra/completion.json')['status']=='shared_primary_certificate'
assert c.read(w/'cases/smoke/mip_extra/completion.json')['status']=='shared_primary_mip'
checks.append('fresh CG, physical two-stage MIP, and both certificate-sharing branches pass')
tiny['status_path']=p['result_path'];tiny['status_sha256']=p['result_sha256']
c.save(w/'manifest.json',dict(smoke,cases={'smoke':tiny}));c.worker('mip_warm','smoke')
assert c.read(w/'cases/smoke/mip_warm/completion.json')['status']=='finished'
checks.append('warm-reference MIP uses authenticated external status and the same one-hour setup')
resume=w/'resume.json';copied=c.copy_checkpoint(p['result_path'],resume)
c.run_process(c.cg_args(tiny,resume,240)+['--resume'],w/'native_resume',300)
r=c.read(resume);assert c.usable(r) and r['wall_s']>=copied['native_elapsed_s'];assert not r.get('inherited_event_pool_status_sha256')
checks.append('copied same-instance checkpoint resumes natively with a larger cumulative wall budget')
os.environ['SLURM_RESTART_COUNT']='1';c.worker('mip_base','smoke')
retry=next((w/'cases/smoke/mip_base').glob('*_r1/state.json'));assert c.read(retry)['status']=='already_complete' and not (retry.parent/'process').exists()
checks.append('requeue after MIP completion does not repeat optimization')
c.B=root;c.save(root/'validation.json',dict(status='passed',checked_utc=c.now(),job_id=os.environ['SLURM_JOB_ID'],checks=checks,tooling_sha256=v['tooling_sha256'],validator_sha256=c.sha(__file__)))
print(json.dumps({'status':'passed','checks':checks}),flush=True)
