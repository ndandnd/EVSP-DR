"""Small native validation; never contributes research data points."""
import csv, json, os, subprocess, sys
from pathlib import Path
from unittest import mock
import campaign as c
v=c.read(c.B/'manifest.json'); out=c.B/'validation'/('job_'+os.environ.get('SLURM_JOB_ID','local')); out.mkdir(parents=True,exist_ok=False)
checks=[]
c.initial_parent_gate(v, require_job=True)
for kind,root in [('cg',c.B/'code'),('mip',c.MIP)]:
    assert c.source_provenance(root,v['source_provenance'][kind]['commit']) == v['source_provenance'][kind]
assert c.read(c.B/'input_validation.json')['status'] == 'passed'
checks.append('immutable CG/MIP sources, borrowed object stores and independent frozen previous-input gate checked')
for chain,p in v['initial_parents'].items():
    assert c.sha(c.PARENT/'manifest.json') == p['parent_manifest_sha256']
    assert c.sha(c.B/'code/data'/p['csv']) == p['csv_sha256']
    if Path(p['status']).exists(): c.authenticate_parent(p['status'],p['csv_sha256'])
checks.append('six frozen k28 inputs and published parent gates checked')
for case in v['cases'].values():
    assert c.sha(c.B/'code/data'/case['csv'])==case['input_sha256']
assert len(v['cases'])==12 and len(v['warm_chains'])==6
checks.append('12 frozen case inputs match manifest')
sys.path.insert(0,str(c.B/'code/src'))
import exact_pricer_expanded as exact
for case in v['cases'].values():
    for cache_only in [False,True]:
        with mock.patch.object(exact,'run_cg',return_value={}) as run:
            assert exact.main(c.cg_argv(v,case,out/'parser.json',cache_only))==0
            run.assert_called_once()
checks.append('24 production CG/cache command lines pass actual parser')
import gurobipy as gp
m=gp.Model('extension_license_2001'); m.Params.OutputFlag=0; m.Params.Threads=1
x=m.addVars(2001,lb=0,ub=1,obj=1); m.addConstr(gp.quicksum(x.values())>=1);m.optimize()
assert m.Status==gp.GRB.OPTIMAL; m.dispose()
checks.append('2001-variable native Gurobi optimization passes size-unrestricted license')
source=c.B/'code/data'/v['cases']['w1_k29']['csv']
with source.open() as f: r=csv.DictReader(f); fields=r.fieldnames; rows=list(r)[:4]
rel='scale_ladder/instances/chain_extension_20260915/validation_four_trips.csv'
with (c.B/'code/data'/rel).open('w') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
tiny={**v['cases']['w1_k29'],'csv':rel,'cg_seconds':90,'cache':str(out/'plain.pkl')}
def run(name,args,timeout=180):
    with (out/(name+'.log')).open('w') as log:
        p=subprocess.run(args,cwd=c.B/'code',stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    assert p.returncode==0,(name,p.returncode)
run('cache_plain',[c.PY,str(c.B/'code/src/exact_pricer_expanded.py'),*c.cg_argv(v,tiny,None,True)])
tiny['cache']=str(out/'instrumented.pkl')
run('cache_instrumented',[c.PY,str(c.B/'graph_entry.py'),str(c.B/'code'),str(out/'progress.jsonl'),*c.cg_argv(v,tiny,None,True)])
assert c.sha(out/'plain.pkl')==c.sha(out/'instrumented.pkl'), 'Progress wrapper changed graph pickle'
checks.append('instrumented/uninstrumented four-trip graphs have identical pickle SHA256')
def without_inheritance(args):
    for key in ['--inherit-event-pool-from','--inherit-event-pool-workers','--inherit-max-columns','--inherit-time-limit-s']:
        i=args.index(key); del args[i:i+2]
    return args
parent=out/'parent.json'
run('cg_parent',[c.PY,str(c.B/'code/src/exact_pricer_expanded.py'),*without_inheritance(c.cg_argv(v,tiny,parent))])
assert c.usable(c.read(parent))
tiny['parent_status']=str(parent); child=out/'child.json'
run('cg_child',[c.PY,str(c.B/'code/src/exact_pricer_expanded.py'),*c.cg_argv(v,tiny,child)])
value=c.read(child); assert c.usable(value)
checks.append('native cache-required CG and full-pool inheritance both return usable LPs')
os.environ.update(EVSP_EXPECTED_COMMIT=c.MIP_COMMIT,EVSP_REQUIRE_DETACHED='1',EVSP_MIP_EXPECTED_RESULT_SHA256=c.sha(child),EVSP_MIP_EXPECTED_JOURNAL_SHA256=c.sha(value['columns_journal']))
run('mip',[c.PY,str(c.MIP/'src/run_exact_pool_mip.py'),'--result',str(child),'--data-dir',str(c.B/'code/data'),'--reference-data-dir',str(c.B/'code/data'),'--cover','--two-stage','--timelimit','30','--stage1-timelimit','15','--threads','2','--mipgap','0.0001','--gurobi-log',str(out/'mip.gurobi.log'),'--out',str(out/'mip.json')])
assert c.read(out/'mip.json').get('physical_replay_validated') is True
checks.append('two-stage MIP with source hashes and physical replay passes')
c.save(c.B/'validation.json',{'status':'passed','checks':checks,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'execution_commit':v['execution_commit'],'tooling_sha256':v['tooling_sha256'],'validator_sha256':c.sha(__file__),'manifest_sha256':c.sha(c.B/'manifest.json'),'input_validation_sha256':c.sha(c.B/'input_validation.json'),'artifact_directory':str(out),'artifacts':{p.name:c.sha(p) for p in out.iterdir() if p.is_file() and p.suffix in ['.json','.log','.jsonl','.pkl']}})
print(json.dumps({'status':'passed','checks':checks}),flush=True)
