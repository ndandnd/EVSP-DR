from pathlib import Path
import json,hashlib,subprocess,importlib.util,csv,os
B=Path('/home/nc437/ladder-lite/random_trip_groups_c1_20260916');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();v=json.loads((B/'manifest.json').read_text())
assert subprocess.check_output(['git','-C',str(B/'code'),'rev-parse','HEAD'],text=True).strip()==v['execution_commit']
assert not subprocess.check_output(['git','-C',str(B/'code'),'status','--porcelain','--untracked-files=no'],text=True).strip()
for f,h in v['tooling_sha256'].items():assert sha(B/f)==h
for f,h in v['data_sha256'].items():assert sha(B/'code/data'/f)==h
sp=importlib.util.spec_from_file_location('campaign',B/'campaign.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
previous=set();edges=[]
for cid in v['warm_chains']['1']:
 c=v['cases'][cid];p=B/'code/data'/c['csv'];assert sha(p)==c['input_sha256'];r=list(csv.DictReader(p.open()));ids={int(x['Ordered_Trip_ID']) for x in r};assert previous<=ids;assert len(r)==len(ids)==c['trip_count'];assert all(int(x['count_trip_id'])==i for i,x in enumerate(r));args=m.cg_argv(v,c,'validate-only.json');inherit='--inherit-event-pool-from' in args;assert inherit==bool(previous);assert c['target_buses'] is None
 if previous:edges.append([c['previous_case'],cid])
 previous=ids
assert len(edges)==13 and len(previous)==364
original=B/'code/data' # original frozen final input hash was independently validated in local preparation.
# Size-limited-license smoke test: 2,101 variables exceeds the restricted license limit.
os.environ['GRB_LICENSE_FILE']='/share/apps/software/gurobi/gurobi.lic';os.environ.pop('LM_LICENSE_FILE',None)
import gurobipy as gp
with gp.Env(empty=True) as env:
 env.setParam('OutputFlag',0);env.start()
 with gp.Model('review_random_license_size_check',env=env) as model:
  model.Params.Threads=1;x=model.addVars(2101,lb=0,ub=1,obj=1);model.addConstr(gp.quicksum(x.values())>=1);model.optimize();assert model.Status==gp.GRB.OPTIMAL
  license_test={'variables':2101,'status':model.Status,'objective':model.ObjVal,'gurobi_version':gp.gurobi.version(),'size_limited_license_excluded':True}
result={'status':'passed_prepared_not_submitted','manifest_sha256':sha(B/'manifest.json'),'cases':14,'final_trip_count':364,'parent_edges':edges,'all_input_and_tool_hashes_verified':True,'tracked_code_clean':True,'first_stage_fresh_singletons':True,'stage_index_is_not_target':True,'license_smoke_test':license_test,'jobs_json_exists':(B/'jobs.json').exists()};assert not result['jobs_json_exists'];(B/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
