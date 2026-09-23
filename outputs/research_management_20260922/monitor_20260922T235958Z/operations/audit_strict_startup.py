from pathlib import Path
import importlib.util,sys,json,hashlib,re
sys.dont_write_bytecode=True
P=Path(__file__).resolve().parent;S=P/'strict_cg';A=S/'attempts/824877_r0';OLD=P.parents[1]/'monitor_20260922T195842Z/strict_review'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
m=read(S/'manifest.json');e=read(A/'execution.json');started=read(A/'CG_STARTED.json');checks=[]
def ck(n,v):checks.append({'check':n,'passed':bool(v)});assert v,n
ck('manifest exact approved bytes',sha(S/'manifest.json')==sha(OLD/'manifest.json')=='206fafab7e514ef9e8f1bee94db5a3cde5767bc9edf7a01a027419fcb96b3c37')
ck('worker execution pin matches approved source',e['worker_sha256']==sha(OLD/'cached_cg_worker.py')==m['worker_sha256'])
ck('execution plan hash',e['plan_sha256']==sha(S/'manifest.json'))
ck('job and unique attempt',e['job_id']=='824877' and e['restart']=='0')
ck('model pin',e['model_commit']=='fedf421461f94727e6b1292a0e7789ab76ed8587')
ck('root exclusive start agrees',read(S/'STARTED.json')==e)
ck('CG started receipt',started['cg_started'] and all(started[k]==e[k] for k in e if k!='cg_started'))
ck('license log digest',started['license_log_sha256']==sha(A/'license.log'))
license_record=json.loads((A/'license.log').read_text().splitlines()[-1]);ck('native Gurobi license probe optimal',license_record['status']=='OPTIMAL' and license_record['version']=='12.0.3')
sp=importlib.util.spec_from_file_location('reviewed_cached_worker',OLD/'cached_cg_worker.py');module=importlib.util.module_from_spec(sp);sp.loader.exec_module(module)
argv=read(A/'command.json');ck('exact approved native command',argv==module.command(m,Path(m['remote_root'])/'attempts/824877_r0'))
ck('no old resume', '--resume' not in argv)
ck('no graph rebuild or MIP',not started['graph_built'] and not started['mip_started'])
account=read(S/'accounting.json');row=account['rows'][0];ck('scheduler running native8CPU16G',row[2]=='RUNNING' and row[5]=='16G' and row[10]=='8')
ck('no terminal or failure receipt',not any((A/n).exists() for n in ['COMPLETE.json','FAILED.json','result.json','worker_result.json']))
logs=read(S/'log_tails.json');log=next(x['tail'] for x in logs if x['path'].endswith('/cg.log'))
models=re.findall(r'Optimize a model with (\d+) rows, (\d+) columns and (\d+) nonzeros',log);objectives=re.findall(r'Optimal objective\s+([0-9.eE+-]+)',log)
ck('native restricted master progressing',bool(models) and models[-1][0]=='331' and bool(objectives))
result={'status':'passed','checks_passed':len(checks),'checks_total':len(checks),'checks':checks,'observed_utc':account['observed_utc'],'scheduler_state':'RUNNING','elapsed':row[3],'node':row[7],'restart':0,'manifest_sha256':sha(S/'manifest.json'),'native_license':license_record,'guarded_native_preflight_complete':True,'preflight_scope':'CG_STARTED is written only after native source HEAD/clean/all253files, inputs, lineage, full cache hash and license checks pass in the hash-matched worker. This local audit does not rehash remote model/cache bytes.','last_logged_restricted_master':{'rows':int(models[-1][0]),'columns':int(models[-1][1]),'nonzeros':int(models[-1][2]),'weighted_objective_printed':objectives[-1]},'cg_endpoint':None,'pricing_certificate':None,'route_weight':None,'fleet_result':None,'shared_capacity_validation':None,'peak_memory':None,'scope':'Interim startup/log evidence only; no pool download or scientific endpoint; no extra SSH'}
(S/'startup_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'checks':len(checks),'status':'passed','last_RMP':result['last_logged_restricted_master']}))
