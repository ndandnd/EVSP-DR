from pathlib import Path
import csv,json,subprocess,os,hashlib,time
B=Path('/home/nc437/ladder-lite/review_frolunda_20260916');PY='/home/nc437/evsp_env/bin/python';v=json.loads((B/'manifest.json').read_text());S=B/'smoke';S.mkdir();(B/'data/smoke').mkdir();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
with (B/'data/fdl_k01.csv').open() as f:rr=list(csv.DictReader(f));fields=list(rr[0]);r=dict(rr[0]);r['count_trip_id']='0'
p=B/'data/smoke/one_trip.csv'
with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(r)
env=dict(os.environ,GRB_LICENSE_FILE='/share/apps/software/gurobi/gurobi.lic',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');env.pop('LM_LICENSE_FILE',None)
out=S/'cg.json';args=[PY,str(B/'fdl_entry.py'),str(B/'cg_code'),str(B/'data'),'--csv','smoke/one_trip.csv','--prices_csv','hourly_prices_flat.csv','--time-model','event','--event-arc-mode','lazy','--fixed-sequence-index','--soc-step','2.5','--block-min','5','--max-iters','20','--columns_per_iter','30','--rc-eps','0.0001','--master-sense','cover','--master-backend','gurobi','--initial-pool','singletons','--wall-limit-s','120','--g-kwh','240','--charge-kw','240','--min-soc-frac','0','--out',str(out)]
with (S/'cg.log').open('w') as f:subprocess.run(args,cwd=B/'cg_code',env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240,check=True)
x=json.loads(out.read_text());assert x['final']['artificials']==0 and x['certified_rc_optimal'];assert x['provenance']['git_commit']==v['code']['cg']['commit'];assert x['provenance']['instance_sha256']==sha(p)
env.update(EVSP_EXPECTED_COMMIT=v['code']['mip']['commit'],EVSP_REQUIRE_DETACHED='1',EVSP_MIP_EXPECTED_RESULT_SHA256=sha(out),EVSP_MIP_EXPECTED_JOURNAL_SHA256=sha(x['columns_journal']))
mip=S/'mip.json';margs=[PY,str(B/'mip_code/src/run_exact_pool_mip.py'),'--result',str(out),'--data-dir',str(B/'data'),'--reference-data-dir',str(B/'data'),'--cover','--two-stage','--timelimit','20','--stage1-timelimit','10','--threads','1','--out',str(mip)]
with (S/'mip.log').open('w') as f:subprocess.run(margs,cwd=B/'mip_code',env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120,check=True)
y=json.loads(mip.read_text());assert y['physical_replay_validated'] and y['buses']==1
result={'status':'passed','native_cg_certificate':x['certified_rc_optimal'],'native_cg_final':x['final'],'mip_buses':y['buses'],'mip_fleet_proven':y['fleet_proven'],'mip_physical_replay':y['physical_replay_validated'],'smoke_input_sha256':sha(p),'code':v['code'],'files_sha256':{str(f):sha(f) for f in [out,mip,S/'cg.log',S/'mip.log']},'no_duty_selection_changed':True};(B/'smoke_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
