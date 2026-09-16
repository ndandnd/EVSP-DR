"""Actual SIGTERM/SIGUSR1 during a tiny pricing call, using test-only instrumentation."""
from pathlib import Path
import json,sys,subprocess,os,hashlib,time
q=json.loads(sys.stdin.read());P=Path(q['temp_root']);CODE=Path('/home/nc437/ladder-lite/review_full40_20260916/code');PY='/home/nc437/evsp_env/bin/python'
script='''from pathlib import Path
import sys,os,signal,json,time
code,marker,sig=sys.argv[1:4];sys.path.insert(0,str(Path(code)/"src"))
import exact_pricer_expanded as e
import master_lp_gurobi as g
import event_pricer_network as n
sent=[False];original=n.EventExpandedNetwork.sink_predecessor_route_batch;solve=g.GurobiRestrictedMaster.solve

def wrapped(self,*a,**kw):
 if not sent[0]:
  sent[0]=True
  with open(marker,"a") as f:f.write("signal_sent "+str(time.perf_counter())+"\\n")
  os.kill(os.getpid(),getattr(signal,sig))
 return original(self,*a,**kw)
def master(self,*a,**kw):
 if sent[0]:
  with open(marker,"a") as f:f.write("master_solve_after_signal\\n")
 t=time.perf_counter()
 result=solve(self,*a,**kw)
 if sent[0]:
  with open(marker,"a") as f:f.write("post_signal_master_seconds "+str(time.perf_counter()-t)+"\\n")
 return result
n.EventExpandedNetwork.sink_predecessor_route_batch=wrapped
g.GurobiRestrictedMaster.solve=master
raise SystemExit(e.main(sys.argv[4:]))
'''
wrapper=P/'signal_hook_timed.py';wrapper.write_text(script);results=[]
for sig in ['SIGTERM','SIGUSR1']:
 d=P/(sig+'_timed_v3');d.mkdir();out=d/'cg.json';marker=d/'marker.txt'
 a=[PY,str(wrapper),str(CODE),str(marker),sig,'--csv',str(P/'fixture.csv'),'--prices_csv','hourly_prices_flat.csv','--time-model','event','--event-arc-mode','lazy','--fixed-sequence-index','--soc-step','2.5','--block-min','5','--columns_per_iter','30','--column-selection','reduced_cost','--column-diversity-weight','0.0','--column-candidate-multiplier','4','--rc-eps','0.0001','--master-sense','cover','--master-backend','gurobi','--initial-pool','singletons','--g-kwh','240','--charge-kw','240','--min-soc-frac','0','--checkpoint-every','25','--out',str(out),'--max-iters','20','--wall-limit-s','300']
 env=dict(os.environ,GRB_LICENSE_FILE='/share/apps/software/gurobi/gurobi.lic',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');env.pop('LM_LICENSE_FILE',None)
 with (d/'native.log').open('w') as f:r=subprocess.run(a,cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=90)
 assert r.returncode==0,(d/'native.log').read_text();x=json.loads(out.read_text());events=marker.read_text().splitlines();assert x['stop_reason']=='external_signal' and x['termination_signal']==sig;assert 'master_solve_after_signal' in events
 results.append({'signal':sig,'returncode':r.returncode,'events':events,'post_signal_to_process_exit_s':time.perf_counter()-float(next(x.split()[1] for x in events if x.startswith('signal_sent '))),'post_signal_master_solve_s':float(next(x.split()[1] for x in events if x.startswith('post_signal_master_seconds '))),'stop_reason':x['stop_reason'],'termination_signal':x['termination_signal'],'columns':x['columns'],'iterations':x['iterations'],'final_lp_source':x['final_lp_source'],'native_status_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'native_log_sha256':hashlib.sha256((d/'native.log').read_bytes()).hexdigest()})
print(json.dumps({'tests':results,'instrumentation':'A test-only wrapper sends real OS signal on first pricing entry and records any subsequent master solve. Pinned solver source is unchanged.','production_modified':False,'slurm_submissions':0,'temp_root':str(P)},indent=2))
