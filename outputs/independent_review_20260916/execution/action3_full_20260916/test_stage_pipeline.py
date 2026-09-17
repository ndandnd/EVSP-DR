import pathlib,sys,tempfile,shutil,json,subprocess,hashlib,time
HERE=pathlib.Path(__file__).resolve().parent
REPO=next(p for p in HERE.parents if (p/'.codex-work/action3-full-20260916').exists());CODE=REPO/'.codex-work/action3-full-20260916'
sys.path[:0]=[str(CODE/'src'),str(HERE/'prepared')]
from sequence_replay import replay,ARMS,sha
from audit_giro_known_columns import build_problem,STATIONS
from event_pricer_network import _event_times,normalize_event_station_prices
from utils_v2 import load_station_hourly_prices,base_station_name
started=time.monotonic()
with tempfile.TemporaryDirectory() as td:
 root=pathlib.Path(td);shutil.copytree(HERE/'prepared',root/'prepared',ignore=shutil.ignore_patterns('__pycache__'))
 for n in ['stage_worker.py','atomic_pool_copy.py','replay_journal.py','submit_continuation.py','worker.sh']:shutil.copyfile(HERE/n,root/n)
 inp=root/'prepared/inputs/w5_k31.csv';shutil.copyfile(REPO/'outputs/independent_review_20260916/execution/p2_strict/smoke/trips2.csv',inp)
 pm=json.loads((root/'prepared/manifest.json').read_text());pm.update(instance_sha256=sha(inp),trip_count=2);(root/'prepared/manifest.json').write_text(json.dumps(pm))
 m=json.loads((HERE/'manifest.json').read_text());m.update(code=str(CODE),prepared_manifest_sha256=sha(root/'prepared/manifest.json'));(root/'manifest.json').write_text(json.dumps(m))
 cm=json.loads((HERE/'continuation_manifest.json').read_text());cm['campaign_manifest_sha256']=sha(root/'manifest.json');cm['cases']={'baseline':dict(cm['cases']['baseline'],input_sha256=sha(inp),trips=2,cg_wall_s=30,mip_wall_s=10)};(root/'continuation_manifest.json').write_text(json.dumps(cm))
 problem=build_problem(inp.parent,inp.name,reference_data_dir=CODE/'data',max_station_to_trip_wait_min=1560);prices=load_station_hourly_prices(CODE/'data/hourly_prices_flat.csv',sorted({base_station_name(s) for s in STATIONS}));events=_event_times(problem,normalize_event_station_prices(prices,horizon_min=1560,strict_tariff_coverage=False),5)
 seed=root/'seed';seed.mkdir();routes=[]
 for t in range(2):
  outcome,r=replay(problem,prices,events,(t,),ARMS['baseline'],{0:'18E1',1:'18E1'});assert outcome['status']=='feasible';routes.append(r)
 (seed/'seed_pool.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in routes));(seed/'assembly.json').write_text(json.dumps({'arm':'baseline','manifest_sha256':sha(root/'prepared/manifest.json'),'cg_seed_ready':True,'seed_pool_sha256':sha(seed/'seed_pool.jsonl'),'unknown_sequences':0}))
 (root/'assembly_status').mkdir();(root/'assembly_status/baseline.json').write_text(json.dumps({'assembly':str(seed),'assembly_sha256':sha(seed/'assembly.json')}))
 records=[]
 for stage in ['cache','cg','mip']:
  p=subprocess.run([sys.executable,str(root/'stage_worker.py'),'--root',str(root),'--stage',stage,'--case','baseline'],capture_output=True,text=True,timeout=60)
  if p.returncode:
   logs='\n'.join(q.read_text()[-5000:] for q in root.rglob('solver.log'));raise AssertionError(p.stdout+p.stderr+logs)
  records.append({'stage':stage,'returncode':p.returncode})
 cg=json.loads((root/'continuation/baseline/cg.json').read_text());mip=json.loads((root/'continuation/baseline/mip.json').read_text());assert cg['certified_rc_optimal'];assert cg['final']['route_weight']==1.;assert mip['result']['stage1']['incumbent_fleet']==1
 # Completed-cache retry validates existing files and exits cleanly.
 p=subprocess.run([sys.executable,str(root/'stage_worker.py'),'--root',str(root),'--stage','cache','--case','baseline'],capture_output=True,text=True,timeout=30);assert p.returncode==0,p.stderr
 report={'status':'PASS','stages':records,'cg_certified':True,'route_weight':1,'mip_fleet':1,'cache_completed_retry_valid':True,'elapsed_s':time.monotonic()-started,'worker_sha256':sha(HERE/'stage_worker.py')}
 (HERE/'stage_pipeline_test.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
