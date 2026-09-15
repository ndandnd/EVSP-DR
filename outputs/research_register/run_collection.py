from pathlib import Path
import subprocess,datetime,time,json
r=Path('/Users/nadan/Documents/projects/demandresponse');mon=r/'outputs/post_meeting_20260910/monitor';previous_record=r/'outputs/research_register/last_successful_collection_path.txt'; previous_record=previous_record if previous_record.exists() else r/'outputs/research_register/latest_snapshot_path.txt'; previous=Path(previous_record.read_text().strip());old=json.loads(previous.read_bytes());assert datetime.datetime.fromisoformat(old['timestamp_utc']) < datetime.datetime.now(datetime.timezone.utc)
with (r/'outputs/research_register/preemption_study/register_compact_unions.py').open('rb') as registry_source:
 registration=subprocess.run(['ssh','-S','/Users/nadan/.ssh/evsp-unicorn.sock','-o','BatchMode=yes','-o','ConnectTimeout=8','nc437@unicorn-login-01.coecis.cornell.edu','python3 -'],stdin=registry_source,capture_output=True)
 print(json.dumps({'compact_union_restart_registration_returncode':registration.returncode,'detail':registration.stdout.decode()[-1000:],'error':registration.stderr.decode()[-1000:]}),flush=True)
registration=subprocess.run(['ssh','-S','/Users/nadan/.ssh/evsp-unicorn.sock','-o','BatchMode=yes','-o','ConnectTimeout=8','nc437@unicorn-login-01.coecis.cornell.edu','/home/nc437/evsp_env/bin/python /home/nc437/ladder-lite/union_target_feasibility_20260915/register_attempts.py'],capture_output=True,text=True)
print(json.dumps({'target_feasibility_registration_returncode':registration.returncode,'detail':registration.stdout[-1000:],'error':registration.stderr[-1000:]}),flush=True)
now=datetime.datetime.now(datetime.timezone.utc);stamp=now.strftime('%Y%m%dT%H%M%SZ');out=mon/(stamp+'.json');err=mon/(stamp+'_stderr.txt');meta=dict(snapshot=str(out),previous=str(previous),started_utc=now.isoformat());Path('/tmp/evsp-monitor-current.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True);start=time.time()
with (r/'outputs/meeting_20260910/collect_remote.py').open('rb') as src,out.open('wb') as dst,err.open('wb') as stderr:
 p=subprocess.run(['ssh','-S','/Users/nadan/.ssh/evsp-unicorn.sock','-o','BatchMode=yes','-o','ConnectTimeout=8','nc437@unicorn-login-01.coecis.cornell.edu','python3 -'],stdin=src,stdout=dst,stderr=stderr)
meta.update(returncode=p.returncode,elapsed_s=time.time()-start,stderr_path=str(err),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());Path('/tmp/evsp-monitor-current.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)
if p.returncode:print(err.read_text()[-4000:])
if p.returncode == 0:
 value=json.loads(out.read_bytes());assert value.get('campaigns') and value.get('timestamp_utc')
 assert datetime.datetime.fromisoformat(value['timestamp_utc']) > datetime.datetime.fromisoformat(old['timestamp_utc'])
 pointer=r/'outputs/research_register/last_successful_collection_path.txt'; temp=pointer.with_suffix('.tmp');temp.write_text(str(out)+'\n');temp.replace(pointer)
raise SystemExit(p.returncode)
