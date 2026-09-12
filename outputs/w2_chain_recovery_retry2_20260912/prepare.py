from pathlib import Path
import json,subprocess,hashlib
old=Path('/home/nc437/ladder-lite/w2_chain_recovery_20260912');new=Path('/home/nc437/ladder-lite/w2_chain_recovery_retry2_20260912');assert not new.exists();new.mkdir();(new/'logs').mkdir()
for fn in ['cg15_manifest.json','mip14_manifest.json','mip15_manifest.json']:
 m=json.loads((old/fn).read_text());oldout=m['output'];newout=oldout.replace(str(old),str(new));m['output']=newout
 m['argv']=[x.replace(oldout,newout) for x in m['argv']]
 if fn=='mip15_manifest.json':
  prev=m['source_status'];m['source_status']=prev.replace(str(old),str(new));m['argv']=[m['source_status'] if x==prev else x for x in m['argv']]
 (new/fn).write_text(json.dumps(m,indent=2)+'\n')
worker=(old/'worker.sub').read_text().replace('source /home/', 'export PYTHON_BIN=/home/nc437/evsp_env/bin/python\nsource /home/');(new/'worker.sub').write_text(worker)
subprocess.run(['bash','-n',str(new/'worker.sub')],check=True)
# Execute precisely the wrapper environment + preflight, stopping before the workload.
preflight=worker[:worker.index('exec /home/nc437/evsp_env/bin/python')]
r=subprocess.run(['bash'],input=preflight,capture_output=True,text=True);assert r.returncode==0,r.stderr
(new/'preflight_verified.json').write_text(json.dumps({'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2)+'\n')
manifest=json.loads((old/'manifest.json').read_text());manifest['supersedes_launcher_jobs']=['37532','37533','37534'];manifest['repair']='Set required PYTHON_BIN before preflight; source solver and scientific settings unchanged; new output paths.';manifest['files']=[{'path':str(new/fn),'sha256':hashlib.sha256((new/fn).read_bytes()).hexdigest()} for fn in ['cg15_manifest.json','mip14_manifest.json','mip15_manifest.json','worker.sub']];(new/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({fn:(new/fn).read_text() for fn in ['cg15_manifest.json','mip14_manifest.json','mip15_manifest.json','worker.sub','manifest.json','preflight_verified.json']}))