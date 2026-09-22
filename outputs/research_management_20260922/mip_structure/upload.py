from pathlib import Path
import subprocess,json,base64,hashlib
P=Path(__file__).resolve().parent;files=[P/'manifest.json',*sorted((P/'code').glob('*'))];payload={str(p.relative_to(P)):base64.b64encode(p.read_bytes()).decode() for p in files if p.is_file()};script='files='+repr(payload)+'''\nfrom pathlib import Path
import base64,hashlib,json,os
root=Path('/home/nc437/ladder-lite/mip_structure_20260922');root.mkdir(exist_ok=True)
for name,encoded in files.items():
 p=root/name;p.parent.mkdir(parents=True,exist_ok=True);b=base64.b64decode(encoded)
 if p.exists():assert p.read_bytes()==b,'Existing file differs: '+str(p)
 else:p.write_bytes(b)
 if name.endswith('.sh'):p.chmod(0o755)
for name in ['slurm','prepared']: (root/name).mkdir(exist_ok=True)
m=json.loads((root/'manifest.json').read_text());missing=[p for c in m['cases'] for p in c['pins'] if not Path(p).is_file()]
assert not missing,missing
print(json.dumps({'uploaded_files':len(files),'all_source_paths_exist':True,'sha256':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files}}))
''';r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','unicorn','python3 -'],input=script,text=True,capture_output=True,check=True);(P/'upload_receipt.json').write_text(r.stdout);print(r.stdout)
