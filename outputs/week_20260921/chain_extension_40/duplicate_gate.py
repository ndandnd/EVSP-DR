"""Read-only cross-campaign and live-queue duplicate gate; refreshed before launch."""
import datetime, subprocess
import campaign as c
v=c.read(c.B/'manifest.json'); target=set(v['cases']); collisions=[]; scanned=[]
for name in ['case_jobs.json','jobs.json']:
 for path in sorted(c.B.parent.glob('*/'+name)):
  if path.parent==c.B:
   collisions.append({'path':str(path),'reason':'own production record exists'});continue
  data=c.read(path); ids=set(data) if isinstance(data,dict) else {x.get('case_id') for x in data if isinstance(x,dict)}
  overlap=ids&target; scanned.append({'path':str(path),'sha256':c.sha(path),'overlap':sorted(overlap)})
  if overlap:
   mp=path.parent/'manifest.json'; other=c.read(mp) if mp.exists() else {}
   for cid in overlap:
    digest=other.get('cases',{}).get(cid,{}).get('input_sha256')
    if digest is None or digest==v['cases'][cid]['input_sha256']: collisions.append({'path':str(path),'case_id':cid})
raw=subprocess.check_output([c.SLURM+'squeue','-u','nc437','-h','-o','%i|%j|%T|%Z|%o'],text=True,timeout=45)
for line in raw.splitlines():
 fields=line.split('|',4)
 if len(fields)!=5: raise ValueError(line)
 job,name,state,workdir,command=fields
 if any(name=='drX_'+cid+'_'+mode for cid in target for mode in ['cg','mip','cache']): collisions.append({'queue_row':line})
 if str(c.B/'worker.sub') in command: collisions.append({'queue_row':line})
 if name=='drX_graphs' and 'chain_extension_' in command:
  root=command.split('/worker.sub')[0].strip().split()[-1]; mp=c.Path(root)/'manifest.json'
  if mp.exists():
   other=c.read(mp)
   if any(cid in other.get('cases',{}) and other['cases'][cid].get('input_sha256')==v['cases'][cid]['input_sha256'] for cid in target): collisions.append({'queue_row':line})
record={'status':'failed' if collisions else 'passed','checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_sha256':c.sha(c.B/'manifest.json'),'scanned':scanned,'queue_raw':raw,'collisions':collisions}
c.save(c.B/'duplicate_launch_audit.json',record)
assert not collisions,collisions
print('PASS: cross-campaign job maps and current queue contain no equivalent production')
