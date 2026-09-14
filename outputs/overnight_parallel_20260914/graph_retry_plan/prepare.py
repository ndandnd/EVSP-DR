"""Create a separate frozen recovery plan locally; never changes the source campaign."""
from pathlib import Path
import hashlib,importlib.util,json
B=Path(__file__).resolve().parent
OLD=B.parents[1]/'chain_extension_20260914'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def main():
 assert not (B/'manifest.json').exists(),'Do not overwrite frozen plan'
 original=json.loads((OLD/'manifest.json').read_text());jobs=json.loads((OLD/'case_jobs.json').read_text());q=json.loads((B.parent/'status_20260914T172907Z/status.json').read_text());queued={r['job_id']:r for r in q['rows']}
 sp=importlib.util.spec_from_file_location('oldcampaign',OLD/'campaign.py');mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
 root='/home/nc437/ladder-lite/chain_extension_20260914';code=root+'/code';cases={};edges=[]
 for cid,c in original['cases'].items():
  change=dict(c,cache='{attempt}/network.pkl');v=dict(original,graph_seconds=86400)
  args=[mod.PY,root+'/graph_entry.py',code,'{attempt}/progress.jsonl',*mod.cg_argv(v,change,None,True)]
  before=queued[jobs[cid]['cg']];assert before['state']=='PENDING' and before['reason']=='Dependency'
  edge={'case_id':cid,'cg_job_id':jobs[cid]['cg'],'original_graph_job':jobs[cid]['cache'],'original_dependency_at_audit':before['dependency'],'proposed_graph_parent':'afterok:{gate_job_id}','preserve_every_other_dependency':True,'apply_only_if_cg_still_pending':True,'graph_gate_dependency':'afterany:'+jobs[cid]['cache']};edges.append(edge)
  cases[cid]=dict(original_case_dir=root+'/cases/'+cid,original_graph_job=jobs[cid]['cache'],original_cg_job=jobs[cid]['cg'],input_sha256=c['input_sha256'],argv=args,
   cache_identity={'schema':'evsp-dr-event-network-cache-v1','git_commit':original['execution_commit'],'instance_sha256':c['input_sha256'],'reference_sha256':original['data_sha256']['Ref_dict.csv'],'deadhead_sha256':original['data_sha256']['par_ref_dhd.csv'],'prices_sha256':original['data_sha256']['hourly_prices_flat.csv'],'g_kwh':240.0,'charge_kw':240.0,'reserve_kwh':0.0,'soc_step':2.5,'block_min':5,'event_arc_mode':'lazy','strict_tariff_coverage':False},
   static_hashes={**{code+'/'+n:d for n,d in original['source_sha256'].items()},**{code+'/data/'+n:d for n,d in original['data_sha256'].items()},**{root+'/'+n:d for n,d in original['tooling_sha256'].items()},code+'/data/'+c['csv']:c['input_sha256']})
 m=dict(schema='evsp-conditional-graph-timeout-gate-v1',status='prepared_not_submitted',source_code=code,execution_commit=original['execution_commit'],original_manifest_path=root+'/manifest.json',original_manifest_sha256=sha(OLD/'manifest.json'),graph_seconds=86400,resources={'cpus':2,'mem':'64G','allocation_s':88200,'partition':'default_partition','exclude':'scaglione-compute-01','independent_gate_count':18,'concurrency':18},cases=cases,
  publication_scope='Only absent expected cache slots may receive hash-verified recovery links and explicit recovery cache_result; original graph attempts never modified.',tooling_sha256={n:sha(B/n) for n in ['gate.py','process_worker.py','worker.sub','prepare.py']})
 save(B/'manifest.json',m);save(B/'dependency_amendment.json',dict(status='prepared_not_applied',original_manifest_sha256=m['original_manifest_sha256'],recovery_manifest_sha256=sha(B/'manifest.json'),edges=edges,record_required='Before any update, freeze actual gatejobIDs and actual current dependency strings in an append-only ledger; replace only each graph edge. No original manifest changes.'))
if __name__=='__main__':main()
