"""Read-only operational metadata adapter; never opens graph pickle files."""
from pathlib import Path
import hashlib,json

def collect_graph_timeout_gates(root):
 root=Path(root)
 def read(p):return json.loads(Path(p).read_text())
 def artifact(p):
  raw=Path(p).read_bytes()
  return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),value=json.loads(raw))
 manifest=artifact(root/'manifest.json');digest=manifest['sha256'];m=manifest['value']
 out=dict(kind='operational_graph_timeout_recovery',optimization_run=False,pricing_certificate=None,manifest=manifest,workflow={},cases=[])
 for n in ['jobs.json','case_jobs.json','dependency_amendment.json','deployment_validation.json','before_dependencies.json','after_dependencies.json','validation.json','v1_amendment.json','storage_binding.json','native_accounting_v2.json']:
  if (root/n).exists():out['workflow'][n]=artifact(root/n)
 for cid,c in m['cases'].items():
  row=dict(case_id=cid,original_graph_job=c['original_graph_job'],original_cg_job=c['original_cg_job'],attempts=[])
  completion=root/'cases'/cid/'completion.json'
  if completion.exists():
   value=artifact(completion);d=value['value'];observed=d.get('manifest_sha256') or d.get('recovery_provenance',{}).get('recovery_manifest_sha256')
   row.update(completion=value,manifest_matches=observed==digest)
  for attempt in sorted((root/'cases'/cid/'attempts').glob('*')):
   records={n:artifact(attempt/n) for n in ['execution.json','recovery_provenance.json'] if (attempt/n).is_file()}
   if records:row['attempts'].append(dict(attempt=attempt.name,records=records))
  marker=Path(c['original_case_dir'])/'cache_result.json'
  if marker.exists():row['canonical_cache_marker']=artifact(marker)
  out['cases'].append(row)
 return out
