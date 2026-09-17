"""Generate a read-only publication inventory; never copy or commit files."""
from pathlib import Path
import json,hashlib,re,gzip,datetime
B=Path(__file__).resolve().parent;E=B/'execution';REPO=B.parents[1]
EXACT={
 'action3_full_20260916/deploy.tar.gz':'redundant deployment archive; expanded hashed source files retained',
 'advisor_sequence_20260916/single_factor_pilot/deploy.tar.gz':'redundant deployment archive; expanded hashed source files retained',
 'f1/replay_results.json':'uncompressed replay; gzip retained',
 'f1/strict_arrival_replay_results.json':'uncompressed replay; archive retained separately',
 'p2/tariffs/se3_20250915_16_raw.json':'restricted real-price raw data',
 'p2/tariffs/se3_20250915_h26.csv':'restricted real-price-derived CSV',
 'p2/tariffs/provenance.json':'contains restricted real-price extrema; public metadata copy retained',
}
PREFIXES={'f4/pinned/':'vendored pinned source tree','f9/register_preview/':'large redundant preview','f6/sources/':'large copied primary artifacts'}
PATTERNS={
 'github_token':re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})'),
 'openai_secret':re.compile(r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}'),
 'credential_url':re.compile(r'https?://[^\s/<>"\']+:[^\s/<>"\']+@'),
 'private_key':re.compile(r'-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----'),
 'bearer_token':re.compile(r'(?i)authorization\s*[:=]\s*["\']?bearer\s+[A-Za-z0-9._-]{20,}'),
 'gurobi_secret':re.compile(r'(?i)\b(?:WLSSECRET|WLSACCESSID)\s*=\s*[A-Za-z0-9-]{16,}'),
}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 source=E/'p2/tariffs/provenance.json'
 if source.exists():
  value=json.loads(source.read_text());value.pop('min_eur_kwh',None);value.pop('max_eur_kwh',None)
  (source.parent/'provenance_public.json').write_text(json.dumps(value,indent=2)+'\n')
 # Preserve a compressed strict-arrival audit before omitting its uncompressed copy.
 p=E/'f1/strict_arrival_replay_results.json'
 if p.exists():
  target=p.with_suffix(p.suffix+'.gz')
  with gzip.GzipFile(filename=str(target),mode='wb',mtime=0) as f:f.write(p.read_bytes())
 include=[];exclude=[];findings=[];bundles=[]
 roots = [E, B/'time_only_vsp_20260916']
 for p in sorted(p for root in roots if root.exists() for p in root.rglob('*')):
  if not p.is_file():continue
  rel=p.relative_to(E).as_posix() if E in p.parents else '../'+p.relative_to(B).as_posix();reason=EXACT.get(rel)
  for prefix,why in PREFIXES.items():
   if rel.startswith(prefix):reason=why
  if '__pycache__' in p.parts or p.suffix=='.pyc':reason='Python cache'
  if 'private_internal' in p.parts:reason='restricted real-price-derived output'
  if p.is_symlink():reason='symlink requires separate source review'
  if reason:exclude.append(dict(path=str(p.relative_to(REPO)),reason=reason));continue
  raw=p.read_bytes()
  if p.suffix=='.gz':text=gzip.decompress(raw).decode('utf8')
  elif p.suffix=='.bundle':
   # Git bundles are compressed commits; accompanying patches are scanned.
   text='';bundles.append(str(p.relative_to(REPO)))
  else:text=raw.decode('utf8',errors='replace')
  hits=[]
  for name,pattern in PATTERNS.items():
   for match in pattern.finditer(text):hits.append(dict(pattern=name,line=text[:match.start()].count('\n')+1))
  if hits:
   findings.append(dict(path=str(p.relative_to(REPO)),hits=hits));exclude.append(dict(path=str(p.relative_to(REPO)),reason='potential credential; requires review'));continue
  include.append(dict(path=str(p.relative_to(REPO)),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
 report=dict(generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='execution tree and explicit time_only_vsp_20260916 audit; no commit/copy performed',included_files=len(include),included_bytes=sum(x['bytes'] for x in include),excluded_files=len(exclude),credential_findings=findings,binary_git_bundles=bundles,notes=['Git bundles contain newly authored code commits; accompanying text patches/code are separately reviewed. No source-tree .git directories are included.','This is a point-in-time allowlist. Re-run after edits and recheck file hashes before publication.','SE3 source URL/date/zone/license and content hashes are public provenance; numerical prices and derived result values are excluded.'],files=include,excluded=exclude)
 (B/'PUBLICATION_ALLOWLIST.json').write_text(json.dumps(report,indent=2)+'\n');(B/'PUBLICATION_ALLOWLIST.txt').write_text('\n'.join(x['path'] for x in include)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['files','excluded','binary_git_bundles','notes']}))
if __name__=='__main__':main()
