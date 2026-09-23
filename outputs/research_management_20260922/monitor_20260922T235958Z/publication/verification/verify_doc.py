"""Read-only Doc export verification; use bundled Python with pypdf."""
from pathlib import Path
import json,hashlib,re
from pypdf import PdfReader
P=Path(__file__).resolve().parent.parent;M=P.parent
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
b=(P/'current_before.md').read_text();a=(P/'current_after.md').read_text();checks=[]
def ck(n,v):checks.append({'check':n,'passed':bool(v)})
ops=read(M/'operations/verified_summary.json');startup=read(M/'operations/strict_cg/startup_audit.json')
ck('source252baseline checks',ops['checks_passed']==ops['checks_total']==252)
ck('source15startup checks',startup['checks_passed']==startup['checks_total']==15)
def stable(t):
 t=re.sub(r'^\*\*22 September,.*$', '[STATUS]',t,flags=re.M)
 t=re.sub(r'^Latest larger endpoints:.*$', '[LARGER]',t,flags=re.M)
 key='Graph preparation is accounted for separately from the next four-hour CG run.'
 start=t.index(key)+len(key);end=t.index('\n\n[Completed graph benchmark',start)
 t=t[:start]+'[STRICT UPDATE]'+t[end:]
 # A standalone source link may be inserted beside either affected section.
 t=re.sub(r'^\[22 September, 20:01 EDT:[^\n]+\]\(https://github.com/[^\n]*monitor_20260922T235958Z/operations\)\n*','',t,flags=re.M)
 return t
ck('all other Doc text exact',stable(a)==stable(b))
urls=lambda t:set(re.findall(r'https?://[^)\s]+',t))
ck('all prior links preserved',urls(b)<=urls(a))
ck('all native table markdown rows preserved',[x for x in a.splitlines() if x.startswith('|')]==[x for x in b.splitlines() if x.startswith('|')])
ck('source evidence link once',a.count('monitor_20260922T235958Z/operations')==1)
status=next(x for x in a.splitlines() if x.startswith('**22 September,'))
ck('only timestamp bold not whole status',status.startswith('**22 September, 20:01 EDT:** ') and not status.endswith('**'))
ck('queue counts exact','42/44' in status and ('Thirteen' in status or '13' in status or 'thirteen' in status) and '52' in status)
ck('C2k35 new fleet bound37/34','42/37/36/42 buses, with pool bounds 34/34/35/34' in a)
ck('k36 CG weights scoped uncertified','fractional route weights 35/36/35, without pricing certificates' in a)
ck('C6MIP pending and k35CG scope','its k35 MIP is running' in a and 'C2/C6 k35 CGs also stopped without certificates, each with route weight 34' in a)
ck('historical numerical disclosure retained','0.038 cost units' in a)
ck('strict job identity and request retained','Recovery job 824877 was submitted at 17:43 EDT with 8 CPUs, 16 GiB and a five-hour allocation.' in a)
ck('strict running native checks','At 20:01 EDT it was running, restart 0, after the native source/input/cache checks and Gurobi license probe passed.' in a)
ck('strict no endpoint or memory claim','there is no final CG result or pricing certificate yet. Full-run memory remains unmeasured.' in a)
ck('not dangling separated pronoun',not re.search(r'^\s*At 20:01 EDT it was running',a,re.M))
ck('physical limits preserved','duplicate removal and shared capacity remain unvalidated' in a)
pdf=PdfReader(P/'current_after.pdf');texts=[' '.join((x.extract_text() or '').split()) for x in pdf.pages];pdftext=' '.join(texts)
ck('PDF queue and strict identity present','42/44' in pdftext and '824877' in pdftext and 'Full-run memory remains unmeasured.' in pdftext)
ck('PDF retains11pages',len(pdf.pages)==11)
result={'status':'passed' if all(x['passed'] for x in checks) else 'failed','checks_passed':sum(x['passed'] for x in checks),'checks_total':len(checks),'checks':checks,'hashes':{n:sha(P/n) for n in ['current_before.md','current_after.md','current_after.pdf']},'source_hashes':{'operations':sha(M/'operations/verified_summary.json'),'strict_startup':sha(M/'operations/strict_cg/startup_audit.json')},'affected_pdf_pages':[i+1 for i,t in enumerate(texts) if any(s in t for s in ['42/44','Latest larger endpoints','824877'])],'scope':'Local read-only exported Doc QA; no live edits'}
(P/'verification/doc_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'passed':result['checks_passed'],'total':result['checks_total'],'failures':[x for x in checks if not x['passed']],'pages':result['affected_pdf_pages']},indent=2))
