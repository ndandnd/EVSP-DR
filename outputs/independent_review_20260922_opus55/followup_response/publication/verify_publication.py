"""Verify scoped Doc/Slides edits against fresh before/after exports."""
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as E
import hashlib,json,re,posixpath
P=Path(__file__).resolve().parent
N={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
checks=[]
def check(name,test):checks.append(dict(check=name,passed=bool(test)))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def text(data):return '\n'.join(t.text or '' for t in E.fromstring(data).findall('.//a:t',N))
def note(data):
 for sp in E.fromstring(data).findall('.//p:sp',N):
  ph=sp.find('.//p:ph',N)
  if ph is not None and ph.get('type')=='body':return ''.join(t.text or '' for t in sp.findall('.//a:t',N))
 raise ValueError('no notes')
def media(z):return sorted(hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('ppt/media/'))
old=(P/'doc_before.md').read_text();new=(P/'doc_after.md').read_text()
a='In 24 matched cases, fresh and sequential CG both certify their LP endpoints, but their one-hour MIPs match 6 versus 24 GIRO targets.'
b='In 24 matched cases, both methods certify the event-grid LP. Their one-hour MIPs reach the minimum baseline fleet in 6/24 fresh cases versus 24/24 sequential cases. New exact time-and-travel certificates establish these fleet minima; charging optimality and full GIRO compliance do not follow. Sequential growth adds complete GIRO duties.'
c='Its effect on real fleets remains unmeasured. Numerical results are unchanged; the original review and corrections are preserved separately.'
u='https://github.com/ndandnd/EVSP-DR/blob/codex/week-evidence-20260921/outputs/independent_review_20260922_opus55/followup_response/README.md'
d=f'The [follow-up and exact certificates]({u}) now establish all 24 benchmark fleet minima. C1/C3 k15 still require 15 buses without the 57-minute cap. Charging optimality and broader operating constraints remain separate.'
check('exactly two intended Doc replacements',old.count(a)==old.count(c)==1 and old.replace(a,b).replace(c,d)==new)
check('Doc table rows unchanged',[s for s in old.splitlines() if s.startswith('|')]==[s for s in new.splitlines() if s.startswith('|')])
check('Doc image references and inline data unchanged',[s for s in old.splitlines() if s.startswith(('![','[image'))]==[s for s in new.splitlines() if s.startswith(('![','[image'))])
check('Doc source links retained',set(re.findall(r'https?://[^\s)]+',old))<=set(re.findall(r'https?://[^\s)]+',new)))
with ZipFile(P/'slides_before.pptx') as z,ZipFile(P/'slides_after.pptx') as y:
 slidekeys=lambda f:sorted(n for n in f.namelist() if re.fullmatch(r'ppt/slides/slide\d+.xml',n))
 keys=slidekeys(z)
 check('42 slides preserved',len(keys)==42 and keys==slidekeys(y))
 changed=[n for n in keys if text(z.read(n))!=text(y.read(n))]
 check('only slides2and28 visible text changed',set(changed)=={'ppt/slides/slide2.xml','ppt/slides/slide28.xml'})
 check('all images identical',media(z)==media(y))
 for n in [2,28]:
  key=f'ppt/notesSlides/notesSlide{n}.xml'
  before=note(z.read(key));after=note(y.read(key))
  compact=lambda t:re.sub(r'\s+','',t)
  check(f'slide{n} previous notes preserved',compact(after).startswith(compact(before)))
  check(f'slide{n} exact source link in notes',u in after)
  check(f'slide{n} scope in notes','covering' in after and 'charging' in after.lower())
 others=[n for n in z.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+.xml',n) and n not in [f'ppt/notesSlides/notesSlide{i}.xml' for i in [2,28]]]
 check('all other notes unchanged',all(note(z.read(n))==note(y.read(n)) for n in others))
 norm=lambda v:re.sub(rb'\{[0-9a-fA-F-]{36}\}',b'{GUID}',v)
 check('all other slide XML unchanged except export GUID',all(norm(z.read(n))==norm(y.read(n)) for n in keys if n not in changed))
 table=lambda f:[norm(E.tostring(t)) for n in keys for t in E.fromstring(f.read(n)).findall('.//a:tbl',N)]
 check('all editable tables preserved except export GUID',table(z)==table(y))
 check('table style definitions unchanged except export GUID',norm(z.read('ppt/tableStyles.xml'))==norm(y.read('ppt/tableStyles.xml')))
cert=json.loads((P.parent/'time_bounds.json').read_text());rows=cert['cases'];small=[r for r in rows if r['cohort']=='figure1']
check('35 exact witnesses',len(rows)==35 and all(r['no_antichain_pair_connected_by_any_path'] for r in rows))
check('24 sequential baseline fleet optima',len(small)==24 and all(r['sequential_fleet_matches_lower_bound'] for r in small))
check('6 fresh baseline fleet optima',sum(r['fresh_fleet_matches_lower_bound'] for r in small)==6)
check('script matches recorded execution',sha(P.parent/'time_bound_check.py')==cert['source_hashes']['time_bound_check.py'])
result=dict(status='passed' if all(c['passed'] for c in checks) else 'failed',checks_passed=sum(c['passed'] for c in checks),checks_total=len(checks),checks=checks,export_hashes={n:sha(P/n) for n in ['doc_before.md','doc_after.md','slides_before.pptx','slides_after.pptx']},scope='Current Doc body and slides2/28 only. All seven tab names verified in UI; other tab bodies not re-exported. PDF visual review of slides2/28 passed; Doc review paragraph inspected in UI.')
(P/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['checks','export_hashes']},indent=2));print([x for x in checks if not x['passed']])
assert result['status']=='passed'
