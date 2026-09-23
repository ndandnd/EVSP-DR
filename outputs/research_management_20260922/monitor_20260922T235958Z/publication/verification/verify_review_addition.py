"""Additive final stage: preserve original 38-check receipts and all prior content."""
from pathlib import Path
import json,hashlib,re,zipfile,xml.etree.ElementTree as ET,posixpath
from collections import Counter
from pypdf import PdfReader
P=Path(__file__).resolve().parent.parent;ROOT=P.parents[3]
N={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
checks=[]
def ck(n,v):checks.append({'check':n,'passed':bool(v)})
old=(P/'before_review_link_current_after.md').read_text();new=(P/'current_after.md').read_text()
paras=[x for x in new.split('\n\n') if x.startswith(('[Figure guide:]','[Independent Opus review, checked:]'))]
ck('exactly two designated added paragraphs',len(paras)==2)
rest=new
for paragraph in paras:rest=rest.replace(paragraph+'\n\n','',1)
ck('remove two additions gives exact previous final Doc bytes',rest==old)
ck('figure explanation accurately scoped','pool MIP selects saved routes; integer-directed pricing creates complementary routes. Heuristic fleet solutions are upper bounds.' in paras[0] and '512-route limit and packed graphs' in paras[0])
ck('review restriction and unmeasured impact exact','direct trip connections are capped at 57-minute gaps, and station bridges require positive charging. Its effect on real fleets remains unmeasured.' in paras[1])
ck('no numerical or certificate reclassification','Numerical results are unchanged; the original review and corrections are preserved separately.' in paras[1])
figure_url='https://github.com/ndandnd/EVSP-DR/blob/codex/week-evidence-20260921/outputs/research_management_20260922/figure_explanations/README.md'
review_url='https://github.com/ndandnd/EVSP-DR/blob/codex/week-evidence-20260921/outputs/independent_review_20260922_opus55/ASSESSMENT.md'
ck('exact two source links',figure_url in paras[0] and review_url in paras[1])
figure=ROOT/'outputs/research_management_20260922/figure_explanations/README.md';assessment=ROOT/'outputs/independent_review_20260922_opus55/ASSESSMENT.md';longwait=assessment.parent/'audit_long_wait.md'
f=figure.read_text();a=assessment.read_text();l=longwait.read_text()
ck('figure guide supports pool and512 explanation','It cannot create a missing route' in f and '512 is an earlier limit on the number of inherited routes' in f and 'upper bound' in f)
ck('assessment supports exact restriction scope','direct connections are limited to 57-minute gaps, and station bridges require positive charging' in a and 'whether this restriction changes production fleets or charging costs is unresolved' in a)
ck('longwait audit defines gap precisely','next start − previous end >57' in l and 'not when idle time' in l and 'synthetic representability witness' in l)
prior_doc=read(P/'verification/doc_verification.json');prior_slides=read(P/'verification/slides_verification.json');initial=read(P/'verification/verification.json')
ck('original38checkstage preserved',initial['automated_checks_passed']==initial['automated_checks_total']==38)
ck('retained Doc matches originally audited bytes',sha(P/'before_review_link_current_after.md')==prior_doc['hashes']['current_after.md'] and sha(P/'before_review_link_current_after.pdf')==prior_doc['hashes']['current_after.pdf'])
ck('retained deck matches originally audited bytes',sha(P/'before_review_link_slides_after.pptx')==prior_slides['export_hashes']['slides_after.pptx'])
norm=lambda x:re.sub(rb'\{[0-9A-Fa-f-]{36}\}',b'{EXPORT_GUID}',x)
def body(data):
 for sp in ET.fromstring(data).findall('.//p:sp',N):
  ph=sp.find('./p:nvSpPr/p:nvPr/p:ph',N)
  if ph is not None and ph.get('type')=='body':return ' '.join(t.text or '' for t in sp.findall('.//a:t',N))
 raise ValueError('notes body missing')
def images(z,n):
 rel='ppt/slides/_rels/'+Path(n).name+'.rels';out={}
 if rel not in z.namelist():return out
 for r in ET.fromstring(z.read(rel)):
  if r.attrib.get('Type','').endswith('/image'):
   target=posixpath.normpath(posixpath.join('ppt/slides',r.attrib['Target']));out[r.attrib['Id']]=hashlib.sha256(z.read(target)).hexdigest()
 return out
with zipfile.ZipFile(P/'before_review_link_slides_after.pptx') as b,zipfile.ZipFile(P/'slides_after.pptx') as c:
 slides=[n for n in b.namelist() if re.fullmatch(r'ppt/slides/slide\d+.xml',n)]
 ck('all42visible slides unchanged',len(slides)==42 and all(norm(b.read(n))==norm(c.read(n)) for n in slides))
 ck('table style definitions unchanged',norm(b.read('ppt/tableStyles.xml'))==norm(c.read('ppt/tableStyles.xml')))
 ck('all slide image references unchanged',all(images(b,n)==images(c,n) for n in slides))
 notes=[n for n in b.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+.xml',n)]
 ck('only notes42change',[n for n in notes if b.read(n)!=c.read(n)]==['ppt/notesSlides/notesSlide42.xml'])
 beforebody=body(b.read('ppt/notesSlides/notesSlide42.xml'));afterbody=body(c.read('ppt/notesSlides/notesSlide42.xml'))
 ck('entire previous notes42 preserved as prefix',afterbody.startswith(beforebody))
 appendix=afterbody[len(beforebody):]
 ck('notes appendix restriction qualification and sources','57-minute gaps' in appendix and 'positive charging' in appendix and 'production fleet/cost impact is unmeasured' in appendix and figure_url in appendix and review_url in appendix)
 ck('notes appendix preserves numerical certificate scope','No numerical results or certificates were changed.' in appendix)
ck('visible Slides PDF unchanged from38checkstage',sha(P/'slides_after.pdf')==prior_slides['export_hashes']['slides_after.pdf'])
pdf=PdfReader(P/'current_after.pdf');text=' '.join(' '.join((x.extract_text() or '').split()) for x in pdf.pages)
ck('new paragraphs present in finalPDF','Figure guide:' in text and 'Independent Opus review, checked:' in text and '57-minute gaps' in text)
result={'status':'passed' if all(x['passed'] for x in checks) else 'failed','checks_passed':sum(x['passed'] for x in checks),'checks_total':len(checks),'checks':checks,'final_export_sha256':{n:sha(P/n) for n in ['current_after.md','current_after.pdf','slides_after.pptx','slides_after.pdf']},'original_stage_receipt_sha256':sha(P/'verification/verification.json'),'source_sha256':{str(x.relative_to(ROOT)):sha(x) for x in [figure,assessment,longwait]},'doc_pdf_pages':len(pdf.pages),'scope':'Two Doc paragraphs plus slide42notes appendix only; original38checkstage unchanged; no UI or cluster calls'}
(P/'verification/review_addition_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'passed':result['checks_passed'],'total':result['checks_total'],'failures':[x for x in checks if not x['passed']],'docpages':len(pdf.pages)},indent=2))
