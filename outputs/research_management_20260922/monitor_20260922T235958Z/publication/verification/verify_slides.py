"""Read-only final PPTX/PDF preservation and evidence checks."""
from pathlib import Path
import json,hashlib,re,zipfile,xml.etree.ElementTree as ET,posixpath
from collections import Counter
from pypdf import PdfReader
P=Path(__file__).resolve().parent.parent
N={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
txt=lambda x:' '.join(t.text or '' for t in ET.fromstring(x).findall('.//a:t',N))
norm=lambda x:re.sub(rb'\{[0-9A-Fa-f-]{36}\}',b'{EXPORT_GUID}',x)
checks=[]
def ck(n,v):checks.append({'check':n,'passed':bool(v)})
def note_body(data):
 root=ET.fromstring(data)
 for sp in root.findall('.//p:sp',N):
  ph=sp.find('./p:nvSpPr/p:nvPr/p:ph',N)
  if ph is not None and ph.get('type')=='body':return ' '.join(t.text or '' for t in sp.findall('.//a:t',N))
 raise ValueError('speaker notes body not found')
def images(z,n):
 rel='ppt/slides/_rels/'+Path(n).name+'.rels';out={}
 if rel not in z.namelist():return out
 for r in ET.fromstring(z.read(rel)):
  if r.attrib.get('Type','').endswith('/image'):
   target=posixpath.normpath(posixpath.join('ppt/slides',r.attrib['Target']));out[r.attrib['Id']]=hashlib.sha256(z.read(target)).hexdigest()
 return out
with zipfile.ZipFile(P/'slides_before.pptx') as a,zipfile.ZipFile(P/'slides_after.pptx') as b:
 names=[n for n in a.namelist() if re.fullmatch(r'ppt/slides/slide\d+.xml',n)]
 ck('42slides same membership',len(names)==42 and set(names)==set(n for n in b.namelist() if re.fullmatch(r'ppt/slides/slide\d+.xml',n)))
 changed=[int(re.search(r'slide(\d+)',n).group(1)) for n in names if norm(a.read(n))!=norm(b.read(n))]
 ck('only slides10/42 semantic edits',sorted(changed)==[10,42])
 ck('table definitions preserved modulo export GUID',norm(a.read('ppt/tableStyles.xml'))==norm(b.read('ppt/tableStyles.xml')))
 media=lambda z:Counter(hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('ppt/media/'))
 ck('all embedded image bytes preserved',media(a)==media(b))
 ck('image shape references preserved',all(images(a,n)==images(b,n) for n in names))
 tables=lambda z:[ET.tostring(t) for t in ET.fromstring(z.read('ppt/slides/slide42.xml')).findall('.//a:tbl',N)]
 ck('k33 native editable table unchanged',len(tables(a))==len(tables(b))==1 and norm(tables(a)[0])==norm(tables(b)[0]))
 notes=[n for n in a.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+.xml',n)]
 changednotes=[int(re.search(r'notesSlide(\d+)',n).group(1)) for n in notes if a.read(n)!=b.read(n)]
 ck('only notes10/42 changed',sorted(changednotes)==[10,42])
 for number in [10,42]:
  old=note_body(a.read(f'ppt/notesSlides/notesSlide{number}.xml'));new=note_body(b.read(f'ppt/notesSlides/notesSlide{number}.xml'))
  ck(f'notes{number} preserve entire old body before append',new.startswith(old))
  ck(f'notes{number} have new evidence reference','monitor_20260922T235958Z' in new)
 s10=txt(b.read('ppt/slides/slide10.xml'));s42=txt(b.read('ppt/slides/slide42.xml'))
 ck('slide10 running scope with no final result','cached CG is running after native license checks' in s10 and 'No final result yet' in s10)
 ck('slide10 historical physics and graph evidence retained','331 trips from 11 reference duties' in s10 and 'shared capacity is not enforced' in s10 and '8,397 initial routes agree after reload (16 s)' in s10)
 ck('slide42 caption newdate unchanged proof limits','20:01 EDT' in s42 and 'not certified LP lower bounds' in s42 and 'shared capacity remain unvalidated' in s42)
 n10=note_body(b.read('ppt/notesSlides/notesSlide10.xml'));n42=note_body(b.read('ppt/notesSlides/notesSlide42.xml'))
 ck('notes10 correct currentjob and startup','824877' in n10 and ('15 startup' in n10 or '15/15' in n10) and 'no final' in n10.lower())
 ck('notes42 latest fleet and separate scopes','37' in n42 and '34' in n42 and '237,336' in n42 and '226 extra' in n42)
pdf=PdfReader(P/'slides_after.pdf');ck('42PDFpages',len(pdf.pages)==42)
ck('PDF10newrunningtext','cached CG is running' in ' '.join(pdf.pages[9].extract_text().split()))
ck('PDF42newtimestamp','20:01 EDT' in ' '.join(pdf.pages[41].extract_text().split()))
result={'status':'passed' if all(c['passed'] for c in checks) else 'failed','checks_passed':sum(c['passed'] for c in checks),'checks_total':len(checks),'checks':checks,'export_hashes':{n:sha(P/n) for n in ['slides_before.pptx','slides_after.pptx','slides_after.pdf']},'normalization':'Only generated GUIDs; actual table definitions and per-slide image references checked separately','scope':'Final local export check; initial notes draft excluded; no live edits'}
(P/'verification/slides_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'passed':result['checks_passed'],'total':result['checks_total'],'failures':[x for x in checks if not x['passed']]},indent=2))
