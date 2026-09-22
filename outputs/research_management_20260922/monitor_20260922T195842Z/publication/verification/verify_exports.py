"""Read-only export checks. Run with bundled Python containing pypdf."""
from pathlib import Path
import hashlib,json,re,zipfile,xml.etree.ElementTree as ET
from collections import Counter
import posixpath
from pypdf import PdfReader
P=Path(__file__).resolve().parent.parent
M=P.parent
N={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
checks=[]
def ck(name,value):
 checks.append({'check':name,'passed':bool(value)})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def txt(xml):return ' '.join((t.text or '') for t in ET.fromstring(xml).findall('.//a:t',N))
def normalized(data):
 return re.sub(rb'\{[0-9A-Fa-f-]{36}\}',b'{EXPORT_STYLE_GUID}',data)
before=(P/'current_before.md').read_text();after=(P/'current_after.md').read_text()
ops=json.loads((M/'operations/verified_summary.json').read_text());strict=json.loads((M/'strict_review/graph_gate_audit.json').read_text())
ck('source operations423/423',ops['checks_passed']==ops['checks_total']==423)
ck('source strict graph gate passed',strict['status']=='passed' and strict['receipt_hash_chain_verified'] and strict['fresh_initial_pool_full_record_equality'])
# Only the existing extension section and the strict graph paragraph may change.
def unaffected(text):
 text=re.sub(r'\*\*22 September,.*?(?=This extension retains)', '[EXTENSION]',text,flags=re.S)
 text=re.sub(r'The packed recovery is complete\..*?(?=\n\n\[Completed graph benchmark)', '[STRICT]',text,flags=re.S)
 return text
ck('Doc outside approved two regions identical',unaffected(before)==unaffected(after))
urls=lambda x:set(re.findall(r'https?://[^)\s]+',x))
ck('all prior Doc source/history/figure links retained',urls(before)<=urls(after))
ck('Doc current scheduler values','41/44' in after and 'All 62 pending' in after and 'nine baseline jobs' in after)
ck('Doc C5 exact row','| C5 | 768 | 239.7 | 32 | 39 / 32' in after)
ck('Doc C6 exact row','| C6 | 796 | 239.9 | 32 | 38 / 32' in after)
ck('Doc six uncertified CGs','All six CGs hit the four-hour limit without a pricing certificate' in after)
ck('Doc MIP limitations retained','All six completed MIPs missed 33' in after and 'duplicate removal and shared capacity remain unvalidated' in after)
ck('Doc new duplicate counts','C5 397, C6 182' in after)
ck('Doc larger fleet/bound tuples','42/36/42 buses, with pool bounds 34/35/34' in after and 'C2/C6 found 36/40 buses, both with pool bound 33' in after)
ck('Doc historical C2 discrepancy retained','0.038 cost units' in after)
ck('Doc strict graph measured reload/RSS','15.96 seconds' in after and '6.08 GiB scheduler peak memory' in after and abs(strict['reload_graph_s']-15.96)<.005)
ck('Doc graph proof scope correct','8,397 initial routes reproduce exactly, two diagnostic pricing queries agree' in after and 'not CG optimality' in after and 'separately from the next four-hour CG run' in after)
with zipfile.ZipFile(P/'slides_before.pptx') as a,zipfile.ZipFile(P/'slides_after.pptx') as b:
 names=sorted(x for x in a.namelist() if re.fullmatch(r'ppt/slides/slide\d+.xml',x))
 ck('42 slides preserved',len(names)==42 and set(names)==set(x for x in b.namelist() if re.fullmatch(r'ppt/slides/slide\d+.xml',x)))
 changed=[int(re.search(r'slide(\d+)',x).group(1)) for x in names if normalized(a.read(x))!=normalized(b.read(x))]
 ck('only slides10/42 change after export GUID normalization',sorted(changed)==[10,42])
 ck('table style definitions unchanged after GUID normalization',normalized(a.read('ppt/tableStyles.xml'))==normalized(b.read('ppt/tableStyles.xml')))
 notes=[x for x in a.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+.xml',x)]
 changed_notes=[int(re.search(r'notesSlide(\d+)',x).group(1)) for x in notes if a.read(x)!=b.read(x)]
 ck('only notes10/42 change',sorted(changed_notes)==[10,42])
 media=[x for x in a.namelist() if x.startswith('ppt/media/')]
 ck('all embedded historical figure bytes preserved despite export filename reassignment',Counter(hashlib.sha256(a.read(x)).hexdigest() for x in media)==Counter(hashlib.sha256(b.read(x)).hexdigest() for x in b.namelist() if x.startswith('ppt/media/')))
 def image_refs(z,slide):
  rel='ppt/slides/_rels/'+Path(slide).name+'.rels'
  if rel not in z.namelist():return {}
  out={}
  for item in ET.fromstring(z.read(rel)):
   if item.attrib.get('Type','').endswith('/image'):
    target=posixpath.normpath(posixpath.join('ppt/slides',item.attrib['Target']))
    out[item.attrib['Id']]=hashlib.sha256(z.read(target)).hexdigest()
  return out
 ck('every slide image reference resolves identical bytes',all(image_refs(a,n)==image_refs(b,n) for n in names))
 s10=txt(b.read('ppt/slides/slide10.xml'));s42=txt(b.read('ppt/slides/slide42.xml'))
 ck('slide10 exact graph scope','331 trips from 11 reference duties' in s10 and '8,397 initial routes agree after reload (16 s)' in s10 and 'Next: four hours of cached CG' in s10)
 ck('slide10 omitted shared capacity clear','shared capacity is not enforced' in s10)
 x=ET.fromstring(b.read('ppt/slides/slide42.xml'));table=x.find('.//a:tbl',N)
 rows=[[''.join(t.text or '' for t in cell.findall('.//a:t',N)) for cell in row.findall('a:tc',N)] for row in table.findall('a:tr',N)]
 expected=[['Chain','Trips','CG minutes','Fractional buses*','Integer buses','Pool bound'],['C1','785','239.3','33','36','33'],['C2','787','239.9','32','36','32'],['C3','770','239.3','33','34','33'],['C4','785','239.1','32','38','32'],['C5','768','239.7','32','39','32'],['C6','796','239.9','32','38','32']]
 ck('slide42 editable native7x6 table exact values',rows==expected)
 ck('slide42 certificate and physical qualifiers','not certified LP lower bounds' in s42 and 'Duplicate cleanup and shared capacity remain unvalidated' in s42)
 ck('slide42 separate graph and ancestor cost','9.56–15.10 h separately; prior CG excluded' in s42)
 for number in [10,42]:
  note=txt(b.read(f'ppt/notesSlides/notesSlide{number}.xml'))
  ck(f'slide{number} source notes have evidence links','https://github.com/ndandnd/EVSP-DR' in note)
 ck('slide42 notes latest audit reference','monitor_20260922T195842Z' in txt(b.read('ppt/notesSlides/notesSlide42.xml')))
 ck('slide10 notes graph audit reference','monitor_20260922T195842Z' in txt(b.read('ppt/notesSlides/notesSlide10.xml')))
 ck('submittedCG note preserves pending scope','824877' not in after or ('824877' in txt(b.read('ppt/notesSlides/notesSlide10.xml')) and 'was submitted at 17:43 EDT with 8 CPUs, 16 GiB and a five-hour allocation' in after))
doc=PdfReader(P/'current_after.pdf');slides=PdfReader(P/'slides_after.pdf')
ck('PDF page counts11Doc42Slides',len(doc.pages)==11 and len(slides.pages)==42)
ck('visible slide10 and42 PDF content matches exports',all(' '.join((slides.pages[n-1].extract_text() or '').split()).find(key)>=0 for n,key in [(10,'8,397'),(42,'C5'),(42,'239.7')]))
files=['current_before.md','current_after.md','slides_before.pptx','slides_after.pptx','current_after.pdf','slides_after.pdf']
result={'status':'passed' if all(x['passed'] for x in checks) else 'failed','checks_passed':sum(x['passed'] for x in checks),'checks_total':len(checks),'checks':checks,'source_operations_sha256':sha(M/'operations/verified_summary.json'),'source_strict_audit_sha256':sha(M/'strict_review/graph_gate_audit.json'),'export_sha256':{name:sha(P/name) for name in files},'normalized_export_variation':'Table-style GUIDs and media filenames reassigned by export; actual style definitions and every resolved image reference compared separately','visual_review':{'status':'passed','rendered_slides':[10,42],'rendered_doc_pages':[2,3,6,7],'findings':'Readable, no clipping or overlap; complete six-chain native table fits, caption below table. Updated submission text and source link spacing also render cleanly.'},'scope':'Local export verification; no live edits, SSH or solver activity'}
(P/'verification/verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'checks_passed':result['checks_passed'],'checks_total':result['checks_total'],'failures':[x for x in checks if not x['passed']]},indent=2))
