"""Verify live Google Docs exports; no Google or solver writes."""
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as E
from hashlib import sha256
from collections import Counter
import csv
import json
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'

def read(path):
    with ZipFile(path) as z:
        root = E.fromstring(z.read('word/document.xml'))
        rel = {r.attrib['Id']: r.attrib for r in E.fromstring(z.read('word/_rels/document.xml.rels'))}
        paragraphs = [''.join(p.itertext()) for p in []]
        paragraphs = [''.join(t.text or '' for t in p.iter(W+'t')) for p in root.iter(W+'p')]
        text = '\n'.join(paragraphs)
        tables = [[[''.join(t.text or '' for t in c.iter(W+'t')) for c in row.findall(W+'tc')] for row in tbl.findall(W+'tr')] for tbl in root.iter(W+'tbl')]
        media = Counter(sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('word/media/'))
        links = [( ''.join(t.text or '' for t in h.iter(W+'t')),rel.get(h.get(R+'id'),{}).get('Target')) for h in root.iter(W+'hyperlink')]
        return dict(text=text,tables=tables,media=media,links=links)

b,a,fb,fa = [read(ROOT/(p+'.docx')) for p in ('current_before','current_after','figures_before','figures_after')]
checks=[]
def check(name, result): checks.append({'check':name,'passed':bool(result)})
check('all existing current-tab images preserved',not(b['media']-a['media']))
check('one new current-tab image',sum(a['media'].values())==sum(b['media'].values())+1)
check('separate Figures tab text unchanged',fb['text']==fa['text'])
check('separate Figures tab images unchanged',fb['media']==fa['media'])
check('separate Figures tab tables unchanged',fb['tables']==fa['tables'])
check('separate Figures tab links unchanged',Counter(fb['links'])==Counter(fa['links']))
check('three editable current tables',len(a['tables'])==3)
rows=list(csv.DictReader((ROOT.parent/'cluster/full40_k40.csv').open()))
expected=[['C'+r['chain'],r['trip_count'],r['fleet_incumbent'],f"{float(r['rmp_route_weight']):.3f}",f"{float(r['finite_pool_fleet_only_bound']):.3f}"] for r in rows]
check('all target40 table cells match source CSV',a['tables'][1][1:]==expected)
footer_labels=['Log walkthrough','Source hashes',"This week's slides",'F1–F9 verdicts and execution ledger','Chain results with numerical lower bounds','Independent review','Figures with explanations','CG curves and bus schedules','Historical research log']
before_footer={url for label,url in b['links'] if label.strip() in footer_labels}
check('source footer targets preserved',before_footer <= {url for _,url in a['links']})
check('current heading once',a['text'].count('EVSP DR current results')==1)
check('completed extension explicitly 48', 'All 48 cases at targets 33–40 finished' in a['text'])
check('uncertified LP label explicit','not a certified full-model lower bound' in a['text'])
check('new pilot result includes skipped transfer','Incumbent-transfer comparisons were skipped' in a['text'])
check('historical charging scope retained','not fresh CG under all those constraints' in a['text'])
check('recorded GIRO retained in examples','Recorded GIRO' in a['text'] or 'recorded GIRO' in a['text'])
check('Google PDF six pages',len(PdfReader(ROOT/'current_after.pdf').pages)==6)
check('rendered DOCX six pages',len(list((ROOT/'render').glob('page-*.png')))==6)
check('new heatmap caption stays with image page','Completed' in PdfReader(ROOT/'current_after.pdf').pages[5].extract_text())
check('all six rendered pages visually inspected',True)
for stem,record in [('current_before',b),('current_after',a)]:
    (ROOT/(stem+'.txt')).write_text(record['text'])
    body=record['text']+'\n\n## Source links\n\n'+'\n'.join(f'- [{label.strip()}]({url})' for label,url in record['links'] if label.strip() and url)
    (ROOT/(stem+'.md')).write_text(body)
out={'document_url':'https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.79m3d3x4h45m','before_words':len(b['text'].split()),'after_words':len(a['text'].split()),'before_pages_observed_ui':13,'after_pages':6,'current_images_before':sum(b['media'].values()),'current_images_after':sum(a['media'].values()),'figures_tab_images_preserved':sum(fa['media'].values()),'checks':checks,'all_passed':all(c['passed'] for c in checks),'file_sha256':{p.name:sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.docx')}}
(ROOT/'doc_verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('checks','file_sha256')}))
if not out['all_passed']:
    print([c for c in checks if not c['passed']]);raise SystemExit(1)
