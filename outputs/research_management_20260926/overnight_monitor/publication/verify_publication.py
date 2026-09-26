"""Verify the narrow native Doc/Slides edits from their before/after exports."""
from pathlib import Path
from zipfile import ZipFile
import collections
import hashlib
import json
import posixpath
import re
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
checks = []
def check(name, passed):
    checks.append({'name': name, 'passed': bool(passed)})
    assert passed, name
def digest(data):
    return hashlib.sha256(data).hexdigest()
def normalize_slide(data):
    # Google generates fresh table-style UUIDs on export; content/style is stable.
    return re.sub(rb'<a:tableStyleId>.*?</a:tableStyleId>',
                  b'<a:tableStyleId>EXPORT-UUID</a:tableStyleId>', data)
def texts(data):
    return [e.text or '' for e in ET.fromstring(data).iter() if e.tag.endswith('}t')]
def media_bindings(z, slide):
    r = f'ppt/slides/_rels/slide{slide}.xml.rels'
    if r not in z.namelist():
        return {}
    return {e.attrib['Id']: digest(z.read(posixpath.normpath('ppt/slides/'+e.attrib['Target'])))
            for e in ET.fromstring(z.read(r))
            if e.attrib['Type'].endswith('/image')}

before = (ROOT/'doc_before.md').read_text()
after = (ROOT/'doc_after.md').read_text()
al, bl = before.splitlines(), after.splitlines()
check('Doc line count retained', len(al) == len(bl))
changed = [i+1 for i, (a,b) in enumerate(zip(al,bl)) if a != b]
check('Only date and two current-work paragraphs changed', changed == [3,68,70])
check('Doc figures and source footer byte-identical',
      before.split('## **Figures and examples to keep**',1)[1] ==
      after.split('## **Figures and examples to keep**',1)[1])
check('Doc recovery ID and proof limits present',
      all(s in after for s in ('520378','61/80','55/80','54 from fresh','25 from fallbacks',
                              'no shared charger limits','60 pools','open in 19')))
with ZipFile(HERE/'slides_before.pptx') as a, ZipFile(HERE/'slides_after.pptx') as b:
    an = [n for n in a.namelist() if re.fullmatch(r'ppt/slides/slide\d+.xml',n)]
    bn = [n for n in b.namelist() if re.fullmatch(r'ppt/slides/slide\d+.xml',n)]
    check('One slide appended:54 to55',len(an)==54 and len(bn)==55)
    for i in range(1,55):
        n=f'ppt/slides/slide{i}.xml'
        check(f'Slide{i} retained except generated table UUID',normalize_slide(a.read(n))==normalize_slide(b.read(n)))
        check(f'Slide{i} image bindings retained',media_bindings(a,i)==media_bindings(b,i))
        notes=f'ppt/notesSlides/notesSlide{i}.xml'
        if notes in a.namelist():
            check(f'Slide{i} notes retained',texts(a.read(notes))==texts(b.read(notes)))
    media = lambda z: collections.Counter(digest(z.read(n)) for n in z.namelist() if n.startswith('ppt/media/'))
    check('All prior embedded images retained',media(a)==media(b))
    new=b.read('ppt/slides/slide55.xml')
    check('New result table is native editable table',new.count(b'<a:tbl>')==1)
    s=' '.join(texts(new))
    check('New slide counts and caveats',all(t in s for t in ('61 / 80','55 / 80','79 / 80','60 / 79','GIRO-based','No shared charger limits')))
    notes=' '.join(texts(b.read('ppt/notesSlides/notesSlide55.xml')))
    check('New slide source and recovery notes',all(t in notes for t in ('terminal_receipts_0400.json','520378','350 kW','17 fallback','overbroad')))
files = ['slides_before.pptx','slides_after.pptx','slides_after.pdf','slide55.png']
report={'checks':checks,'passed':len(checks),'failed':0,'doc_changed_lines':changed,
        'hashes':{n:digest((HERE/n).read_bytes()) for n in files},
        'doc_hashes':{n:digest((ROOT/n).read_bytes()) for n in ('doc_before.md','doc_after.md')},
        'visual_review':'Slide55 rendered from native Google PDF and inspected; text/table readable, no clipping. Doc status paragraph inspected in browser.',
        'slides_url':'https://docs.google.com/presentation/d/1F6udjgkiPH51vMUcku7ZT3PhZPAxsQwUCi3TK9vFRDM/edit?slide=id.h5de1531e14aadef0_4_1',
        'doc_url':'https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.79m3d3x4h45m'}
(HERE/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'{len(checks)} publication checks passed')
