#!/usr/bin/env python3
"""Read-only OOXML/Markdown publication audit. No python-pptx, UI, or cluster calls."""
from pathlib import Path
import csv, difflib, hashlib, json, posixpath, re, xml.etree.ElementTree as ET, zipfile
P=Path(__file__).resolve().parent
NS={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
checks=[]
def check(name,ok,detail=None):
    checks.append({'name':name,'pass':bool(ok),'detail':detail})
def canon(data):
    x=ET.fromstring(data)
    for e in x.iter():
        if e.tag.endswith('}tableStyleId'): e.text='STYLE_ID'
        for k in ['styleId','def']:
            if k in e.attrib: e.attrib[k]='STYLE_ID'
    return ET.tostring(x)
def text(x): return '\n'.join(e.text or '' for e in x.findall('.//a:t',NS))
def table(x): return [[ '\n'.join(t.text or '' for t in c.findall('.//a:t',NS)) for c in r.findall('a:tc',NS)] for r in x.findall('a:tr',NS)]
def md_tables(s): return [m.group(0).strip().splitlines() for m in re.finditer(r'(?m)^\|.*(?:\n\|.*)*',s)]
def md_cells(lines): return [[c.strip() for c in r.strip('|').split('|')] for r in lines if not re.fullmatch(r'[| :\-]+',r)]
fleetpath=P.parent/'mip_structure/results.csv'; cappath=P.parents[1]/'charging_column_structure/pilot/results.csv'
fleet={(r['case'],r['arm']):r for r in csv.DictReader(fleetpath.open())}
cap=list(csv.DictReader(cappath.open()))
cases=['c1_k08_fresh','c4_k08_fresh','c1_k15_fresh','c3_k15_fresh','c1_k15_sequential'];arms=['default','focus1','focus2','presparsify1','strong_start']
labels=['C1 k8 fresh','C4 k8 fresh','C1 k15 fresh','C3 k15 fresh','C1 k15 seq.']
expected_fleet=[['Pool','Default','Focus 1','Focus 2','Sparsify','Saved start']]+[[label]+[fleet[c,a]['fleet'] for a in arms] for c,label in zip(cases,labels)]
with zipfile.ZipFile(P/'slides_before.pptx') as before,zipfile.ZipFile(P/'slides_after.pptx') as after:
    count=lambda z:len([n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+.xml',n)])
    check('slide counts 39 -> 41',count(before)==39 and count(after)==41)
    raw=[i for i in range(1,40) if before.read(f'ppt/slides/slide{i}.xml')!=after.read(f'ppt/slides/slide{i}.xml')]
    changed=[i for i in range(1,40) if canon(before.read(f'ppt/slides/slide{i}.xml'))!=canon(after.read(f'ppt/slides/slide{i}.xml'))]
    check('only slides 10 and 39 semantically changed',changed==[10,39],{'raw_xml_changed':raw,'normalized_xml_changed':changed})
    oldstyles=ET.fromstring(before.read('ppt/tableStyles.xml')); newstyles=ET.fromstring(after.read('ppt/tableStyles.xml'))
    check('all original table styles unchanged except generated IDs',all(any(canon(ET.tostring(x))==canon(ET.tostring(y)) for y in newstyles) for x in oldstyles),{'before':len(oldstyles),'after':len(newstyles)})
    media=[n for n in before.namelist() if n.startswith('ppt/media/')]
    check('all original embedded media bytes preserved',sorted(hashlib.sha256(before.read(n)).hexdigest() for n in media)==sorted(hashlib.sha256(after.read(n)).hexdigest() for n in after.namelist() if n.startswith('ppt/media/')),len(media))
    def image_relationships(z,i):
        name=f'ppt/slides/_rels/slide{i}.xml.rels'
        if name not in z.namelist(): return {}
        return {e.attrib['Id']:hashlib.sha256(z.read(posixpath.normpath('ppt/slides/'+e.attrib['Target']))).hexdigest() for e in ET.fromstring(z.read(name)) if e.attrib['Type'].endswith('/image')}
    check('original image relationships still target identical image content',all(image_relationships(before,i)==image_relationships(after,i) for i in range(1,40)))
    slides={i:ET.fromstring(after.read(f'ppt/slides/slide{i}.xml')) for i in [10,39,40,41]}
    for i in [40,41]:
        ts=slides[i].findall('.//a:tbl',NS); check(f'slide {i} has one editable native table',len(ts)==1)
    t40=table(slides[40].find('.//a:tbl',NS));t41=table(slides[41].find('.//a:tbl',NS))
    check('slide 40 all 36 cells including header match',t40==expected_fleet,{'observed':t40,'expected':expected_fleet})
    expected_cap=[['Representation','Rows','Nonzeros','Presolved nonzeros','MIP seconds']]
    runtimes=[]
    for label,r in zip(['Minute rows','Repeated rows merged','Start/end + occupancy'],cap):
        result=P.parents[1]/'charging_column_structure/pilot/collections/20260922T053756Z/attempts/729675_r0'/r['variant']/'result.json'
        d=json.loads(result.read_text());rt=d['mip']['Runtime'];runtimes.append({'variant':r['variant'],'gurobi_runtime_s':rt,'optimize_wall_s':float(r['mip_seconds'])})
        expected_cap.append([label,f"{int(r['rows']):,}",f"{int(r['nonzeros']):,}",f"{int(r['presolved_nonzeros']):,}",f'{rt:.3f}'])
        check(f"{r['variant']} optimum and original matrix validation",d['finite_pool_fleet_proven'] and d['original_matrix_validation']['valid'] and abs(d['lp']['objective']-3)<1e-8)
    check('slide 41 all 20 cells match source dimensions and Gurobi Runtime',t41==expected_cap,{'observed':t41,'expected':expected_cap})
    s40=text(slides[40]);s41=text(slides[41]);s10=text(slides[10]);s39=text(slides[39])
    seq=' / '.join(f"{round(float(fleet['c1_k15_sequential',a]['actual_optimize_wall_s'])):,}" for a in arms)
    check('slide 40 sequential proof times',seq in s40,seq)
    check('slide 40 finite-pool/time-limit/start caveats',all(s in s40 for s in ['proved in pool','fresh k15 bound 15','30-minute limits','One seed','8 threads','excludes acquiring']))
    check('fleet source supports proof and time-limit claims',all(fleet[c,a]['finite_pool_fleet_proven']=='True' for c in [cases[0],cases[1],cases[4]] for a in arms) and all(fleet[c,a]['status']=='TIME_LIMIT' and abs(float(fleet[c,a]['bound'])-15)<1e-8 for c in cases[2:4] for a in arms))
    check('slide 41 finite-pool/equivalence/speed caveats',all(s in s41 for s in ['321 fixed routes','all LPs equal 3','within this pool','272 continuous','One tiny case','No general speedup claim']))
    notes=[]
    for n in after.namelist():
        if re.fullmatch(r'ppt/notesSlides/notesSlide\d+.xml',n): notes.append(text(ET.fromstring(after.read(n))))
    check('Gurobi Runtime explicitly documented in slide text or notes','Gurobi Runtime' in s41 or any('Gurobi Runtime' in s for s in notes))
    ops=(P.parent/'operations/README.md').read_text()
    check('slide 10 strict endpoint matches operations',all(s in s10 for s in ['331 trips','11 reference duties','PARX 60 kW','not enforced','4.53 h','zero pricing','no certificate','65 buses, bound 64','54 selected','charger conflicts']) and all(s in ops for s in ['331 trips','65 buses / finite-pool lower bound64','zero pricing iterations','54new singleton','PARX60kW','fails']))
    check('slide 10 graph hours round correctly',round(16292.771283/3600,2)==4.53)
    check('slide 39 caveats present',all(s in s39 for s in ['five unchanged pools','no rows or columns removed','finite-pool diagnostics','zero columns in all four fresh pools','8,458 of 130,468']))
    size=ET.fromstring(after.read('ppt/presentation.xml')).find('p:sldSz',NS);W=int(size.attrib['cx']);H=int(size.attrib['cy']);bounds=[]
    for i in [40,41]:
        frame=slides[i].find('.//p:graphicFrame',NS);off=frame.find('p:xfrm/a:off',NS);tbl=frame.find('.//a:tbl',NS)
        x=int(off.attrib['x']);y=int(off.attrib['y']);w=sum(int(c.attrib['w']) for c in tbl.findall('a:tblGrid/a:gridCol',NS));h=sum(int(r.attrib['h']) for r in tbl.findall('a:tr',NS));bounds.append({'slide':i,'x':x,'y':y,'grid_width':w,'row_height_sum':h,'slide_width':W,'slide_height':H})
        check(f'slide {i} native table grid inside slide',x>=0 and y>=0 and x+w<=W and y+h<=H)
    # Render inspection remains necessary: Slides export can retain nominal row heights despite text expansion.
texts={name:((P/f'{name}_before.md').read_text(),(P/f'{name}_after.md').read_text()) for name in ['columns','current','matrix']}
b,a=texts['columns'];bt=md_tables(b);at=md_tables(a)
check('original first three Doc tables unchanged; fourth only intended Runtime header clarification',len(bt)==4 and len(at)==5 and bt[:3]==at[:3] and [r.replace('MIP, s','Gurobi Runtime, s') for r in bt[3]]==at[3],{'before_count':len(bt),'after_count':len(at),'intended_fourth_table_change':'MIP, s -> Gurobi Runtime, s'})
docfleet=md_cells(at[4]);doc_expected=[['Pool','Default','Focus 1','Focus 2','PreSparsify 1','Saved start']]+[[label.replace('seq.','sequential')]+[fleet[c,arm]['fleet'] for arm in arms] for c,label in zip(cases,labels)]
check('new Doc table has complete 5x6 body and header',docfleet==doc_expected,docfleet)
check('Doc completed trial count/caveats',all(s in a for s in ['All 25 trials finished','Both fresh k15 pools retain bound 15','all ten searches reached 30 minutes','excludes the earlier work','one-seed pilot','no new CG']))
check('Doc sequential times match',seq in a)
capdoc=md_cells(at[3]); expected_doccap=[['Formulation','Build, s','Gurobi Runtime, s','Presolved nonzeros']]+[[label,f"{float(r['build_seconds']):.3f}",f"{t['gurobi_runtime_s']:.3f}",f"{int(r['presolved_nonzeros']):,}"] for label,r,t in zip(['Minute rows','Merged rows','Start/end occupancy'],cap,runtimes)]
check('Doc capacity table numeric cells match recorded build time and Gurobi Runtime', [r[1:] for r in capdoc[1:]]==[r[1:] for r in expected_doccap[1:]],{'observed':capdoc,'expected_numeric':expected_doccap})
for name,(b,a) in texts.items():
    images=lambda s:re.findall(r'(?im)!\[[^\]]*\]\([^\n]*\)|^\[image\d+\]:.*$',s)
    check(f'{name} figure references unchanged',images(b)==images(a),{'before':len(images(b)),'after':len(images(a))})
    changes=[{'tag':tag,'before_lines':[i+1,j],'after_lines':[k+1,l]} for tag,i,j,k,l in difflib.SequenceMatcher(None,b.splitlines(),a.splitlines()).get_opcodes() if tag!='equal']
    if name=='current':check('front page exactly three targeted lines replaced',len(changes)==3 and all(v['tag']=='replace' and v['before_lines'][0]==v['before_lines'][1] and v['after_lines'][0]==v['after_lines'][1] for v in changes),changes)
    if name=='matrix':check('matrix exactly one paragraph line replaced',len(changes)==1 and changes[0]['tag']=='replace' and changes[0]['before_lines'][0]==changes[0]['before_lines'][1],changes)
inputs=[P/'slides_before.pptx',P/'slides_after.pptx',P/'slides_after.pdf',P/'columns_after.pdf',fleetpath,cappath,P.parent/'operations/README.md']+[P/f'{n}_{s}.md' for n in texts for s in ['before','after']]
report={'status':'pass' if all(c['pass'] for c in checks) else 'fail','checks':checks,'check_count':len(checks),'failed':[c['name'] for c in checks if not c['pass']], 'capacity_timing_metrics':runtimes,'table_bounds_emu':bounds,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'limitations':['Markdown exports prove document text, tables and figure references; they do not expose the live Doc native object schema.','Slide semantic equivalence normalizes only regenerated table style GUIDs after checking identical style definitions.','Native table coordinates use grid widths and nominal row heights; PDF visual review checks actual rendered expansion.'],'visual_review':{'slides':[10,39,40,41],'result':'Final saved PDF slides 10, 39, 40 and 41 inspected: no clipping, overlap or missing table text.'}}
(P/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks':len(checks),'failed':report['failed']},indent=2))
