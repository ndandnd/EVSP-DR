#!/usr/bin/env python3
"""Independent saved-export audit; standard library only, no UI or cluster calls."""
from pathlib import Path
import csv, difflib, hashlib, json, posixpath, re, xml.etree.ElementTree as E, zipfile
P=Path(__file__).resolve().parent
NS={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
checks=[]
def ck(name,ok,detail=None):checks.append({'name':name,'pass':bool(ok),'detail':detail})
def canon(data):
 x=E.fromstring(data)
 for e in x.iter():
  if e.tag.endswith('}tableStyleId'):e.text='STYLE_ID'
  for k in ['styleId','def']:
   if k in e.attrib:e.attrib[k]='STYLE_ID'
 return E.tostring(x)
def txt(x):return '\n'.join(e.text or '' for e in x.findall('.//a:t',NS))
def cells(x):return [['\n'.join(e.text or '' for e in c.findall('.//a:t',NS)) for c in r.findall('a:tc',NS)] for r in x.findall('a:tr',NS)]
def tables(s):return [m.group(0).strip().splitlines() for m in re.finditer(r'(?m)^\|.*(?:\n\|.*)*',s)]
def md_cells(t):return [[c.strip() for c in r.strip('|').split('|')] for r in t if not re.fullmatch(r'[| :\-]+',r)]
def image_refs(s):return re.findall(r'(?im)!\[[^\]]*\]\([^\n]*\)|^\[image\d+\]:.*$',s)
def image_rels(z,i):
 n=f'ppt/slides/_rels/slide{i}.xml.rels'
 if n not in z.namelist():return {}
 return {e.attrib['Id']:hashlib.sha256(z.read(posixpath.normpath('ppt/slides/'+e.attrib['Target']))).hexdigest() for e in E.fromstring(z.read(n)) if e.attrib['Type'].endswith('/image')}
cg=list(csv.DictReader((P.parent/'operations/cg_endpoints.csv').open()));mip=list(csv.DictReader((P.parent/'operations/mip_endpoint.csv').open()))[0]
expected_source=[{'chain':'C'+r['case'][1],'trips':r['trips'],'cg_minutes':f"{float(r['reported_cg_wall_s'])/60:.1f}",'route_weight':str(round(float(r['route_weight'])))} for r in cg]
ck('CG source endpoints uncertified wall limits',all(r['pricing_certified']=='False' and r['stop_reason']=='wall_limit' for r in cg))
ck('C3 MIP source is 34/33 without fleet proof or target',mip['buses']=='34' and abs(float(mip['finite_pool_bound'])-33)<1e-8 and mip['fleet_proven']=='False' and mip['target_attained']=='False')
with zipfile.ZipFile(P/'slides_before.pptx') as before,zipfile.ZipFile(P/'slides_after.pptx') as after:
 count=lambda z:len([n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+.xml',n)])
 ck('slide counts 41 -> 42',count(before)==41 and count(after)==42)
 raw=[i for i in range(1,42) if before.read(f'ppt/slides/slide{i}.xml')!=after.read(f'ppt/slides/slide{i}.xml')]
 changed=[i for i in range(1,42) if canon(before.read(f'ppt/slides/slide{i}.xml'))!=canon(after.read(f'ppt/slides/slide{i}.xml'))]
 ck('only optional strict slide 10 changed among original slides',changed in [[],[10]],{'raw_xml_changes':raw,'semantic_changes':changed})
 oldstyles=E.fromstring(before.read('ppt/tableStyles.xml'));newstyles=E.fromstring(after.read('ppt/tableStyles.xml'))
 ck('all original table styles preserved',all(any(canon(E.tostring(x))==canon(E.tostring(y)) for y in newstyles) for x in oldstyles))
 media=lambda z:sorted(hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('ppt/media/'))
 ck('all original media bytes preserved',media(before)==media(after))
 ck('all original slide image bindings preserved',all(image_rels(before,i)==image_rels(after,i) for i in range(1,42)))
 slide=E.fromstring(after.read('ppt/slides/slide42.xml'));ts=slide.findall('.//a:tbl',NS)
 ck('new slide 42 contains one editable native table',len(ts)==1)
 slide_table=cells(ts[0]) if ts else []
 s42=txt(slide)
 size=E.fromstring(after.read('ppt/presentation.xml')).find('p:sldSz',NS)
 bounds=[]
 for frame in slide.findall('.//p:graphicFrame',NS):
  t=frame.find('.//a:tbl',NS);off=frame.find('p:xfrm/a:off',NS)
  if t is None or off is None:continue
  x=int(off.attrib['x']);y=int(off.attrib['y']);w=sum(int(e.attrib['w']) for e in t.findall('a:tblGrid/a:gridCol',NS));h=sum(int(e.attrib['h']) for e in t.findall('a:tr',NS));bounds.append([x,y,w,h])
  ck('new native table grid fits slide',x>=0 and y>=0 and x+w<=int(size.attrib['cx']) and y+h<=int(size.attrib['cy']))
 slide10_before=txt(E.fromstring(before.read('ppt/slides/slide10.xml')));slide10_after=txt(E.fromstring(after.read('ppt/slides/slide10.xml')))
b=(P/'current_before.md').read_text();a=(P/'current_after.md').read_text();bt=tables(b);at=tables(a)
ck('all original front-tab table contents preserved',all(t in at for t in bt),{'before_count':len(bt),'after_count':len(at)})
ck('front-tab gains exactly one table',len(at)==len(bt)+1)
ck('front-tab figure references preserved',image_refs(b)==image_refs(a),{'before_count':len(image_refs(b)),'after_count':len(image_refs(a))})
newtables=[t for t in at if t not in bt];doc_table=md_cells(newtables[0]) if len(newtables)==1 else []
changes=[{'tag':tag,'before_range':[i+1,j],'after_range':[k+1,l],'before':b.splitlines()[i:j],'after':a.splitlines()[k:l]} for tag,i,j,k,l in difflib.SequenceMatcher(None,b.splitlines(),a.splitlines()).get_opcodes() if tag!='equal']

expected_slide=[['Chain','Trips','CG minutes','Fractional buses*','Integer buses','Pool bound']]+[[r['chain'],r['trips'],r['cg_minutes'],r['route_weight'],'34' if r['chain']=='C3' else 'MIP running','33' if r['chain']=='C3' else '—'] for r in expected_source]
expected_doc=[['Chain, k=33','Trips','CG minutes','Fractional buses\\*','Integer buses / pool bound']]+[[r['chain'],r['trips'],r['cg_minutes'],r['route_weight'],'34 / 33 — open' if r['chain']=='C3' else 'MIP running'] for r in expected_source]
ck('slide 42 all 24 table cells match endpoint sources',slide_table==expected_slide,expected_slide)
ck('Doc all 20 new table cells match endpoint sources',doc_table==expected_doc,expected_doc)
ck('exactly original slide 10 changed semantically',changed==[10])
ck('slide 10 previous evidence preserved except final sentence',slide10_before.split('Duplicates and charger conflicts remain.')[0]==slide10_after.split('Duplicates and charger conflicts remain.')[0])
ck('front only extension paragraph and strict sentence region changed',len(changes)==2 and changes[0]['before_range']==[29,29] and changes[1]['before_range']==[78,78])
old_strict=changes[1]['before'][0];new_strict=changes[1]['after'][0]
old_sentence='The next repair is to prepare and reuse the graph before giving CG its own recorded allowance. '
ck('strict historical facts and links unchanged around new sentence',old_strict.split(old_sentence)[0]==new_strict.split('The graph-save/reload fix')[0] and old_strict.split('[Audited MIP endpoint')[1]==new_strict.split('[Audited MIP endpoint')[1])
ops=(P.parent/'operations/README.md').read_text()
ck('running MIPs remain scoped to 07:57 EDT snapshot','snapshot 22 Sep, 07:57 EDT' in s42 and '07:57 EDT' in a and 'Running C1/C4 MIPs' in ops)
ck('no pricing certificate or full-model LP-bound claim',all(x in s42 for x in ['no pricing certificate','not LP lower bounds']) and all(x in a for x in ['final restricted-master route weight','without a pricing certificate','not certified LP lower bounds']))
ck('C3 MIP timeout and physical scope caveats',all(x in s42 for x in ['both MIP stages timed out','118 extra trip assignments','shared capacity unvalidated']) and all(x in a for x in ['has not matched 33 buses','118 extra assignments across 95 trips','Shared capacity and duplicate removal remain unvalidated']))
builds=[float(r['graph_original_build_s'])/3600 for r in cg];graph_range=f'{min(builds):.2f}–{max(builds):.2f} h'
ck('graph preparation range separated from CG timing',graph_range in s42 and graph_range in a and 'CG minutes exclude earlier smaller instances' in a,graph_range)
ck('queue counts agree with scoped operations',all(x in a for x in ['26/44 graphs','25 jobs run','18 graphs, five CGs and two MIPs','85 pending','Ten cumulative']))
strictroot=P.parents[1]/'strict_graph_reuse';testdir=strictroot/'native_preflight/collections/20260922T121421Z/attempts/741034_r0'
testresult=json.loads((testdir/'result.json').read_text());testlog=(testdir/'tests.log').read_text();localtest=(strictroot/'cache_tests_timed.log').read_text()
ck('13 native tests passed without solver or full graph',testresult['status']=='passed' and testresult['test_count']==13 and testresult['solver_started']==False and testresult['source_files_verified']==True and 'no full k19 graph or CG' in testresult['scope'] and 'Ran 13 tests' in testlog and testlog.rstrip().endswith('OK'))
ck('native test log hash matches receipt',hashlib.sha256((testdir/'tests.log').read_bytes()).hexdigest()==testresult['log_sha256'])
ck('13 local tests passed','Ran 13 tests' in localtest and '\nOK' in localtest)
ck('cache publication retains full-size/CG limitations',all(x in a for x in ['13 local tests','all 13 tests','job 741034','No full 331-trip graph or new CG has run','full-size graph before restarting k19']) and 'full-size graph and pool checks are next' in slide10_after)

report={'status':'pass' if all(c['pass'] for c in checks) else 'fail','check_count':len(checks),'checks':checks,'failed':[c['name'] for c in checks if not c['pass']],'source_expected':expected_source,'slide42_table':slide_table,'doc_new_table':doc_table,'front_diff':changes,'slide10_before':slide10_before,'slide10_after':slide10_after,'slide42_text':s42,'table_bounds_emu':bounds,'limitations':['OOXML native tables are verified; Markdown exports verify document content, not live native Doc schema.','Only regenerated table-style GUIDs are normalized, with original style definitions separately checked.','Nominal table bounds do not replace rendered PDF inspection.']}
inputs=[P/n for n in ['current_before.md','current_after.md','slides_before.pptx','slides_after.pptx','slides_after.pdf','current_after.pdf']]+[P.parent/'operations'/n for n in ['cg_endpoints.csv','mip_endpoint.csv','README.md']]
inputs += [testdir/'result.json',testdir/'tests.log',strictroot/'cache_tests_timed.log']
report['visual_review']={'slides':[10,42],'doc_pages':[2,3,6,7,8],'result':'Final PDF renders inspected; new table and edited scope text are legible, with no clipping or overlapping elements.'}
report['source_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs if p.exists()}
(P/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks':len(checks),'failed':report['failed']},indent=2))
