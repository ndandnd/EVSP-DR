from pathlib import Path
import json,hashlib,io,re,difflib,subprocess
from pptx import Presentation
from PIL import Image,ImageChops,ImageDraw,ImageFilter
import numpy as np
from pypdf import PdfReader
P=Path(__file__).resolve().parent;FIG=P.parents[1]/'chain_comparison_mip_times';report={'checks':{},'slides':{},'documents':{},'issues':[]}
sha=lambda b:hashlib.sha256(b).hexdigest()
def pixels(b):
 im=Image.open(io.BytesIO(b)).convert('RGB');return {'size':im.size,'rgb_sha256':sha(im.tobytes())}
def normalized(im):return np.asarray(im.convert('RGB').resize((512,244),Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(1)),dtype=float)
def identify(im):
 a=normalized(im);scores=[float(np.sqrt(((a-normalized(Image.open(FIG/f'chain{i}_charts.png')))**2).mean())) for i in range(1,7)];order=sorted(range(6),key=lambda k:scores[k]);return dict(best_chain=order[0]+1,rmse_by_chain=scores,verified=scores[order[0]]<1.5 and scores[order[1]]>5.0)
def slide_data(sl):
 text=[];pictures=[];tables=[]
 for sp in sl.shapes:
  if sp.has_text_frame:text.append(sp.text)
  if sp.shape_type==13:pictures.append(pixels(sp.image.blob))
  if sp.has_table:tables.append([[c.text for c in r.cells] for r in sp.table.rows])
 return dict(text=text,pictures=pictures,tables=tables,geometry=[(sp.left,sp.top,sp.width,sp.height) for sp in sl.shapes])
before=Presentation(P/'slides_before.pptx');after=Presentation(P/'slides_after.pptx');report['checks']['28_slides']=len(after.slides)==28;report['slides']['before_count']=len(before.slides);report['slides']['after_count']=len(after.slides)
original=[]
for i in range(14):
 a,b=slide_data(before.slides[i]),slide_data(after.slides[i]);original.append(dict(slide=i+1,text_equal=a['text']==b['text'],images_equal=a['pictures']==b['pictures'],tables_equal=a['tables']==b['tables'],geometry_equal=a['geometry']==b['geometry']))
report['slides']['original_preservation']=original;report['checks']['original_1_to_14_preserved_except2']=all(all(r[k] for k in ['text_equal','images_equal','tables_equal','geometry_equal']) for r in original if r['slide']!=2)
report['slides']['slide2_before']=slide_data(before.slides[1]);report['slides']['slide2_after']=slide_data(after.slides[1])
imgchecks=[]
for chain in range(1,7):
 expected=(FIG/f'chain{chain}_charts.png').read_bytes();sl=after.slides[14+chain];photos=[sp.image.blob for sp in sl.shapes if sp.shape_type==13];imgchecks.append(dict(chain=chain,slide=15+chain,expected_sha256=sha(expected),image_count=len(photos),byte_match=any(sha(b)==sha(expected) for b in photos),pixel_match=any(pixels(b)==pixels(expected) for b in photos)))
for x in imgchecks:
 blobs=[sp.image.blob for sp in after.slides[x['slide']-1].shapes if sp.shape_type==13];x['resampled_identification']=[identify(Image.open(io.BytesIO(b))) for b in blobs]
report['slides']['chain_images']=imgchecks;report['checks']['six_correct_chain_images']=all(any(y['verified'] and y['best_chain']==x['chain'] for y in x['resampled_identification']) for x in imgchecks)
math=[]
for n in range(24,27):
 d=slide_data(after.slides[n-1]);math.append(dict(slide=n,text_frames=len(d['text']),text=d['text'],pictures=len(d['pictures'])))
report['slides']['editable_math']=math;report['checks']['math24_to26_editable']=all(x['text_frames']>=2 and x['pictures']==0 for x in math)
tab=slide_data(after.slides[26]);report['slides']['slide27_tables']=tab['tables'];report['checks']['slide27_editable_table']=len(tab['tables'])>=1
notes=[]
for n in range(15,29):
 sl=after.slides[n-1];text=sl.notes_slide.notes_text_frame.text if sl.has_notes_slide else '';urls=re.findall(r'https?://[^\s]+',text);notes.append(dict(slide=n,text=text,urls=urls))
report['slides']['notes']=notes;report['checks']['notes15_to28_source_urls']=all(r['urls'] for r in notes)
slidepdf=PdfReader(P/'slides_after.pdf');doc=PdfReader(P/'followup_after.pdf');report['checks']['slides_pdf28']=len(slidepdf.pages)==28;report['checks']['followup_pdf6']=len(doc.pages)==6
md=(P/'followup_after.md').read_text();tableheads=re.findall(r'^\|[^\n]+\n\|\s*:?-',md,re.M);report['documents']['new_doc_tables']=len(tableheads);report['checks']['new_doc3tables']=len(tableheads)==3
report['documents']['new_doc_markdown_images']=len(re.findall(r'!\[',md));report['checks']['new_doc6_markdown_images']=len(re.findall(r'!\[',md))==6
full='\n'.join(pg.extract_text() or '' for pg in doc.pages);placeholders=[x for x in ['TODO','TBD','PLACEHOLDER','INSERT IMAGE','Lorem ipsum'] if x.lower() in full.lower()];report['documents']['placeholder_hits']=placeholders;report['checks']['no_placeholder_text']=not placeholders
# Compare decoded embedded Doc PDF images against original chart pixels (PDF may re-encode losslessly).
docimgs=[]
for n,pg in enumerate(doc.pages,1):
 for image in pg.images:
  im=image.image.convert('RGB');px={'size':im.size,'rgb_sha256':sha(im.tobytes())};matches=[i for i in range(1,7) if px==pixels((FIG/f'chain{i}_charts.png').read_bytes())];docimgs.append(dict(page=n,name=image.name,size=im.size,matching_chains=matches,rgb_sha256=px['rgb_sha256'],resampled_identification=identify(im)))
report['documents']['embedded_images']=docimgs;report['checks']['new_doc6_correct_resampled_images']=all(r['resampled_identification']['verified'] for r in docimgs) and [r['resampled_identification']['best_chain'] for r in docimgs]==list(range(1,7))
a=(P/'current_before.md').read_text();b=(P/'current_after.md').read_text()
# line-level diff, strict one replaced paragraph/line.
sm=difflib.SequenceMatcher(None,a.splitlines(),b.splitlines(),autojunk=False);changes=[(t,i,j,k,l) for t,i,j,k,l in sm.get_opcodes() if t!='equal'];report['documents']['current_changed_line_blocks']=changes;report['checks']['current_exactly_one_paragraph_updated']=len(changes)==1 and changes[0][0]=='replace' and changes[0][2]-changes[0][1]==1 and changes[0][4]-changes[0][3]==1
report['documents']['current_diff']='\n'.join(difflib.unified_diff(a.splitlines(),b.splitlines(),fromfile='current_before.md',tofile='current_after.md'))
oldfooter=a[a.rfind('## Source'): ] if '## Source' in a else a.splitlines()[-1];report['checks']['current_source_footer_preserved']=oldfooter in b
# Current PDF embedded image content multiset preserved.
def pdfpixels(p):return sorted((im.image.size,sha(im.image.convert('RGB').tobytes())) for pg in PdfReader(p).pages for im in pg.images)
report['checks']['current_pdf_old_images_preserved']=pdfpixels(P/'current_before.pdf')==pdfpixels(P/'current_after.pdf')
report['checks']['old_slide_images_preserved']=all(r['images_equal'] for r in original)
report['source_hashes']={n:sha((P/n).read_bytes()) for n in ['slides_before.pptx','slides_after.pptx','slides_before.pdf','slides_after.pdf','current_before.md','current_after.md','current_before.pdf','current_after.pdf','followup_after.md','followup_after.pdf']}
report['image_resampling_note']='Published charts are 2048x977 rather than source2730x1302. Exact hashes differ. Each uniquely matches its expected source by normalized pixel RMSE<1.5 versus >5 for all alternatives; no byte identity claim.'
report['visual_review']={'slides_pdf_pages':list(range(15,29)),'followup_doc_pages':list(range(1,7)),'all_rendered_with':'bundled pdftoppm,1500pixel longest dimension','inspection':'No clipping, missing images, overlapping text or placeholders found in the new slides and all six document pages.'}
for name,ok in report['checks'].items():
 if not ok:report['issues'].append(name)
(P/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checks':report['checks'],'issues':report['issues'],'doc_images':docimgs},indent=2))
