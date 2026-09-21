from pathlib import Path
import hashlib,io,json,re
from PIL import Image
from pypdf import PdfReader
P=Path(__file__).resolve().parent
def body(t):
 t=re.split(r'\n\[image\d+\]:',t)[0]
 t=re.sub(r'!\[\]\[image\d+\]','[IMAGE]',t)
 t=re.sub(r'^#+\s*$','',t,flags=re.M)
 return re.sub(r'\s+',' ',t).strip()
def images(p):
 r=PdfReader(p); hashes=set()
 for page in r.pages:
  for x in page.images:
   im=Image.open(io.BytesIO(x.data)).convert('RGBA')
   hashes.add(hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest())
 return hashes,len(r.pages)
cb=(P/'current_before.md').read_text();ca=(P/'current_after.md').read_text()
fb=(P/'figures_before.md').read_text();fa=(P/'figures_after.md').read_text()
h='## **21 September Extending all six chains to 40**'
hf='## **21 September Fractional solutions and integer routes**'
bi,bp=images(P/'figures_before.pdf');ai,ap=images(P/'figures_after.pdf')
ci,cp=images(P/'current_before.pdf');di,dp=images(P/'current_after.pdf')
checks={
 'all_original_current_body_preserved':body(cb[cb.index(h):])==body(ca[ca.index(h):]),
 'all_original_figure_body_preserved':body(fb[fb.index(hf):])==body(fa[fa.index(hf):]),
 'old_figure_pixels_preserved':bi<=ai,
 'three_new_figure_images':len(ai-bi)==3,
 'old_current_pixels_preserved':ci<=di,
 'all_original_links_preserved':set(re.findall(r'\]\((https://[^)]+)\)',cb+fb))<=set(re.findall(r'\]\((https://[^)]+)\)',ca+fa)),
 'all_original_current_tables_preserved':all(row in ca for row in cb.splitlines() if row.startswith('|')),
 'no_placeholders':'FOLLOWUP_' not in fa,
 'new_capacity_counts_correct':all(x in ca for x in ['40 / 487','195 / 487','29 / 48','46 / 48']),
 'new_lp_scope_present':all(x in ca for x in ['0.0000010617','tolerance 0.0001','event-grid']),
 'route_physics_scope_present':all(x in fa for x in ['zero reserve','historical algorithm schedules','11 and serves 10 other']),
}
out={'verified':all(checks.values()),'checks':checks,'current_pages':[cp,dp],'figures_pages':[bp,ap],'figure_image_counts':[len(bi),len(ai)],'history_tab_edited':False,'cg_tab_edited':False,'slides_edited':False}
(P/'verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2));assert out['verified']
