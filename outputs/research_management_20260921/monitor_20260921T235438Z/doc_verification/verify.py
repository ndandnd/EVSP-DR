from pathlib import Path
import re,io,json,hashlib
from PIL import Image
from pypdf import PdfReader
p=Path(__file__).resolve().parent
b=(p/'current_before.md').read_text();a=(p/'current_after.md').read_text()
def body(t):return re.split(r'\n\[image\d+\]:',t)[0]
def unchanged(t):
 blocks=body(t).split('\n\n')
 return [re.sub(r'\s+',' ',x).strip() for x in blocks if x.strip() and not any(y in x for y in ['The pricing pilot recovered eight buses','The packed recovery is complete.','Completed graph benchmark and replay checks'])]
def imgs(f):
 r=PdfReader(f);h=set()
 for page in r.pages:
  for item in page.images:
   im=Image.open(io.BytesIO(item.data)).convert('RGBA');h.add(hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest())
 return h,len(r.pages)
bi,bp=imgs(p/'current_before.pdf');ai,ap=imgs(p/'current_after.pdf')
checks={'only_three_targeted_paragraphs_changed':unchanged(b)==unchanged(a),'all_old_links_preserved':set(re.findall(r'\]\((https://[^)]+)\)',b))<=set(re.findall(r'\]\((https://[^)]+)\)',a)),'all_tables_preserved':[x for x in b.splitlines() if x.startswith('|')]==[x for x in a.splitlines() if x.startswith('|')],'current_images_identical':bi==ai,'old_incomplete_status_removed':'Twelve allocations remain running; the paired comparison is incomplete' not in a,'complete_replication_and_scope':all(x in a for x in ['7/8 treatment runs','0/8 controls','two seeds each','One validates exactly-once','4.71 seconds']),'strict_endpoint_correctly_qualified':all(x in a for x in ['5,236 iterations','without a pricing certificate','1,000,391.890434']),'fixed_dual_scope':all(x in a for x in ['53- and 90-trip','not end-to-end CG speedups','shared capacity is omitted'])}
r={'verified':all(checks.values()),'checks':checks,'pages_before':bp,'pages_after':ap,'current_image_count':len(ai),'figure_history_cg_tabs_edited':False,'slides_edited':False}
(p/'verification.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));assert r['verified']
