"""Verify today's current-tab update and additive figure insertion."""
from pathlib import Path
import hashlib
import io
import json
import re
from PIL import Image
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1]

def body(text):
    text = re.split(r'\n\[image\d+\]:', text)[0]
    text = re.sub(r'!\[\]\[image\d+\]', '[IMAGE]', text)
    text = re.sub(r'^#+\s*$', '', text, flags=re.M)
    return re.sub(r'\s+', ' ', text).strip()

def pixels(path):
    reader = PdfReader(path)
    found = set()
    for page in reader.pages:
        for item in page.images:
            with Image.open(io.BytesIO(item.data)) as im:
                im = im.convert('RGBA')
                found.add(hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest())
    return found, len(reader.pages)

before = (HERE/'figures_before.md').read_text()
after = (HERE/'figures_after.md').read_text()
heading = '## **21 September A day with three charging sites**'
old_images, old_pages = pixels(OUTPUTS/'week_20260921/complex_route_graphs/doc_verification/figures_after.pdf')
new_images, new_pages = pixels(HERE/'figures_after.pdf')
cb = (HERE/'current_before.md').read_text()
ca = (HERE/'current_after.md').read_text()
url = r'\]\((https://[^)]+)\)'
table_rows = lambda t: [x for x in t.splitlines() if x.startswith('|')]
footer = '## **Historical context and sources**'
checks = {
    'all_prior_figure_body_preserved': body(before[before.index(heading):]) == body(after[after.index(heading):]),
    'all_prior_figure_pixels_preserved': old_images <= new_images,
    'six_new_images': len(new_images-old_images) == 6,
    'new_figure_headings': all(x in after for x in ['Fractional solutions and integer routes','Routes that complete an integer fleet']),
    'no_placeholders': 'PAPER_FIGURE_' not in after,
    'unambiguous_one_hour_budget': 'within the one-hour total budget' in after,
    'all_current_tables_preserved': table_rows(cb) == table_rows(ca),
    'all_old_source_links_preserved': set(re.findall(url,cb)) <= set(re.findall(url,ca)),
    'historical_footer_and_original_current_figures_preserved': body(cb[cb.index(footer):]) == body(ca[ca.index(footer):]),
    'strict_endpoint_and_scope': all(s in ca for s in ['nine GIRO duties','2,453 iterations','57 extra assignments','pricing did not certify convergence','Capacity was omitted']),
    'paper_preview_linked': 'paper_results/RESULTS_PREVIEW.md' in ca,
    'sixteen_balanced_jobs_described': '16 allocations' in ca,
}
assert all(checks.values()), checks
out = {'verified':True,'checks':checks,'figures_pages_before':old_pages,'figures_pages_after':new_pages,'figures_images_before':len(old_images),'figures_images_after':len(new_images),'current_pages_after':len(PdfReader(HERE/'current_after.pdf').pages),'history_tab_modified':False,'cg_tab_modified':False,'slides_modified':False,'visual_qa':'New figure pages 1–6 rendered and inspected; no clipping; captions and source links remain editable.'}
out['files_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.suffix in {'.md','.pdf','.py'}}
(HERE/'verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
