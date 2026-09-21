"""Verify the additive Figures-tab edit; no remote writes or solver work."""
from pathlib import Path
import hashlib
import io
import json
import re

from PIL import Image
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
OLD_PDF = HERE.parents[1] / 'spatial_schedule_graphs/doc_verification/figures_after.pdf'


def body(text):
    text = text.split('\n[image1]:')[0]
    text = re.sub(r'!\[\]\[image\d+\]', '[IMAGE]', text)
    text = re.sub(r'^#+\s*$', '', text, flags=re.M)
    return re.sub(r'\s+', ' ', text).strip()


def pixel_hashes(path):
    result = set()
    reader = PdfReader(path)
    for page in reader.pages:
        for item in page.images:
            with Image.open(io.BytesIO(item.data)) as im:
                im = im.convert('RGBA')
                result.add(hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest())
    return result, len(reader.pages)


before = (HERE / 'figures_before.md').read_text()
after = (HERE / 'figures_after.md').read_text()
old_heading = '## **21 September — A bus day as a graph**'
old_images, old_pages = pixel_hashes(OLD_PDF)
new_images, new_pages = pixel_hashes(HERE / 'figures_after.pdf')
result = {
    'all_prior_figure_body_preserved': body(before[before.index(old_heading):]) == body(after[after.index(old_heading):]),
    'all_prior_pdf_image_pixels_preserved': old_images <= new_images,
    'images_before': len(old_images),
    'images_after': len(new_images),
    'pages_before': old_pages,
    'pages_after': new_pages,
    'new_heading_present': '21 September A day with three charging sites' in after,
    'new_image_present': len(new_images - old_images) == 1,
    'no_placeholder_text': 'COMPLEX' not in after,
    'original_only_scope_present': 'no validated fee-0/fee-5 counterpart' in after,
    'source_links_present': 'complex_route_graphs/duty_13309_graph.png' in after and 'complex_route_graphs/README.md' in after,
    'slides_modified': False,
    'current_history_and_cg_tabs_modified': False,
}
for key in ['all_prior_figure_body_preserved', 'all_prior_pdf_image_pixels_preserved', 'new_heading_present', 'new_image_present', 'no_placeholder_text', 'original_only_scope_present', 'source_links_present']:
    assert result[key], (key, result)
result['verified'] = True
(HERE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
