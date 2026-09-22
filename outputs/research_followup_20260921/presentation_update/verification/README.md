# Presentation and document verification

Read-only QA of the exported presentation and documents. No authored artifact or live UI was modified. Final source hashes and detailed evidence are recorded in `verification.json`; `verify_exports.py` reproduces structural and image checks.

## Checks passed

- Presentation has 28 slides. Original slides 1–14 preserve text, images, tables, and shape geometry, except the authorized summary update on slide 2. All original slide images remain unchanged.
- Slides 16–21 contain the correct chain 1–6 charts in order. Slides 24–26 use native editable text; slide 27 contains a native editable table. Every added slide 15–28 has source URLs in its notes.
- The follow-up document has six pages, three Markdown/native tables, and six correct charts. No placeholder text was found.
- The current document differs by exactly one replaced paragraph. Its source footer and all embedded figure pixels are preserved.

## Image comparison and visual review

The export resamples chart images from 2730×1302 to 2048×977, so embedded image bytes and pixels are not identical to the source PNGs. Each image was uniquely identified against all six source charts after normalization to 512×244 and a one-pixel Gaussian blur. Correct-source RGB RMSE is approximately 0.80–0.82; every incorrect-source comparison exceeds 8.57. The complete comparison matrix is in the JSON receipt. This verifies chart identity without claiming byte identity.

All added presentation pages 15–28 and all six follow-up document pages were rendered with bundled `pdftoppm` and inspected using the saved `review-slide-*`, `review-contact-*`, and `review-doc-*` PNGs. No clipped content, overlapping text, missing charts, or visible placeholders were found. The charts and native table are legible at the reviewed scale.

The receipt assesses the supplied exported artifacts and their correspondence with the pinned chart sources. It does not independently re-prove the scientific results summarized by the presentation.

Final receipt was refreshed after the two wording corrections on slides 25–26 to “positive reduced-cost”. Both pages were re-rendered and visually re-inspected; the corrected sentences fit cleanly. All 16 automated checks pass with no reported issue.
