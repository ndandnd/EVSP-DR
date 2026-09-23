# Independent final publication QA

**PASS: 38/38 automated checks and visual inspection of five rendered pages/slides.** Final byte hashes are in `doc_verification.json` and `slides_verification.json`; their hashes and the visual receipt are in `verification.json`.

Doc: 19/19 checks pass. Only the approved status, larger-endpoint and strict-recovery regions changed, plus the new source link. All other text, native table rows and existing links are preserved. Facts match the252-check operations audit and15-check strict startup audit. The timestamp alone is bold, and job824877's original identity/resources are restored before the running-status update. No final strict result or certificate is implied.

Slides: 19/19 checks pass. The42-slide deck changes only slides10/42 and their notes. The entire original speaker-note bodies remain exact prefixes before the new updates. All images and their per-slide references are preserved; the editable six-chain k33table is unchanged. Generated export GUIDs are normalized only after checking actual table-style definitions. Final notes are audited; `slides_after_initial.pptx` remains an initial draft, not the final deliverable.

Rendered Doc pages2/3/7 and slides10/42 are readable, with no clipping, split words or overlaps. The native table and caption retain separation. Initial unintended full-paragraph bolding and the lost strict-job sentence were reported to the parent and repaired; **the verifiers passed without changing the expected predicates**. Initial before/draft files were not modified or deleted by this review.

Reproduce with the bundled Python executable:

```sh
/Users/nadan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 /Users/nadan/Documents/projects/demandresponse/outputs/research_management_20260922/monitor_20260922T235958Z/publication/verification/verify_doc.py
/Users/nadan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 /Users/nadan/Documents/projects/demandresponse/outputs/research_management_20260922/monitor_20260922T235958Z/publication/verification/verify_slides.py
```

The scripts inspect local Markdown, PPTX ZIP/XML and PDFs using pypdf. Visual findings are independent reviewer observations. PNGs in `rendered/` are local QA intermediates; exclude them from Git publication. This review made no live edits, UI actions, cluster calls or submissions.
