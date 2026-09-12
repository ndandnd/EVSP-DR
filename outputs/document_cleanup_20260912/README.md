# Document cleanup — 12 September 2026

This folder records the local content restructuring for the EVSP–DR current document and figures. The current document is kept compact: verified status and current figures belong in the Current results and Current figures tabs. Superseded plots, dated narratives and obsolete claims belong in the separate historical archive.

## Destinations

- [Current results](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.lumf8xm66fow)
- [Current figures](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ts4vwph3s99i)
- [Historical archive](https://docs.google.com/document/d/1f0orWtM1-_VWAqjj6GnWP78webCOnvoc01x2VQvaA_k/edit)
- [Existing current Slides deck](https://docs.google.com/presentation/d/11bJ-4B5khXtSPwv1sNlvGgme8JCB65jVIT-RSWu3x9E/edit) (nine slides with native editable tables and three plots)
- [Historical Slides copy](https://docs.google.com/presentation/d/1RAzaiZSh7DRf_By32mQXPCcwT1PvPDsk0S2xzxMOnDQ/edit)

The local register and monitoring runbook now point to these destinations and instruct future monitoring to update current tabs in place without appending chronological clutter. The one-time user-directed Slides cleanup on 12 September does not change the future monitoring rule: monitoring must not instruct or perform automatic Slides edits.

## Verified sources

- [`current_content_audit.md`](current_content_audit.md) is the bounded current audit and slide outline.
- [`../post_meeting_20260910/monitor/20260912T041951Z.json`](../post_meeting_20260910/monitor/20260912T041951Z.json), SHA-256 `691f1de415cd15cb7a3a7a64dfd5fdbb9db389da676f440f0c6a210d204c5ffa`, is the latest monitor snapshot used for current status and matched-return charging records.
- [`../overnight_extension_20260912/cluster_snapshot.json`](../overnight_extension_20260912/cluster_snapshot.json), SHA-256 `281cfaa38eddefc60c70397be0cd3b19b24df29d6375a98ece29b7ab42edaed3`, supplies the warm import-phase timings.
- [`../research_register/CURRENT_CHAIN_TABLES.md`](../research_register/CURRENT_CHAIN_TABLES.md) supplies the dated warm/fresh and capacity summaries.
- [`../post_meeting_20260910/terminal_energy/README.md`](../post_meeting_20260910/terminal_energy/README.md) defines the common 280.7833253 kWh terminal-energy comparison and separates expanded-grid objectives, physical replay, and finite-pool proof scope.
- The current research Markdown export used for import-time wording has SHA-256 `9d47e145bfd6fdc99f1d3ddeeef236c480c8cb3ca6d54c6ea164dfaedeb05708`.

## Current figures

- [`charging_current.png`](charging_current.png) shows physical replay bars for fixed-duty and joint charging, with the original GIRO repriced interval as an errorbar. The exact expanded-grid objectives remain in the audit table.
- [`charging_current_slide.png`](charging_current_slide.png) is the wide `figsize=(10.5, 4.3)` slide variant with the same data.
- [`import_time_current.png`](import_time_current.png) is a horizontal stacked chart of inherited full-pool import/validation and other CG work for Chain 3 k8/k10 and Chain 5 k5/k6/k10.
- [`import_time_current_slide.png`](import_time_current_slide.png) is the wide `figsize=(10.5, 4.3)` slide variant with the same data.
- [`generate_current_figures.py`](generate_current_figures.py) reads the two JSON snapshots and writes both figures.
- [`figure_provenance.json`](figure_provenance.json) records source hashes, exact plotted values, generator details and output hashes.
- [`slides_qa.md`](slides_qa.md) records final PDF/PPTX QA: nine slides, five native tables, three image objects, and no concrete layout defects.

The root task edited Google Docs and Slides through their browser interfaces after making archive copies. The document now has two tabs and the deck has nine slides; tables, headings and captions are editable. No cluster jobs were submitted, changed or canceled by this cleanup. Existing evidence and historical source paths remain preserved.

## Final exports and document QA

The saved Google document was exported and visually checked: Current results is three pages, Current figures is three pages. Tables remain native; no table splits, orphan headings, blank pages or clipped figures remain. The deck was independently checked as nine slides, five native tables and three plots.

[Saved current results](exports/current_results.pdf), [saved current figures](exports/current_figures.pdf), and [saved current slides](exports/current_slides.pdf) preserve this edit. Editable DOCX/PPTX and Markdown exports are alongside them; [export hashes](exports/SHA256.json) identify the files. Live results use the 00:20 EDT snapshot, not a new cluster collection. The completed storage-cleanup note links to its separate verified record at commit b1129bf4.
