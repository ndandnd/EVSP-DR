# Fleet timing tables and final-MIP structure — 22 September 2026

Published to the live Google Doc and weekly deck, then exported and checked. The previous turn had saved the timing tables locally but had not added them to the live artifacts; this update closes that gap.

- [Doc: Fleet times & matrix](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.lt33xg84cn65): five pages, six native per-chain tables, per-chain and combined summaries, eight-row matrix table, definitions and official Gurobi references. The current-research front page links directly here. Existing figures, CG curves and history were retained.
- [Weekly Slides: first new table, slide 29](https://docs.google.com/presentation/d/1F6udjgkiPH51vMUcku7ZT3PhZPAxsQwUCi3TK9vFRDM/edit?slide=id.g3fbd02b91a9_8_14): slides 29–34 show Chains 1–6; 35 shows per-chain means; 36 the combined Chains 2–6 summary; 37 matrix sizes; 38 timing definitions; 39 structure and next tests. Tables and text are native editable objects. Original 28 slides are unchanged. Historical decks were not edited.

Fleet-search time is the first optimizer call of the final two-stage MIP, minimizing bus count over its saved route pool. It excludes CG pricing, master LPs, setup, charging optimization and ancestor MIPs. Stage two constrains fleet <= the best validated first-stage fleet and minimizes electricity plus start fees with the remaining budget. These timings are to optimizer termination, not first target discovery. Means include time limits and are not mean time-to-optimality.

## Evidence and verification

- [Timing source](../chain_comparison_mip_times/fleet_search_tables.md): all 48 endpoint timers and source hashes, original campaign only.
- [Matrix audit](../mip_matrix_audit/README.md): all 48 original/presolved stage matrices, 577 consistency checks, full Gurobi 12.0.3 logs and exact execution source. Published at `9ea683dc28c2afa85ed30e195ae5adc97c51679c`.
- [Structure recommendations](../mip_matrix_audit/gurobi_structure_notes.md): official Gurobi links, current implementation and mathematical reduction prerequisites. No proposed setting was benchmarked or activated.
- [Slides verification](slides_data_verification.json): original slides preserved; every new native table matches source data; native shapes stay within bounds.
- [Doc verification](doc_verification.json): timing and summary cells, exact matrix densities, stage-two dimensions and front-page preservation. Initial heading duplication and one double-rounded percentage were corrected before final delivery.
- Final `doc_tables_after.md` is the published tab text. Current front-page before/after Markdown records its one-link addition.
- PDF/PPTX exports and draft renders are local QA backups, not included in Git to avoid duplicating large slide imagery. Their hashes are recorded in `artifact_hashes.json`. Native Google revision history remains the editable artifact history.

Visual review: all six new timing slides and five summary/method slides inspected; final Doc pages reviewed for complete tables and unclipped text. Final Doc is five pages: definitions + Chains 1–2, Chains 3–6, summaries, matrix, methods. Native table cells verified independently. No solver was run for this task.

## Standing publication rule

New verified research tables/figures and meaningful result changes now update BOTH the current Doc and the current weekly deck automatically, without a new user reminder. [Before/after policy receipt](automatic_publication_policy.md). The existing four-hour heartbeat retains quiet unchanged checks. Preserve historical decks, earlier figures and evidence; do not regenerate artifacts for unchanged results.
