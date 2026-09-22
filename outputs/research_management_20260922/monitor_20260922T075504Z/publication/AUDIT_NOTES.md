# Independent saved-publication audit

**Pass: 32 checks.** Reproduce with `python3 verify_publication.py`. The final exported files are hashed in `verification.json`. No UI, cluster, optimizer or publication edits were made by this audit.

- Only slides 10 and 39 changed semantically among the original 39. Fourteen other slide XML files only received regenerated table-style GUIDs. Every original style definition remains identical, and all eight original media objects and their slide relationships retain identical content; media filenames were reordered by export.
- New slides 40 and 41 contain editable native DrawingML tables. All 36 fleet-table cells and all 20 capacity-table cells, including headers and labels, match the source data. The latter uses native Gurobi `mip.Runtime`, explicitly stated in the final slide caption; `pilot/results.csv` instead reports optimize wall time. Both measurements are retained and must not be conflated.
- The Doc's first three original tables are unchanged. Its fourth has only the intended header clarification from “MIP, s” to “Gurobi Runtime, s”; numeric cells remain unchanged and match their sources. The new fleet table contains all five data rows and six columns, plus its header. All 25 outcomes, sequential rounded proof times and proof/time-limit/saved-incumbent caveats agree with the recorded results.
- The front Doc export changes exactly three targeted lines. Figure references remain identical. The matrix tab changes exactly one paragraph line. Markdown verifies the exported content and figure references, not the live Doc's internal native-object schema.
- Strict slide 10 matches the scoped operations record: 331 trips, 11 reference duties, graph build 4.53 hours, zero pricing, fleet 65 / finite-pool bound 64, 54 selected singletons and omitted capacity conflicts. It does not assert a full-model bound or dispatch feasibility.
- Native table bounds fit within the slide canvas. PDF renders of slides 10, 39, 40 and final 41, and final Doc page 4, were visually inspected: no clipping, overlap, missing cells or unreadable table text. The fixed-pool and single-case speed limitations remain visible.

Source CSVs, operation record and all before/after exports are read-only. The transient inspection PNGs in `audit_renders/` need not be published.
