# Publication audit

**Passed: 70 of 70 checks.** Final exported Doc content, complete PPTX, source endpoints, native gate receipts and PDF renders were independently audited after the root agent's final-export-ready message. [verification.json](verification.json) records every check, exact changes and source/export hashes. [verify_publication.py](verify_publication.py) reproduces the automated checks. No cluster, Git, live UI or shared-pointer action was taken by this audit.

## Source expectations

The current operations collection is scoped to 22 September 2026 at 15:58:38 UTC. It records 20 running baseline jobs (11 graph preparations, five CGs and four MIPs), 33 of 44 completed graph preparations, and 75 solver dependency waits. Ten cumulative graph preemptions remain historical. The queue is not a new solver certificate.

The newly completed k33 MIPs have integer fleet / finite-pool fleet bounds C1 36 / 33, C2 36 / 32 and C4 38 / 32. Both optimizer stages time out in each case. None proves its fleet optimum or reaches target 33. C3's earlier 34 / 33 outcome remains supported by the previous collection. Native selected-route replay passes, while duplicate cleanup and shared capacity remain unvalidated.

The five new CG endpoints are C1/C3/C4 at k34 and C2/C6 at k33. Each stops at its scientific wall limit with zero artificials and no pricing certificate. The positive-only saved route representation for C2 k33 reconstructs the weighted objective 0.037537633 above the recorded scalar; this discrepancy must remain disclosed in the linked audit. Route weights are distinct from weighted objectives and certified lower bounds.

## Verified publication changes

The existing k33 table now has five rows, C1/C2/C3/C4/C6, with unchanged headers. Each displayed trip count, one-decimal CG time, rounded restricted-master route weight and integer fleet/pool-bound cell agrees with exact current or prior source CSVs. Twelve cited endpoint files also match their recorded SHA256 hashes. C5's graph remains pending in prose, and C6's MIP remains running at the 11:58 EDT snapshot. The four completed MIP gaps remain open; all five table CGs remain uncertified. The Doc and slide notes retain separate k34 endpoint scope, the C2 numerical discrepancy and physical limitations.

Slide 10 and the existing strict paragraph now report native job 768638. The seven collected files match the gate receipt's hashes, its nine receipt checks pass, and its result records 8,343 inherited records equal in order after normalizing only `cg_checkpoint_id`, 54 saved singleton routes and 8,397 successful route replays. The 35-second number is scheduler elapsed time, distinct from 29.075 seconds of gate work and 32.440 seconds of worker time. The earlier 13 local/native cache tests remain a separate result. No graph, solver, new CG certificate or fresh singleton optimum is claimed; full 331-trip graph/reload parity remains required.

## Exact preservation scope

The deck retains 42 slides. Only slides 10 and 42 change semantically; only their notes change. All other slide XML is identical after normalizing table-style identifiers, and the original table style definitions remain present. All media bytes and every slide's image bindings are preserved. Master/layout/notes-master XML is preserved. Google permuted `theme1.xml` and `theme2.xml`; their complete byte hashes and all three resolved presentation/master theme bindings are unchanged. This is export renumbering, not a theme change.

The Doc retains all four front-tab tables, with only the existing k33 table edited. Every other table, figure reference, and text outside the authorized extension-update and strict-gate regions is unchanged. Within the extension, its historical introduction, complete baseline/full40 physics paragraph and geography links are byte-identical. Historical source links and the front-tab footer remain intact.

The root's separate [live UI receipt](live_ui_verification.json) records “Saved to Drive,” seven visible Doc tabs, no edits to other tabs, and no edits to historical decks. The audit verifies that receipt and its scope; it does not claim independent content hashes of unexported tabs or native Doc schema from Markdown. The slide table is independently verified as native editable OOXML.

## Visual review

Bundled `pdftoppm` rendered the final PDFs at 120 dpi. Doc pages 2, 3, 6, 7 and 8, and slides 10 and 42 were individually opened and inspected. The entire five-row k33 table and its scope footnote fit Doc page 2 and remain readable. The MIP/k34/numerical scope on page 3 and the strict-gate paragraph on pages 6–7 are legible without clipping or overlap. Slide 42's table and caption, and slide 10's final qualification, fit their canvases. The unchanged historical k5 table still continues at the same row boundary across pages 7–8 as the previous export.

The [visual review record](audit_renders/visual_review.json) binds these observations to the exact final PDF and rendered PNG hashes. Rendered PNGs are private QA evidence, not new published artifacts.
