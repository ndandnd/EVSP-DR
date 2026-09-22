# Completed k8 integer-directed replication — independent audit

Verified against the complete saved results and full Gurobi logs collected with the **21 September 2026 23:54 UTC monitor**: all 16 allocations completed, and **7/8 treatments reached and proved fleet 8 in their saved pools, versus 0/8 controls** (all nine buses). This is four fixed cases with two seeds each, not eight independent instances. C1, C3 and C5 succeeded at both seeds; C4 succeeded only at seed 20260922.

[Editable 16-row results](verified_results.csv) · [Machine-readable checks](audit_summary.json) · [Full solver proof/start line references](proof_log_lines.json) · [Artifact retrieval hashes](retrieval_manifest.json) · [Independent remote provenance checks](remote_provenance_checks.json)

|Case|Seed|Control fleet / bound|Treatment fleet / bound|New treatment columns|Treatment charged s|Treatment actual minutes|Treatment duplicated trips|
|---|---:|---|---|---:|---:|---:|---:|
|c1_k08 |20260921|9 / 8 (open)|8 / 8 (proved)|15,966|2025.93|35.10|7|
|c1_k08 |20260922|9 / 9 (proved)|8 / 8 (proved)|18,255|2043.62|35.41|2|
|c3_k08 |20260921|9 / 8 (open)|8 / 8 (proved)|15,823|1091.19|18.77|1|
|c3_k08 |20260922|9 / 8 (open)|8 / 8 (proved)|12,148|817.53|14.14|1|
|c4_k08 |20260921|9 / 9 (proved)|9 / 8 (open)|23,310|3601.95|61.57|20|
|c4_k08 |20260922|9 / 9 (proved)|8 / 8 (proved)|18,569|2008.61|34.88|0|
|c5_k08 |20260921|9 / 9 (proved)|8 / 8 (proved)|11,169|1008.56|17.74|1|
|c5_k08 |20260922|9 / 9 (proved)|8 / 8 (proved)|14,049|1064.20|18.67|1|

The C4/20260921 treatment is a **completed target miss**, not failed execution: the dive stopped at its wall limit after generating 23,310 columns without an integer incumbent; the subsequent MIP returned nine buses and bound eight. Both final MIP stages hit their limits. No infeasibility or nine-bus optimum follows. Five controls proved their **finite-pool** minimum to be nine (C1 seed 20260922, C4 both seeds, C5 both seeds); the other three controls have bound eight. No control attained eight.

## Budget accounting

The registered common budget is 3,600 seconds of **dive subprocess wall plus MIP solver runtime**. All treatment cache loading, hashing, setup, pricing and publication within the dive subprocess count. Every MIP limit is exactly `floor(3600 − dive_wall)` (3,600 for controls), with stage-one limit equal to half the remaining budget and **no minimum-floor extension**. Arguments, receipts and result runtimes agree in all 16 cells. Seeds and eight-thread requests agree with registration.

Successful treatments used 817.53–2,043.62 charged seconds and 14.14–35.41 actual end-to-end minutes. MIP physical preparation/replay and other subprocess overhead remain separately measured external overhead. Graph construction was a pre-existing shared prerequisite, separately recorded, and was not performed anew.

This was not a hard elapsed-time cap: all eight time-limited controls and the treatment miss slightly exceeded nominal charged time, by **0.68–4.70 seconds**. The treatment miss used 3,601.95 charged seconds and 3,694.41 actual seconds. The largest excess was C4 control seed 20260921 (3,604.70 charged seconds). These observed solver termination overruns are retained without relabeling them as strict 3,600-second runs. Actual end-to-end times include external overhead and are not claimed budget-capped.

## Handoff, source identity and certificate scope

- Every successful treatment exported its **own dive's eight-bus incumbent**, passed the independent MIP route replay, and the full solver log explicitly accepted the objective-eight MIP start. Every export SHA256 matches the execution/result start receipts, with zero new/replacement handoff columns. The generic source field `added_giro_route_count=8` is a reused importer label: the actual source path and 56 route record hashes resolve to the seven own dive journals, not external GIRO witnesses.
- A separate read-only remote audit streamed the seven augmented journals, matched **all 56 exported full records at their recorded ordinals**, and recomputed whole-journal hashes. It also rehashed all four original fresh-pool descriptors/journals and found them unchanged. No large journal, pool or graph was copied locally. Controls read the unchanged frozen fresh pools; treatment sources and augmented descriptors are hash-bound.
- Execution receipts, arguments, source audits and record checks show no external witness route or warm-column input. Warm-named reusable caches contain graph data only; their physical identities, hashes and graph methods were checked. The remote runner, diver and MIP source bytes match execution commit `1be819f0b4e3ea9dc9766497efa169a03e476ad5`.
- All results pass the saved **individual-route continuous replay plus at-least-once trip coverage** checks under historical 240 kWh / 240 kW, zero-reserve, flat-tariff, fee-five, cover-master settings. These are numerical finite-pool fleet proofs; dive-node pricing does not confer a global branch-and-price certificate, full-model fleet optimum, or continuous-electricity optimum.
- Only the C4/20260922 treatment has zero duplicated service and a true `duplicate_trip_removal_validated` flag. The other six successful treatments duplicate 1–7 distinct trips; the sole miss duplicates 20. The CSV separates distinct duplicated trips from extra trip occurrences. **No arm imposes or validates shared charger capacity.** A target-eight result is therefore not automatically a clean operational eight-bus dispatch.

## Evidence and reproducibility

The folder preserves **103 small result/execution/source/handoff artifacts and full dive/MIP Gurobi logs**, 71 newly retrieved and 32 reused from prior local artifacts after exact hash matching. All remote SHA256 values and the available execution-receipt output hashes agree. Five exact-byte root receipts retain registration, jobs, scheduler and collector views. Snapshot JSON text normalized CSV line endings, so root CSV/TSV receipts were fetched as bytes and independently hashed rather than falsely treating normalized text as exact file bytes.

`collect_receipts.py` retrieves only the explicit small inventory; `remote_provenance_audit.py` is the read-only remote hash/ordinal check; `verify_audit.py` reproduces the local checks and editable result table; `write_report.py` regenerates this report. Complete proof lines remain in full `results/**/mip_gurobi.log` files. Original campaign files and earlier four-success snapshots are preserved. No new optimization, submissions, register, live Doc, Slides or push was performed by this audit.

Suggested concise status wording: “The balanced k8 replication is complete: integer-directed columns reached fleet 8 in 7/8 case-seed runs, versus 0/8 controls, using accepted own-dive incumbents and the registered shared-budget accounting. Four cases were each repeated with two seeds. The C4 seed-20260921 miss remains 9/bound8; successful treatment wall times were14.1–35.4minutes. These are saved-pool proofs and individual-route replay results; only one successful witness is duplicate-free, and shared charger capacity remains unchecked.”
