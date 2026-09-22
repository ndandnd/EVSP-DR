# Focused heartbeat — 22 September 2026, 03:54 UTC

Reused the successful 03:43 UTC collection, then inspected only the actionable k15 failures, newly terminal strict successor and completed graph dependencies. SSH is healthy. No broad historical collector, figure rebuild or Slides edit was performed.

## Integer-directed pricing at k15

| Case | Unchanged-pool control: buses / bound | New-pricing treatment | Scope |
|---|---|---|---|
| C1 | 19 / 15, open | Dive stopped on an LP time limit; no final MIP | Supplemental saved-pool recovery authorized after physical validation |
| C3 | 17 / 15, open | **15 / 15, proved in its augmented pool** | Own-dive incumbent accepted; charging gap 10.93%; 12 extra trip assignments; shared capacity unvalidated |
| C5 | 16 / 15, open | Dive stopped on a numerical row-feasibility check; no final MIP | Supplemental saved-pool recovery authorized after physical validation |

These three cases use one seed and the registered nominal 7,200-second dive-wall-plus-MIP allowance, under historical 240 kWh/240 kW, zero-reserve, flat-tariff, fee-5 covering physics. They are separate from the completed k8 replication. Do not count the two software stops as completed scientific misses or calculate a completed three-pair hit rate.

[Full logs, original result payloads and exact failures](../operations/finalcheck_20260922/README.md). Root independently rechecked every indexed proof line against the four full Gurobi logs and their SHA-256 hashes. C3's fleet proof is at line 63; charging remains open. No global pricing certificate or full-GIRO dispatch claim follows from this pool proof.

## Recover useful saved work

[Immutable-pool preflight and exact recovery plan](../operations/k15_salvage_preflight_20260922/README.md). All eight original/augmented source-file hashes match; both augmented journals preserve the original journal prefix. C1/C5 contain 17,859/19,943 appended complete diving records and no exported incumbent. No failed-node LP solution, dual or fixing is transferred.

The remaining final-MIP allowances are **1,852 seconds for C1** and **3,266 seconds for C5**, each with half initially reserved for fleet search. The compute worker must pass the unchanged native full-pool physical gate with zero rejected columns, record all deterministic mappings/repairs and ordered-pool hashes, then invoke the original pinned MIP with original seed/settings. Original failed attempts remain immutable; this is separately labelled recovery, not a replacement paired trial. Any preemption requires prior solver-work accounting before another attempt.

Submitted C1 **728184** and C5 **728185** with native physical gates. Submission IDs, exact settings and gate outcomes are recorded in the linked recovery directory. Default partition, 8 CPUs, 32G and 90-minute allocation headroom preserve the scientific solver limits. Automatic requeue is disabled specifically to prevent unaccounted reuse of a residual budget. No additional pricing campaign or graph restart is authorized by this receipt.

## Baseline and strict work

The 44 baseline graph tasks comprise 41 RUNNING and three COMPLETED at the scoped check. Completed C6 k34/k35/k36 caches correctly release their graph dependencies; their CGs still need their true previous-k pools. No broken dependency or utilization recovery is indicated. Strict k19 CG is running; its MIP retains its CG dependency. [Strict k17 MIP audit](strict_k17/README.md): parent prefix17 contains ten subgroup duties. Its final cover uses11buses with bound10, both stages time-limited; charging774.872/bound380.338911/gap50.9159%. It covers277trips with73extra assignments across61trips. Omitted shared capacity fails at7880C (3/1) andJON_A (4/1). Ten independent source/log/count checks pass; final individual-route realization is construction/metadata, not a new independent MIP replay.

Historical held array 537227, separate EVSP–V2G stochastic work and all running source pins are untouched.

## Document update

The current-results tab replaces the stale “all six k15 jobs are running” sentence with the verified C3 endpoint and explicit C1/C5 failure status, plus an evidence link. The strict ten-duty MIP status is also updated to11/bound10 with its source link. Existing headings, tables, embedded images, history and figure tabs remain unchanged. `doc_verification.json` records twelve passing checks; changed PDF pages 4–5 were rendered and inspected. No Slides edit.

## Next triggers

1. Inspect the two recovery gate receipts, then their MIP endpoints; preserve failures, finite-pool proofs and physical scope separately.
2. Collect strict k19 only when it publishes a new endpoint. Do not repeat unchanged k17 or fixed-dual benchmarks.
3. Let baseline CG start when both graph and predecessor pool are ready. Do not bypass genuine previous-k dependencies.
4. Before any reattempt of a preempted recovery, debit measured earlier solver work; do not reset the remaining budget.
