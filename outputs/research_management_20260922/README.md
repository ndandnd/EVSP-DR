# Research work — 22 September 2026

## Charging capacity representation: completed pilot

The same saved strict-physics k3 pool (35 trips, 321 routes) was solved three ways. All three LPs have objective 3 and all three MIPs prove three buses within this pool. Every selected solution satisfies the original trip and one-minute capacity matrix. No new routes or new physical validation are claimed.

| Formulation | Rows | Variables | Nonzeros | MIP optimizer runtime (s) |
|---|---:|---:|---:|---:|
| Original minute capacity | 1,789 | 321 binary | 44,844 | 0.338 |
| Merge repeated capacity rows | 303 | 321 binary | 10,232 | 0.058 |
| Start/end occupancy equations | 307 | 321 binary + 272 continuous | 7,714 | 0.099 |

This verifies equivalence and reduces input size. It is one tiny, fixed-order pilot, not a general speedup estimate. Presolve already reduced the original to 5,009 nonzeros, compared with 5,596 and 6,779 for the alternatives. Native optimizer runtime differs slightly from wrapper wall time; both are retained.

[Explanation, exact algebra, input hashes, tests and full logs](charging_column_structure/README.md). [Completed pilot table](charging_column_structure/pilot/results.csv).

## Larger MIP tests: submitted, results tracked separately

Five original pools each receive five fleet searches: default, MIPFocus 1, MIPFocus 2, PreSparsify 1 and a saved same-pool incumbent diagnostic. Each optimizer gets 1,800 seconds, with identical seed, threads, incidence and column order. Separate diagnostics check exact dominance and conservative dual-based fixing. Preparation identities must reproduce the original native physical gate before the case's five independent trials start.

[Current campaign status, logs and any recovery receipts](mip_structure/README.md). The saved-start diagnostic excludes its earlier acquisition cost. This is a pilot, not a seed-robust or end-to-end speed comparison. The original and recovered preparation attempts remain distinguishable.

Early verified endpoints: all five C4 k8 arms prove nine buses within the unchanged pool. Sequential C1 k15 supplied with its saved 15-bus incumbent proves 15 in 11.15 optimizer-call seconds; finding that supplied solution is excluded. The loading-only harness defect is repaired and covered by 13 tests; canceled pre-optimization attempts remain recorded as setup cost. Consult the campaign's current recovery map, not the original job range.

## Operations and publication

[Verified queue and endpoints](operations/README.md): initial 01:22 EDT snapshot had 41 running EVSP–DR jobs. Two supplemental k15 MIPs completed at 18/bound15 and 17/bound15; neither reaches the target. Strict parent-prefix k19 exhausted its included CG budget in graph construction before any pricing. Graph reuse with separately recorded preparation time is the next repair, not an automatic k20 launch.

The existing four-hour overnight heartbeat was updated in place to track these experiments, recover useful failures, preserve real dependencies and remain quiet on unchanged state. Default-partition trials exclude the GPU node; historical holds and other projects remain untouched.

[Live Doc: route columns and tests](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ikbgt85cdszz) contains editable example and result tables. [Existing fleet times and matrix sparsity](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.lt33xg84cn65) remains intact. The current journal has two targeted endpoint updates; figures/history are preserved. No Slides were changed in this task under the current standing instruction.

[Document verification and before/after exports](publication/doc_verification.json). Local PDF renders are retained for visual QA; hashes are published instead of redundant image/PDF copies.
