# Research update — 15 September, 14:26 EDT

**Chain4 now matches target28.** The longer search used the exact original221,624-column pool. It found and proved28 after101.39minutes of fleet search. The original one-hour solve returned29. Individual-route replay passes; charging optimality, duplicate-removal validation and shared capacity are not implied.

| Case | Target | Buses found | Saved-pool fleet bound | Result |
|---|---:|---:|---:|---|
| C4, longer search | 28 | 28 | 28 | Minimum fleet proved within this pool |
| C5, longer search | 28 | 31 | 27 | Improved from34; target and fleet gap unresolved |
| C1, original one-hour MIP | 28 | 31 | 28 | Target and fleet gap unresolved |

Both longer runs used210minutes total, up to180minutes for fleet and the remainder for charging. C5's fleet stage reached180.22minutes. Matching a target in a saved pool is not a full-model optimality certificate. [Unchanged-pool validation and timings](longer_gap_results.csv) · [Original chain results](all_chain_extension_results.csv).

Largest observed target matches including separate longer searches: **27,28,28,28,26,28** for chains1–6. All six reach26; four reach28. Original one-hour maxima remain26,28,27,27,26,28. The short Doc has only its date, headline and C4cell updated; two tables and all figure/history tabs are preserved.

## New useful work

One new longer original-pool search was launched for C1k28: job236276, continuation_gaps4_20260915. It tests the31/bound28 gap with the same validated native worker, solver, greedy initialization and pool. Budgets12600total/10800fleet,8CPU24GB4.5hours/default/requeue/excludedcompute01. It is a fresh search tree, not a saved-tree resume. Full-size license passed and one restart-safe allocation is registered. Source hashes, duplicate checks and launch receipt are in ../../continuation_gaps4_20260915/README.md. No extra time-only repetition was launched for C5, which already received this budget.

The campaign was registered after the main scan began. Its focused collection and in-memory normalization are saved separately; the canonical main snapshot is unchanged and next full scan includes the new root. Collector metadata additions preserve existing result classification. The current main register3349records/78groups contains78CG/78MIP original endpoints among102cases.

C5k29graph also finished and releasedCG224645; C3k29CGcontinues. Remaining22graphs and the newC1MIPgive25running jobs;46wait on true dependencies. No failed/preempted/unsatisfiable job appeared.982allocation records retained. No cancellation, dependency bypass, held historical or EVSPV2G change. SSHworks.
