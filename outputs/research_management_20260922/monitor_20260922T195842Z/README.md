# Evening research check — 22 September 2026

The heartbeat label is19:58UTC; the actual baseline collection was **21:31:53UTC (17:31EDT)**. Reuse that collection for90minutes unless a new actionable report warrants a scoped check. SSH remained healthy.

## Verified results

**423/423 operations checks pass.** [Exact CG/MIP tables, native logs and source hashes](operations/README.md). All six target33 chains now have completed MIPs:

| Chain | Fractional RMP route weight | Integer buses | Finite-pool bound |
|---|---:|---:|---:|
|C1|33|36|33|
|C2|32|36|32|
|C3|33|34|33|
|C4|32|38|32|
|C5|32|39|32|
|C6|32|38|32|

Every k33 CG stopped at four hours without a pricing certificate. Every MIP stopped at both time limits. None of these integer endpoints matches33 or proves its pool optimum. These fractional weights are not certified full-model lower bounds. Native selected-route replay passes the baseline; duplicate cleanup and shared capacity remain unvalidated.

New larger MIPs: C1k34=40/bound33,k35=42/34; C2k34=36/33; C3k34=35/34,k35=36/35; C4k34=42/33,k35=42/34; C6k34=40/33. All remain open. Separate graph, CG, preparation, fleet and charging times are in the source tables. Retain the old C2k33 numerical discrepancy, the small C1k34 selected-cost versus solver-objective difference, CGdirty-source flag and distinct memory-accounting scopes.

## Cluster and recovery

**41/44 graphs complete;9baseline jobs running (3graphs,6CG),62true pending solver dependencies.** The running CGs are C1/C3/C4k36,C2/C6k35,C5k34. No failed allocation or broken predecessor required repair. No duplicate work was launched to inflate concurrency. Historical held537227 and EVSP–V2G remain untouched.

**Full strict graph gate772820 passed**, COMPLETED0:0 in3:10:10. Cold build11255.559s; reload15.955s; all8397initialcolumns and both diagnostic pricing routes agree. Scheduler peak6.080GiB; reload-process peak3.411GiB. These are graph/initializer checks, not a CG certificate or an integer result. [Audited receipts and exact cached recovery](strict_review/README.md).

**Submitted one cached strict CG, job824877, at21:43:16UTC.** Last scoped startup observation21:44:12UTC: PENDING(Priority),restart0; no license or CG execution yet. Resources8CPU16GiB5hdefault,requeue,GPUcompute01excluded. Preserve clean modelfedf4214, authentic parent35770aae and the original4hCGallowance; verified graphload precedes that allowance. The16GiB request retains headroom beyond measured reload memory; full master-growth peak remains unmeasured. Fresh remote resource policy and all source/input/cache/proof hashes were verified. No MIP or k20 submitted.

The worker fails closed on invalid license/source/cache/lineage or restart. An interrupted solve keeps its atomic same-commit checkpoint for a separately reviewed recovery; it never silently rebuilds the graph or starts another freshCG. Preserve old failed-budget and genuine ancestor costs plus lineage32.440s and graphvalidation11407.799s.

## Publication and next triggers

Current Doc extension table now has all six chains; its strict-model paragraph records full graph validation and job824877. Weekly slides10/42 and notes carry the same evidence. Other slides/figures/history tabs and historical decks are preserved. [Publication verification](publication/README.md):32/32 automated checks plus visual review passed. Large exports remain local; small allowlisted evidence, code and logs are published from the isolated evidence worktree.

Next: collect newly terminal baseline endpoints; verify824877native license/start and then pricing progress or terminal checkpoint. Recover real failures without changing scientific settings. Only after a verified cached-CG endpoint consider the already-authorized finite-pool MIP; keep k20 gated. Completed matrix/capacity/salvage campaigns stay closed. Keep four-hour meaningful-change monitoring and notify promptly on SSH loss.
