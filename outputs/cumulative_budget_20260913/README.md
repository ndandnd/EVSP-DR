# Is sequential warm starting better use of the same computation budget?

**Launched 13 September, verified 14:20 EDT:** 120 stage jobs across 24 datasets; 45 running and 3 finished at the first check, with 72 waiting for true prerequisites. No execution errors. [Exact job map](case_jobs.json), [launch checks](launch_verification.json), [native validation](validation.json). These are launch observations, not conclusions from the comparison.

**Latest results, 13 September 14:44 EDT:** all six fresh k=5 CG/MIP comparisons are complete. Five fresh pools recover five buses; chain5 proves six in its fresh pool while the warm pool proves five. Fresh chain6 k=8 also recovers eight. [Current table and precise interpretation](status_20260913T184244Z/README.md).

**Earlier first results, 13 September 14:30 EDT.** Three fresh k=5 runs have pricing certificates. Two have also finished MIP and recovered five buses. This is evidence that inherited columns are unnecessary for these two cases under the cumulative allowance; larger cases are still running.

| Chain / target | Cumulative CG allowance | Fresh CG time | Fresh integer fleet | Warm-pool integer fleet |
|---|---:|---:|---|---|
| C3 / k=5 | 51.0 min | 5.1 min; certified | 5; proved in pool | 5; proved in pool |
| C5 / k=5 | 110.0 min | 8.0 min; certified | MIP pending | 5; proved in pool |
| C6 / k=5 | 15.0 min | 6.1 min; certified | 5; proved in pool | 5; proved in pool |

The two fresh integer solutions passed individual-route physical replay. Pricing certificates concern the tested event graph and reduced-cost tolerance 0.0001; fleet proofs concern each finite pool. The baseline has no shared-station capacity or terminal-SOC floor. These are the first completed cases, not a representative sample of all outcomes. The original first collection contained seven completed warm-reference MIPs and no fresh endpoint; retain it as launch history. [All 24 comparison rows](first_results.csv), [new source records](first_results_collection.json), [launch collection](launch_collection.json).

**Question.** The k=8 warm run benefits from columns produced while solving k=2,…,7. Its marginal k=8 time therefore understates the work needed to obtain that pool. This experiment gives a fresh k=8 run the accumulated time used by that chain, then compares the final integer solutions. The fixed panel contains all six chains at k=5,8,10,15: **24 datasets**, chosen by size and chain, without filtering on outcomes.

## The two budgets

Let t(c,j) be the recorded native time for importing routes and running CG at step j of chain c. The primary fresh allowance is B(c,k)=ceil(sum from j=2 to k of t(c,j)). Every fresh run starts from single-trip routes, receives no inherited or GIRO solution columns, and uses only the same input graph. The graph is problem structure, not a saved solution.

**Primary fresh CG allowances, hours.** These are accumulated budgets, not predicted runtimes or new results.

| Chain | k=5 | k=8 | k=10 | k=15 |
|---|---:|---:|---:|---:|
| 1 | 3.39 | 8.94 | 9.68 | 15.87 |
| 2 | 0.57 | 10.16 | 10.97 | 16.09 |
| 3 | 0.85 | 6.30 | 14.77 | 16.59 |
| 4 | 0.34 | 6.45 | 11.94 | 15.26 |
| 5 | 1.83 | 10.51 | 20.16 | 23.14 |
| 6 | 0.25 | 1.74 | 5.89 | 10.54 |

Native CG time includes input preparation, graph loading, inherited-route checks and optimization. Original graph construction occurred separately in every audited ancestor. A second allowance adds the original graph-construction times of the smaller instances j=2,…,k−1. The target graph is common to both methods and is recorded separately; graph-load time is not added twice. This sensitivity credits measured constructor time, not unrecorded historical serialization, hashing or other setup costs. We do not claim to have recovered every second of past computation.

The audit traced **84 distinct successful CG stages and all 78 actual parent links**. Every consumed parent hash matches the current source. All roots use singleton initialization. Budgets exclude intermediate MIPs, queue delays, unrelated tuning experiments and failed attempts outside the actual pool ancestry. Earlier MIPs were not used to produce these inherited columns. [Exact budgets and ancestry](audit/README.md), [CSV](audit/budgets.csv), [machine-readable targets](audit/targets.json).

## What runs

For each of the 24 datasets:

1. Run fresh CG with its primary budget.
2. Run a one-hour MIP on that fresh pool, if CG produced a usable LP.
3. If primary CG stops at its time limit, resume its own same-instance checkpoint to the larger graph-cost allowance. This continuation does not import any warm-chain columns. A primary pricing certificate instead supplies both endpoints; there is no redundant second CG search.
4. Run a one-hour MIP on the larger-budget pool. If the CG endpoint is shared, reuse the first MIP result too.
5. Independently rerun the saved warm target pool through the identical one-hour MIP setup. This makes the integer-search budget comparable and fills the missing original C3k10 MIP artifact.

The first MIP has no influence on fresh CG continuation. Additional budget is not spent manufacturing columns after CG certifies: if it stops early with a small pool, that is an experimental outcome. If longer fresh CG reaches the same certificate but still gives a worse integer solution, the evidence points to pool composition rather than a simple lack of CG time. That remains a hypothesis until the new results arrive.

## Fixed solver and resource settings

CG uses source `e091a4dba549510238507ef5e5367abea958bd30`; all MIPs use `871d057e1067411f09581e37d78f7c1ca43f68bb`. Covering,240kWh/240kW,event2.5kWh/5min,flat prices,zero reserve,no terminal floor,no shared station capacity; route cost100000+electricity+5 per charging start. Thirty columns per iteration; reduced-cost selection; tolerance1e-4; indexed graph replay enabled; unused LP-incidence setup retained. No diversification, greedy or GIRO seed is introduced.

Every MIP gets3600seconds total: up to1800seconds minimizing buses, then the remaining time minimizing electricity plus charging-start cost subject to fleet≤the validated first-stage incumbent. Report proof status even when that incumbent is unproved.

All jobs use default_partition and exclude scaglione-compute-01. CG requests8CPUs/96GiB; MIP8CPUs/24GiB. The 24 primary CG jobs and24 warm-reference MIPs are immediately independent, so all48 are eligible together. Later stages retain only real data dependencies; the larger-budget MIP also waits for the primary MIP so that an aliased endpoint can reuse its result. There is no arbitrary smaller concurrency cap. Existing k16–25 expansion and held historical/V2G jobs are untouched.

The largest primary CG allowance is23.15hours; the largest total sensitivity allowance is33.33hours. These are generous maximum budgets, not promised completion times. Slurm allocations add one hour around CG budgets and allow two hours for each one-hour MIP. The default partition currently has no maximum job time. Both finished-before-budget and censored runs are reported.

## Fair interpretation

The historical ancestry spans two code revisions and different machines; this is a retrospective accumulated-budget control, not a randomized single-code speedup experiment. All new fresh runs share one current source. Warm reference MIPs are newly matched to fresh MIPs, while historical CG time remains measured rather than recreated.

Report requested and actual native CG time, budget overshoot, process elapsed time, measured user/system CPU, Slurm CPU usage/allocation exposure, and lost work from preemption separately. Equal elapsed allowance on eight allocated CPUs does not force equal CPU consumption. Historical CPU figures cover CG allocations, excluding external graph jobs and intermediate MIPs. New interrupted attempts with missing measurements remain explicitly unmeasured, not zero.

Keep weighted LP objective, fractional route count, pricing certificate, finite-pool integer proof, target attainment and physical replay separate. A time-limited RMP is not a full-model LP lower bound. Covering can duplicate trips; duplicate-trip removal and shared station feasibility are not validated by this baseline comparison.

## Files and restart policy

Launch root `/home/nc437/ladder-lite/cumulative_budget_20260913`; new large journals/results are under `/share/scaglione/nc437/evsp-dr/cumulative_budget_20260913`. Existing graph files are hardlinked on the home filesystem without new graph storage. The original producer manifests are retained alongside explicit, previously audited consumer-compatibility attestations; the native loader checks full pickle hashes. Protect their original source roots and the frozen CG/MIP execution checkouts.

`campaign.py prepare` freezes inputs/ancestry/budgets/resources; `validate.py` exercises actual cache loading, fresh CG, native checkpoint continuation, both certificate-sharing branches, warm/fresh MIP replay and completed-MIP requeue protection; `campaign.py submit` records exact jobs/dependencies. Never rerun preparation or submission over an existing manifest/ledger. `collect.py` keeps shared endpoints as aliases rather than duplicate scientific observations and excludes validation fixtures.

Scheduler preemption can requeue into a new job/restart directory. CG resumes only a copied identity-checked checkpoint; MIP starts a new search tree. Completed publications are reused. Only MIPs that actually attempt optimization join the preemption cohort; shared/no-pool jobs do not inflate one-hour survival statistics. Algorithmic time caps continue only through the explicitly budgeted sensitivity, not unlimited blind retries.
