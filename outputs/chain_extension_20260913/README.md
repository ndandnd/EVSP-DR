# Continuing the six chains beyond 15 buses

**Results through14September00:25EDT:** new integer matches C2k21=21,C4k19=19,C6k20=20 have fleet proofs within their saved pools and pass individual-route replay. Highest individual targets by chain are18,21,18,19,18,20; earlier gaps remain open.29MIPs are verified:19target matches and10open one-bus gaps. CG has30certificates among33endpoints; C1/C4/C5k19 hit their time limits with negative reduced costs below tolerance. C4k19 nevertheless matches the integer target. Charging optimality is separate. [Current tables, every MIP and CG stopping reason](../cumulative_budget_20260913/status_20260914T042303Z/README.md).

The six baseline chains have each found 15-bus integer solutions at target k=15. This campaign tests how much farther the same method scales. **k=16–25 are the next 60 cases.** Inputs through k=40 are frozen for later continuation; they are not all submitted now. **Submitted 13 September, verified 10:25 EDT:** graph array **133908**, 60 CG jobs and 60 MIPs. Fifty graph tasks were running and ten waited for an array slot; all CG/MIP dependencies passed validation. The 33 held historical tasks were untouched. See [job map](case_jobs.json), [launch verification](launch_verification.json) and [native validation](validation.json). Submission and live status are recorded separately from scientific results.

| What changes | What stays fixed |
|---|---|
| Add one randomly selected unused GIRO duty at each k | All trips and duty membership already present at the preceding k |
| Rebuild the graph for the enlarged trip set | Covering, 240 kWh batteries, 240 kW charging, flat electricity prices, 5 per charging start |
| Inherit every eligible saved route sequence from the previous k | No GIRO solution injection; singleton initialization supplies feasibility |

The added-duty order is chosen before seeing solver outcomes, using a recorded seed for each chain. Two duty IDs have alternate variants; each chain preserves its existing variants and uses the documented compatible-variant rule. These are six continuations, not six new independent samples. [Inputs and exact selection rules](INPUTS.md), [membership ledger](inputs/membership.csv).

| Chain | New target range | Trips at k=25 |
|---|---:|---:|
| 1 | 16–25 | 608 |
| 2 | 16–25 | 573 |
| 3 | 16–25 | 551 |
| 4 | 16–25 | 617 |
| 5 | 16–25 | 624 |
| 6 | 16–25 | 580 |

## How the jobs run

There are 60 graph preparations, 60 CG jobs and 60 final MIPs. Graphs do not need previous solutions, so an array allows 50 preparations at once. Within each chain, CG at k waits for its own graph and CG at k−1. Each MIP waits only for its own CG. A MIP does not delay the next CG. Six simultaneous CG chains are a data dependency, not an arbitrary cluster throttle.

| Stage | CPUs | Memory | Solver/preparation budget | Slurm allocation |
|---|---:|---:|---:|---:|
| Prepare graph | 2 | 64 GiB | 12 hours | 12½ hours |
| Import saved routes and run CG | 8 | 96 GiB | 4 hours total | 5 hours |
| Final two-stage MIP | 8 | 24 GiB | 1 hour total | 2 hours |

All jobs use the default partition and exclude `scaglione-compute-01`. Slurm decides how many fit available CPUs and memory. Stage 1 has up to 30 minutes to minimize buses. Stage 2 uses the remaining solver time to minimize electricity plus charging-start costs, subject to fleet ≤ the validated stage-1 incumbent. It can run even when stage 1 has not proved that incumbent optimal.

Graph preparation is a real scaling risk: a previous 750-trip graph exceeded 12½ hours before CG. This campaign adds synchronous progress reports during graph construction. The event graph source is `a0e0bb7681c8451e3cbbbfa06aef390026d9af4b`, whose only solver-source difference from successful baseline `e091a4d` defers expensive tie-key calculation while preserving the winner/order rule. The small validation compares instrumented and ordinary graph bytes. MIP source is `871d057e1067411f09581e37d78f7c1ca43f68bb`. The omitted-LP-setup and capacity-pricing changes are not mixed into this campaign.

## What a result will establish

Track these separately: graph preparation time; inherited-route import time; pricing and LP time; CG termination and certificate; weighted LP objective and fractional route count; MIP fleet, bound, gap and proof scope; individual-route physical replay; target attainment. A time-capped RMP value is not a certified full-model lower bound. A proved finite-pool fleet is not automatically full-model integer optimality.

This is the baseline model: no shared-station capacity, reserve or terminal-SOC floor. Duplicate-trip removal is not validated. Success here does not establish feasibility under every GIRO constraint. The [separate model audit](../model_fairness_audit_20260913/README.md) describes the stricter results.

## Storage, restart and monitoring

Launch root: `/home/nc437/ladder-lite/chain_extension_20260913`. Large artifacts: `/share/scaglione/nc437/evsp-dr/chain_extension_20260913`, linked from the launch root. Code borrows Git objects from the graph-recovery checkout; protect that source until the campaign is archived.

Every stage uses a separate job/restart directory. After preemption, CG can copy and resume its identity-checked checkpoint; unpublished graph construction starts again; MIP starts a new search tree. Completed results are published only after clean exits and validation. The monitor records provisional checkpoints as progress, not final research results. Algorithmic time caps do not trigger blind retries. Held historical jobs and V2G work are untouched.

`manifest.json` pins input/source/tool hashes, settings, parents and resources. `jobs.json` and `case_jobs.json` record exact dependencies. `validation.json` contains actual native checks. The first two-second validation failed on a newly written check that confused a compact summary hash with a full-file hash; that assertion was corrected, with both records retained. It was not a solver or cluster-access failure. Default-MIP attempts are added to the existing preemption study.
