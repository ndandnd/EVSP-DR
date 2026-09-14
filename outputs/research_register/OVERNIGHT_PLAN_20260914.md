## Current queue and priorities — 16:13 EDT [One current table of overnight work and verified findings](../overnight_parallel_20260914/status_20260914T200527Z/CURRENT_STATUS.md). **99 jobs running**, 48 solver dependencies and 18 conditional recovery checks; none broken. Scientific collection through 16:09: 28 small-seed CG certificates, seven seed MIPs with six target matches; original chain 1 now matches 23. One MIP auto-restarted after preemption. The dated notices below preserve earlier states.

# Overnight experiments — 14 September

## Additional independent comparisons and queue protection, 14:45 EDT

**Final queue check, 14:44 EDT: 109 EVSP jobs running — 21 CGs, 70 MIPs and 18 graphs.** Another 55 research stages wait for required predecessor outputs; 18 conditional recovery gates wait separately. All 46 new LP-support MIPs were running. No invalid dependencies, resource waits or reserved-GPU-node CPU use appeared. [Exact queue census](../overnight_parallel_20260914/final_queue_20260914T184444Z/FINDINGS.md).

Three reconstructed fresh pools at approximately four hours now have **MIPs 189908–189910 running**. Construction took 15–16 seconds per case by reading saved journals; no CG was repeated. These are logged pre-insertion iteration pools, not exact wall-clock snapshots or new pricing certificates. Their 3½-hour total / three-hour fleet budgets match the current small-seed MIPs. [Settings, cutoffs, hashes and jobs](../overnight_parallel_20260914/prefix_controls/README.md).

**Eighteen conditional graph recovery gates are deployed.** They wait for each original graph, reuse a verified successful cache, and rebuild only after a confirmed timeout. Retries have a 24-hour internal allowance and 24½-hour allocation, with unchanged graph physics and source. Every previous-k and own-CG MIP dependency remains intact. A hardened second version handles brief accounting delays; the unused first-version gates were cancelled only after their replacements were verified. These are operational safeguards, not eighteen new research cases. [Authoritative v2 runbook](../overnight_parallel_20260914/graph_retry_v2/README.md).

**Decomposition wave submitted:** nine pool constructions completed and all 46 independent MIPs were launched: nine individual partitions, 36 pairs and one all-nine union. At 14:35 EDT, 45 MIPs had finished with 34–37 buses; the all-nine union remained running. Every completed fleet was proved within its selected pool and passed route replay, but no completed pair improved its best contributing partition. [Exact results, bounds and source hashes](../overnight_parallel_20260914/status_20260914T183534Z/README.md).

**The second matched selection treatment is submitted:** nine constructions 190339–190347 completed;46 independent MIPs 190402–190447 are admitted with no dependencies. The first 512-route selector preserved each component's integer witness but omitted 3, 187 of 3,413 positive-weight source LP routes. The second retains both the integer witness and all LP support, then fills to the same 512 routes per component. Every component fits within the cap. Inputs, pool sizes, native solver code, integer warm starts and two-/four-hour limits remain matched. These are two treatments of one 750-trip parent, not 92 independent datasets. [First campaign and jobs](../decomposition_pool_union_20260914/README.md); [LP-support treatment and jobs](../decomposition_lp_support_union_20260914/README.md). Native validation accepted all 2,048 source columns, rejected/repaired none, and passed full 750-trip route replay. The short validation MIP is not a production result.

The independent service-overlap lower bound for that parent is **29**, so even a 32-bus solution would match the GIRO target without proving global fleet optimality. [Exact input hash, time windows and 29-trip witness](../decomposition_union_audit_20260914/PARENT_OVERLAP_BOUND.md).

**Submitted and verified.** The 13:07 EDT queue check counted **66 EVSP jobs running and 84 waiting on real dependencies**, excluding 33 held historical tasks. All 36 new warm-start CGs were running. All eight pricing diagnostics started; the duty-13408 pair has already finished with pricing certificates and one-bus pool solutions. Counts change as short cases finish.

The queue has useful work, but too much of it depends on long graph builds or a previous chain result. The additional work below uses already available inputs so those dependencies do not limit the whole campaign.

| Experiment | Independent CG jobs | Later integer solves | Question |
|---|---:|---:|---|
| Small warm starts: previous integer routes versus routes with the largest LP weights | 36 | 36, each waiting only for its own CG | Which inherited routes help create a good integer solution? |
| Find the difficult cases in capacity-aware pricing | 8 | Short diagnostic MIP within each allocation | Which one-bus duties cause pricing to take hours, and does cached pricing help? |
| Existing six-chain extension through k=28 | Already submitted: 18 graph builds, then 18 CGs | 18 | How far does the baseline scale? |

The first two rows add **44 independently eligible CG jobs**. They use the default partition and exclude `scaglione-compute-01`. No smaller concurrency cap is imposed. Slurm determines when the requested resources are available. Existing previous-k dependencies and held historical jobs remain intact.

## What the warm-start comparison changes

For each of six chains at targets 8, 10 and 15, use the preceding target's saved results. One method selects the trip sequences used by its integer solution. The other selects the same number of distinct sequences with the largest positive LP weights. These are 7, 9 or 14 selected sequences, rather than the entire earlier column pool.

Both methods replay those sequences against the current model and initialize the current instance with singleton routes. They use the same input, cached graph, solver version, objective and time limits. No current-target solution supplies seed routes. The experiment matches selected sequence counts; it does not necessarily match trip coverage or the number of columns actually added after replay. Those quantities are recorded separately.

CG receives at most four hours. Each final MIP receives at most three hours for fleet search within three and a half hours total; charging uses the remaining time with fleet constrained by the first-stage incumbent. Parent CG and MIP costs are retained for end-to-end accounting. Earlier fresh and full-pool results are context, not automatically matched runtime controls.

[Frozen cases, validation and jobs](../overnight_parallel_20260914/README.md).

## Why the capacity tests stay small

The four capacity-enforced k=2 runs reached their 220-minute pricing deadline after only three or four completed iterations. Their saved pools contain 26 or 27 columns. Short MIPs prove fleets of 3 or 12 within those pools, but pricing did not certify convergence. This does not prove the full model needs those fleets.

The eight new tests compare reference and cached pricing on duties 13405–13408, under identical flat prices, 240-kWh battery and capacity settings. They bracket known easier and harder duties. Each gets 220 minutes of CG and a short ten-minute MIP diagnostic. This is a pricing diagnostic; a one-bus test cannot establish performance under contention between multiple buses.

[New diagnostic design and jobs](../capacity_pricing_boundary_20260914/README.md); [verified k=2 endpoints](../capacity_pricing_boundary_20260914/strict_capacity_k2_audit_20260914T1659Z.md).

## Why the existing dependencies remain

At the 12:33 EDT snapshot, 34 allocations were running and 53 were waiting for real input data; another 33 historical tasks were held. Eighteen running allocations were graph builds. Fresh filesystem checks confirmed that all eighteen were actively progressing. All 52 checked dependency links in the new chain campaign are valid.

Historical graph progress suggests roughly 8–12 hours total construction time, with two cases close to their 12-hour watchdog. These estimates are approximate. Graph construction has no partial checkpoint, so preserve the active builds; retry individual cases with a longer allowance only after an actual timeout. Simply extending Slurm time would not override their internal watchdog.

[Dependency and resource audit](../overnight_parallel_20260914/queue_audit/FINDINGS.md).

## Reading tomorrow's results

Record imported sequences and added columns, CG time and stopping reason, pricing certificate, integer fleet and finite-pool proof separately. Include the earlier computation that produced each warm start. A scheduler completion or an LP objective alone is not an optimality proof. The large-chain baseline still omits shared charger capacity and a terminal-SOC floor; keep it separate from the capacity diagnostics.

All **80 new production jobs** are submitted: 36 warm-start CGs, 36 later MIPs and eight pricing allocations. [Exact warm-start job map](../overnight_parallel_20260914/case_jobs.json), [eight pricing jobs](../capacity_pricing_boundary_20260914/README.md), [live queue observation](../overnight_parallel_20260914/livequeue.json). The 84 waiting jobs comprise 36 new MIPs and 48 existing chain/graph-dependent stages. No invalid dependency was identified. Each new dependent MIP is cancelled if its own CG fails outright, preventing a permanently blocked queue entry; a valid time-limited CG endpoint still feeds its MIP.

[First completed pricing pair](../capacity_pricing_boundary_20260914/boundary_results_20260914T1707Z.md): both reference and cached pricing certify on duty 13408 after 33 iterations, with 43 saved columns and one integer bus. This is a result for one duty, not a general equivalence or scalability claim.
