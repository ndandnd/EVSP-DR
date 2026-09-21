# 21 September — capacity pricing and strict-physics graph audit

The six capacity tests show a substantial algorithmic improvement: with the shortcut, the saved pools support fleets **1, 2, 5**; without it, **1, 3, 21**, under the same four-hour CG budget. Only the one-bus shortcut run has a pricing certificate (26.87 minutes). All six final MIPs prove optimality **within their own saved pools**; the other five RMP objectives are not full-model lower bounds.

[Comparison chart](capacity_on_off.png) · [CSV with exact results, paths and hashes](capacity_summary.csv) · [Machine-readable results](capacity_summary.json) · [Source checksums](SHA256SUMS.json)

| Case | Switch | CG job | Complete iterations | CG minutes | Pricing certificate | LP fractional route weight | Pool-MIP fleet / bound | Extra trip assignments |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| k1, 14 trips | off |628428|14|240.00|No, pricing deadline|1|1 / 1|0|
| k1, 14 trips | on |628429|67|26.87|Yes|1|1 / 1|0|
| k2, 23 trips | off |628436|6|240.00|No, pricing deadline|3|3 / 3|1|
| k2, 23 trips | on |628437|134|240.00|No, pricing deadline|2|2 / 2|0|
| k3, 35 trips | off |628438|2|240.00|No, pricing deadline|21|21 / 21|0|
| k3, 35 trips | on |628439|51|240.00|No, pricing deadline|3.236408|5 / 5|6|

These are capacity-only baseline physics: 240 kWh battery, 240 kW including PARX, zero reserve, shared opportunity-charger counts, unrestricted PARX charger count, 2.5 kWh SOC grid, 5-minute event block, flat tariff, set covering, weighted CG objective 100000×route weight + charging-related cost. MIP is fleet then charging cost. These are not fully matched GIRO strict-physics tests. Each pair shares input hashes; source commit `cf61e2c8618b067df032c3eb795476e64184ed0a`; singleton start, no GIRO columns; 4 CPUs/128G/6h allocation for CG, 4 CPUs/32G/90m for MIP. Six independent CGs ran without an arbitrary throttle. MIPs depended on their own CG only. Every job excluded scaglione-compute-01. [Manifest](manifest.json), [original jobs](jobs.json); CG retry IDs are in the table.

All six selected MIPs pass the continuous half-open station-capacity sweep. Duplicate service is a separate issue: one extra assignment for k2-off and six for k3-on. The route construction supplies individual event-model feasibility; this sweep is not a comprehensive exact-once GIRO dispatch validation.

## Why the shortcut helps

The bottleneck is pricing, not Gurobi's restricted LP. k2-on spent only 0.758 seconds in its 134 recorded LP solves; k3-on spent 0.340 seconds in 51. In the first matched k2 active-capacity-dual iteration, old pricing took **11,207.65 s**, versus **80.96 s** with the shortcut, with the same reduced cost (`-300012.07200000004`) and six-trip route length. This individual call ratio is 138.4×, measured on different nodes; it is not a hardware-controlled speed benchmark. The end-to-end improvement is supported by many more completed CG iterations and better pools in the fixed budget. k2 iterations 1–6 agree in LP objectives, reduced costs and priced trip counts across arms.

The implementation skips capacity-window recomputation where the relevant station has no active capacity price, and caches candidate windows for active sites. Source code and test history: `.codex-work/capacity-shortcircuit-20260917`. The off arm reports effective reference mode; the on arm activates the grid shortcut. Floating-point window ties can change later trajectories, so we do not claim identical eventual column sequences.

## Strict k15/k16: graph construction consumed the experiment

The strict k15 recovery `628314` finished at the scheduler level after 4h22m, but built **260,658,942 explicit Python arc objects** in **15,441.94 seconds**. Its four-hour CG timer was already exhausted: **zero pricing iterations**. The 596-column inherited/singleton pool then proved **37 buses** in `341291`. This says only what that impoverished finite pool can represent; it does not show that strict physics needs 37 buses. It also has 35 extra service assignments, and its station-capacity audit fails (capacity is not enforced in this single-factor arm).

The next strict group case k16 (`341292`) timed out after 6h02m during graph construction on unicorn-cpu-75, with **211.9 GiB** recorded peak RSS. This was a wall-time failure, not a new out-of-memory failure. k15 peak was 153.8 GiB. Both are the **18E2 subproblem** of global prefixes k15/k16 (8/9 reference duties respectively), not an all-group global fleet total. k16 originally blocks MIP341293 and next-CG341294, then their descendants.

- [k15 CG result: zero iterations and graph metrics](strict_original/cases/w5_k15_18E2/cg.json)
- [k15 MIP result and failed capacity audit](strict_original/cases/w5_k15_18E2/mip.json)
- [k15 CG Gurobi log](strict_original/cases/w5_k15_18E2/cg/628314_r0/result.json.gurobi.log)
- [k16 exact command](strict_original/cases/w5_k16_18E2/cg/341292_r0/command.json)
- [k16 Slurm timeout](strict_original/logs/341292_w5_k16_18E2__cg.err)

Remote source root: `/home/nc437/ladder-lite/review_strict_c5_20260916/`. For the capacity campaign, remote source root is `/home/nc437/ladder-lite/capacity_loop_20260921/`; the local `results/` tree mirrors it. All source artifacts and superseded failure attempts are retained.

## Focused algorithm repair

An isolated branch adds opt-in packed arcs to the strict no-capacity arm, plus an exact deferred JSON tie-key implementation already used elsewhere. Packed arcs store target, cost and reconstruction recipe, rather than hundreds of millions of Python action dictionaries. No-capacity pricing can safely retain the cheapest transition to each identical trip/SOC state; future feasibility and cost depend on that state. It cannot apply the same dominance under capacity duals, because station occupancy differs. The runner therefore rejects packed mode for either capacity-enabled arm.

Input/physics checks and full inherited-route physical replay remain compulsory. A tightly pinned compatibility exception permits only original strict commit `50ceb6c095a580f79f87b53bef536cac31f81963`, only without shared capacity; both execution commits and the exception are explicitly recorded. Any unmatched tariff, deadhead, reference data, input trip or physics value is rejected. New results are a new algorithm treatment, with original results retained. New cluster campaign details and test receipts will be recorded in `recovery/`.

This addresses the observed graph memory/startup failure. Shared-capacity pricing still needs a separate compact representation retaining station-specific alternatives; the successful shortcut helps small cases but is not evidence that larger full-GIRO comparisons are solved.

## New recovery receipt (21 September, 10:58 EDT)

Commit `35770aae2c08e7d5a356cc3b673e67608e5b1036` on branch `codex/strict-packed-20260921` has 44 passing executed tests: 15 strict/compatibility tests, 20 event/oracle tests, and 9 CG/MIP runner tests. The first event-suite attempt found a missing external test fixture; the fixture is now pinned, and the full suite passed. No implementation assertion failed.

| New job | Purpose | Dependency | Requested resources | Current observed state |
|---|---|---|---|---|
|646674|Original explicit / new explicit / new packed graph on the same 26-trip case and node; five fixed-dual pricing comparisons and physical replay|None|4 CPUs, 32G, 2h|Running on snavely-cpu-17|
|646675|New k16 18E2 packed CG treatment, original k15 inherited pool|Successful audited benchmark646674|8 CPUs, 64G, 6h allocation; original 4h CG budget|Pending input audit|
|646676|Fleet then charging MIP on its saved pool|CG646675|8 CPUs, 24G, 75m allocation; original 1h MIP budget|Pending CG|

[Execution/source/input/resource manifest](recovery/manifest.json) · [Verified Slurm submissions](recovery/jobs.json) · [Byte/AST model-compatibility audit](recovery/model_compatibility_audit.json) · [Held old descendant records](recovery/original_descendant_holds.json)

The 21 old blocked descendant jobs (k16 MIP onward) are preserved under a user hold with comment `SUPERSEDED_PENDING_AUDIT_strict_packed_20260921`. This makes their obsolete dependency status explicit while the new algorithm is evaluated. The independently held historical array537227 is untouched. The launcher's first post-submission bookkeeping attempt encountered an already purged completed Slurm ID; the retry examined live IDs and did not resubmit any jobs. Remote campaign: `/home/nc437/ladder-lite/strict_packed_20260921/`.

Packed storage is estimated at 16 bytes per retained edge (targets/costs/recipes), versus hundreds of bytes of Python objects per original explicit edge. The 64G request leaves substantial room for graph-building caches above an estimated 4–6GiB edge buffer; measured RSS from the controlled benchmark and k16 will decide whether further changes are needed. No large-case speed or final-fleet improvement is claimed before those outputs exist.
