# How to read the chain table and constraint tests

The table reports the **largest tested target attained**, not a guaranteed range and not the first failure.

- Original one-hour MIP: each saved CG pool receives 3600 seconds total, with 1800 seconds reserved for the fleet-only first stage. The second stage uses the remaining time with fleet no greater than the first-stage incumbent.
- Longer MIP: a new search tree on the **same saved columns**, usually 10800 seconds for fleet plus 1800 seconds for charging. This is not extra CG.
- Seed repeats: Gurobi seeds 0/1/2 on an identical ordered pool, initialization policy and model. They change search decisions, not which bus duties form the input. The second table row takes the best available result, so it costs more computation than the first row.
- The large-chain CG allowance is 4 hours per k, separate from MIP and external graph construction. Sequential runs also inherit earlier CG work. Fresh controls receive the accumulated import+CG allowance through the selected k, excluding earlier MIPs and separately recorded graph construction. At k 15 those allowances are about 10.5–23.1 hours; actual fresh CGs stop with pricing certificates after about 1.3–4.8 hours.
- The ladder currently stops at 32. There is no tested k 33–39 frontier in this chain campaign. A separate full 40 experiment does not extend every chain. Failure at k does not imply failure at k+1. For example, C4's best reported k 31 search found 37 buses, whereas k 32 searches found 32. These are different finite pools/searches, not contradictory exact optima.

## Why sequential helps

The evidence supports richer integer combinations, not a claim that every fresh failure has one cause. Small subproblems produce routes covering useful bundles of trips; later stages retain them. Fresh CG is driven by fractional cost and need not generate every route useful for an integer combination, even at LP convergence. Controlled tests retained the same LP objective but improved saved-pool MIPs 9→8 and 11→10 with full inheritance; both smaller pools had proved their worse fleet counts. At fresh k 15, however, all 18 longer seed searches retain a bound near 15, so the distinction between insufficient columns and unfinished integer search remains unresolved.

## Requested physical constraints were tried; the headline ladder is a separate baseline

| Treatment | Observed result | What is still missing |
|---|---|---|
| PARX60 kW, capacity disabled, two tested k 2 physics settings | Both recover 2 buses and CG converges | Not evidence for all larger instances or simultaneous charging feasibility |
| Shared capacity enabled, with or without PARX60 kW | k 2 cases stop after 220 minutes of CG; saved-pool MIPs return 3 or 12 buses | No pricing certificate; cannot infer that the full model needs those fleets |
| One-duty combined-constraint flat-tariff pilot | One bus, CG certificate, capacity check passes | One easy case is not a scale result |
| Strict C5 chain:60 kW depot,15% reserve, group-specific batteries and segregation | Early stages collected; larger sequence unfinished | Shared station capacity remains disabled; this is not every GIRO rule together |
| C5k 31 single-factor arms | Approved full-pool replay/graph work is ongoing; no final comparison at 03:57UTC17Sep | Isolates power/reserve/battery/group factors; does not include shared capacity |

The capacity-disabled k 2 witnesses use two simultaneous connections at 2190L, whose documented limit is one. Thus charger capacity is a real constraint violation in those witnesses, not a cosmetic missing label. Capacity-enforced pilots expose a pricing bottleneck; their small pools do not establish a physical impossibility. We have not demonstrated the 32-bus ladder under the combined stricter model.

## Decomposition is a useful heuristic result, with incomplete runtime accounting

The best 34-bus solution on the 32-duty parent is 6.25% above the target. That is a worthwhile result even without exact recovery. The quoted 2–4 hour recombination solves exclude the earlier component graph/CG/MIP work. Nine partitions and 92 recombination searches do not measure the time of a single from-scratch decomposition solve. Before calling it computationally efficient, compare complete critical-path time, total CPU work and fleet quality on the same parent input.

Next: [bounded independent geographic-review request](../geography_review_handoff_20260917/README.md), using existing map/connection data and current source-aligned time-only bounds. No new solver jobs were launched.

Sources: `outputs/chain_extension_31_32_20260915/{manifest.json,campaign.py}`; `outputs/cumulative_budget_20260913/{audit/budgets.csv,status_20260916T194843Z/comparison.csv}`; `outputs/controlled_comparison_20260913/status_20260913T063519Z/README.md`; `outputs/strict_capacity_parallel_20260914/status_20260914T172907Z/README.md`; `outputs/independent_review_20260916/execution/monitor/20260917T035711Z/snapshot.json`.
