# Focused review monitoring

Run once each hour while the review campaigns are active:

```sh
python3 /Users/nadan/Documents/projects/demandresponse/outputs/independent_review_20260916/execution/collect_review.py
```

The validated run at **2026-09-16 22:18 UTC** completed in **2.99 seconds**. All six campaign collectors and both scheduler queries succeeded. `monitor/latest_path.txt` points to the latest dated snapshot; that directory contains `snapshot.json`, `receipt.json`, `squeue.txt` and `sacct.txt`. Collection failures produce `failure.json`; partial collector/query failures are marked `partial_collection`.

| Campaign | Review question |
|---|---|
| P1 MIP repeats | F5 fresh pools; F2/F4 warm-pool misses |
| Minimum-duration tariff comparison | F6/F7 charging response |
| Strict chain 5 | F4 depot power, SOC reserve and vehicle groups |
| Random trip groups | F5 warm starts without GIRO duty grouping |
| Full Partille 40 | F8 full-instance feasibility and bound |
| Frölunda ladder | Generalization (review item 14) |

The collector queries only registered review jobs and their recorded attempts. It invokes each campaign's small native collector; it does not use the old full-project collector, launch jobs, cancel jobs, or resubmit failed jobs. Scheduler state and attempt receipts are kept separate from pricing certificates, finite-pool fleet proofs, physical replay and duplicate-trip removal checks. A successful collection is **not** a scientific verification. A running attempt has incomplete elapsed time; retain completed and interrupted attempts when calculating total effort.

Only the synthetic peak08/12/18 tariff rows are transferred from the demand-response campaign. All SE3 prices, scientific outputs, costs and private attempt details remain on Unicorn. The native internal collection is saved under `/home/nc437/ladder-lite/review_monitor_20260916/<UTC stamp>/dr/private_internal/`. Do not copy that directory into Git, the document, or local reports. Scheduler metadata may identify those jobs but contains no prices or outcomes.

For the hourly monitor: stay quiet when status is unchanged; notify on meaningful completion, worker failure, lost SSH access, or a blocked dependency requiring action. If access fails, say so promptly. Inspect the relevant native evidence before updating the research register or Google Doc. Do not infer a full-model fleet proof from a finite-pool result. Do not restart jobs from this runbook; investigate the recorded attempt first. Preserve held historical jobs and other projects.

An hour is sufficient for the multi-hour CG and MIP budgets. Extra checks are useful after a reported failure or completion, not to fill idle time.

## How to interpret the next results

- **F5 / item 7:** the review's trigger is a matching seed in at least four of six fresh k15 pools. Report every seed and the old 30-minute versus new three-hour fleet budget. Do not call this an isolated causal effect of inheritance.
- **F2/F4 / item 8:** 30 buses in the LP does not imply a 30-bus integer solution. A 12-hour timeout also does not exclude one.
- **F2/F5 / item 9:** include the four reused seed-zero cells once. Verify ordered-pool and initializer identity before attributing differences to seed or search allowance.
- **F6/F7 / item 10:** distinguish original invoice estimates from feasible schedules. Check active charging durations, per-bus SOC, attained ending energy and trip assignment. A phase-I stop is not combined-objective pricing convergence. Joint CG may miss columns present in fixed-duty alternatives; no savings are guaranteed.
- **F4 / item 11:** each prefix needs both vehicle-group components. This changes several physical assumptions and the pricing driver, so it does not isolate a runtime cause. Shared capacity, nonlinear power, minimum-duration/setup rules and terminal SOC are still omitted.
- **F5 / item 12:** intermediate stage numbers are not GIRO fleet targets. Account for graph, initialization/CG, MIP and interrupted attempts separately. Only the final trip set equals C1 k15.
- **F8 / item 13:** reconstruct any F3 bound from this input's paired LP/pricing records and cost envelope. The 48-hour CG clock follows separate graph preparation.
- **F8 / item 14:** preserve the declared conservative Frölunda deadhead conversion. Early small-case success does not establish full generalization.

Record a tested proposition as **VERIFIED**, **REFUTED**, or **UNRESOLVED** in `ledger.json` and the execution README. Missing proof is not proof of the opposite. New native flags are first recorded as such; verify source hashes and scientific validation before promoting a headline.

The current Doc's source is `doc/current.html`. The dashboard builder now reads it; do not regenerate superseded claims from historical prose. Update the current tab in place, retain figure/history tabs, and do not edit Slides. The existing workbook predates the added audit fields; use the audited CSV, then rebuild the workbook when a useful batch of endpoints is verified.

Publish through the worktree named in `outputs/research_register/publication_workspace_path.txt`, preserving the dirty main checkout. Regenerate and inspect `../build_publication_allowlist.py`; never copy restricted real-price outputs or whole result trees blindly. Verify the Unicorn research-register mirror after publication. The original F1 equality-pool and shared-capacity flags remain distinct from the newly validated passenger assignment with empty driving.
