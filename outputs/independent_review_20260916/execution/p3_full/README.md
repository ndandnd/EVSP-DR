# P3 item 13 / F8 — full Partille attempt

**Submitted:** graph job 341404, fresh CG 341405, saved-pool MIP 341406. The graph was observed running; CG and MIP have their required data dependencies. F8's full-instance outcome remains **UNRESOLVED** until results arrive.

The frozen input has **40 canonical duties and 948 regular trips**, selected consistently with chain 1's service-day variants. Partille's raw 987 regular rows include 42 duty variants, so they are not the same simultaneous 40-duty instance. The input CSV, original path and SHA-256 are in `manifest.json`.

This is a fresh singleton initialization, without GIRO columns or inherited pools. Physics and objective match the baseline: 240 kWh, uniform 240 kW including PARX, no SOC reserve or ending-energy floor, no shared charger capacity; set covering; 100000 per bus + flat electricity + 5 per charging start. Event representation is 2.5 kWh / 5 minutes, 30 columns per iteration, reduced-cost tolerance 0.0001.

| Stage | Scientific allowance | Allocation |
|---|---|---|
| Graph preparation | Up to 24 hours, accounted separately | 2 CPUs, 96 GB, 25 hours |
| CG | 48 hours after graph preparation; also at most 100000 iterations | 8 CPUs, 128 GB, 50 hours |
| Saved-pool MIP | 3 hours fleet search, then 30 minutes charging with fleet ≤ incumbent | 8 CPUs, 24 GB, 4.5 hours |

The larger memory request follows the observed 22.6 GB packed graph and 12.4-hour build for the 779-trip k32 case. It is headroom, not a measured full-instance peak. All stages use the default partition, exclude `scaglione-compute-01`, and preserve preemption attempts. CG can resume a validated native checkpoint; MIP restarts its search tree.

CG is pinned to `a0e0bb76`; MIP to `871d057e`. `validation.json` checks the exact input, no inherited-column arguments and the frozen manifest. `INDEPENDENT_WORKER_REVIEW.md` records a separate wrapper review. Submission and scheduler checks are in `jobs.json`, `case_jobs.json` and `startup_verification.json`.

On completion, report the CG stopping reason and exact pricing scope. If a valid paired LP/dual/reduced-cost iteration survives, calculate the F3 numerical event-model lower bound using this input's own cost envelope; do not copy a smaller-instance bound. Keep fleet weight, weighted objective, pool proof and physical dispatch validation separate.
