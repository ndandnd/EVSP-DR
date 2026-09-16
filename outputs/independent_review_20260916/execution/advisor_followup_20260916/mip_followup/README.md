# Item 10 follow-up — saved-pool MIPs (F6/F7)

**Planned, not submitted.** There are 24 fresh-CG pools (six k15 inputs × four tariffs) and three seeds per pool: **72 candidate searches**. Release requires the user's confirmation after reviewing cluster load, completed and frozen source artifacts, and a tested MIP-only runner. A held plan is not a scheduler dependency.

The current item 10 code **already imposes fleet ≤ k in CG and both MIP stages**. This follow-up changes the search allowance and seed, not the fleet cap. It will use only the saved fresh columns: no new CG, repricing, fixed-duty additions, or inheritance. The plan preserves terminal-energy coefficients, the shared ending-energy minimum, 240 kWh / 240 kW physics, zero fee, three-minute charging model and covering rows.

| Setting | Follow-up |
|---|---|
| Fleet objective | Minimize number of buses, with hard fleet ≤ k |
| Fleet time limit | 3 hours per seed |
| Seeds | 0, 1, 2; all reported |
| Charging stage | Minimize electricity with fleet ≤ the stage-one incumbent; the original ≤ k cap remains |
| Total solver allowance | 3.5 hours; charging receives remaining time, at least 30 minutes if stage one consumes all 3 hours |
| Initialization | A new, cold MIP reconstructed from the identical ordered pool; no borrowed incumbent |
| Candidate resources | Default partition; 8 CPUs, 32 GB, 4.5-hour allocation; reserved GPU node excluded |

Before launching, freeze the source completion, ordered journal and input/tariff hashes. Do not read a growing journal as a completed pool. Validate that records contain only physical routes, all trip IDs are legal, and the model reproduces the original coverage, ending-energy and fleet rows. Missing trip coverage establishes that particular saved pool cannot cover the instance; do not spend three hours rediscovering that fact. A different MIP wrapper/root state means this is a new search on the same pool, not a pure continuation of the original tree.

**Charging cost is shown in the comparison only when the selected fleet equals k and the dispatch passes the physical checks.** If a run finds fewer than k buses, report that fleet improvement separately and mark the equal-fleet charging comparison unavailable. A timeout without an incumbent is unresolved, not proof of infeasibility and not a zero charging cost. Report achieved ending energy, charge starts, minimum session duration/SOC, bounds and solver status for every seed. Keep saved-pool infeasibility separate from infeasibility of the full route model.

The legacy `run_exact_pool_mip.py` must not be used blindly: item 10 has a different journal/schema and an aggregate terminal-energy row. A dedicated loader must preserve those features and be tested against a native small case before any release. `plan.json` deliberately leaves execution commit and final pool hashes unset until those gates pass. Synthetic results may be public; SE3 numerical results remain internal.
