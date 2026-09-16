# Item 10 follow-up — saved-pool MIPs (F6/F7)

**Planned, not submitted.** There are 24 fresh-CG pools (six k15 inputs × four tariffs). Replanned sequencing: **24 seed-zero searches first**, each dependent on `afterok` of its own original item-10 CG job. Seeds 1–2 are conditional follow-ups only on an audited unresolved seed-zero miss: at most 48 further searches. **The user explicitly has not authorized action-2 submission yet.** Release also requires completed/frozen source artifacts and a tested MIP-only runner. A held plan is not a scheduler dependency.

The current item 10 code **already imposes fleet ≤ k in CG and both MIP stages**. This follow-up changes the search allowance and seed, not the fleet cap. It will use only the saved fresh columns: no new CG, repricing, fixed-duty additions, or inheritance. The plan preserves terminal-energy coefficients, the shared ending-energy minimum, 240 kWh / 240 kW physics, zero fee, three-minute charging model and covering rows.

| Setting | Follow-up |
|---|---|
| Fleet objective | Minimize number of buses, with hard fleet ≤ k |
| Fleet time limit | 3 hours per seed |
| Seeds | Seed 0 first; seeds 1 and 2 only after an unresolved miss |
| Charging stage | Minimize electricity with fleet ≤ the stage-one incumbent; the original ≤ k cap remains |
| Total solver allowance | 3.5 hours; charging receives remaining time, at least 30 minutes if stage one consumes all 3 hours |
| Initialization | A new, cold MIP reconstructed from the identical ordered pool; no borrowed incumbent |
| Candidate resources | Default partition; 8 CPUs, 32 GB, 4.5-hour allocation; reserved GPU node excluded |

Before launching, freeze the source completion, ordered journal and input/tariff hashes. Do not read a growing journal as a completed pool. Validate that records contain only physical routes, all trip IDs are legal, and the model reproduces the original coverage, ending-energy and fleet rows. Missing trip coverage establishes that particular saved pool cannot cover the instance; do not spend three hours rediscovering that fact. A different MIP wrapper/root state means this is a new search on the same pool, not a pure continuation of the original tree.

**Charging cost is shown in the comparison only when the selected fleet equals k and the dispatch passes the physical checks.** If a run finds fewer than k buses, report that fleet improvement separately and mark the equal-fleet charging comparison unavailable. A timeout without an incumbent is unresolved, not proof of infeasibility and not a zero charging cost. Report achieved ending energy, charge starts, minimum session duration/SOC, bounds and solver status for every seed. Keep saved-pool infeasibility separate from infeasibility of the full route model.

The legacy `run_exact_pool_mip.py` must not be used blindly: item 10 has a different journal/schema and an aggregate terminal-energy row. A dedicated loader must preserve those features and be tested against a native small case before any release. `plan.json` deliberately leaves execution commit and final pool hashes unset until those gates pass. Synthetic results may be public; SE3 numerical results remain internal.

## Conditional release details

The original item-10 CG job includes its original two-stage MIP, so `afterok:<source_job>` waits for that entire job. A failed/preempted terminal parent cannot satisfy `afterok`; do not silently switch to `afterany`. Validate source completion and freeze the pool inside the follow-up worker before loading it.

The hard cap makes a fleet above k impossible. A seed-zero **miss eligible for repeats** means a correctly terminated, valid run with no physically valid fleet<=k and no pool-infeasibility proof. A validated fleet<k is a success, reported separately for charging; do not run more seeds just because it differs from k. Proven pool infeasibility does not benefit from seed repeats. Invalid output or infrastructure failure is a diagnostic case, not a scientific miss. Submit seeds1/2 only after inspecting this gate, with `afterok` on seed0; do not prequeue all72 unconditionally.

The P1/action1 endpoint reporting gate does not automatically authorize this campaign. No Slurm commands are executed by the plan generator.
