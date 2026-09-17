# F4 / Action 3: full single-factor comparison

**Six full experiments are submitted.** Each starts from all **254,068 distinct ordered trip sequences** in the original chain 5, target-31-bus saved pool (716 trips). The earlier 20-sequence tests were timing and correctness pilots; these new runs are not samples.

For each experiment we first optimize charging while holding each saved trip sequence fixed. Feasible sequences become initial columns for a new CG solve under that experiment's physics. A final two-stage MIP selects buses from the resulting pool. This tests which individual modeling changes remove useful routes and whether fresh CG can recover good solutions.

| Experiment | Only change from control | CG job | MIP job |
|---|---|---:|---:|
| Control | None: 240 kWh, all charging 240 kW, no SOC reserve, groups may mix | 342668 | 342669 |
| Slower depot charging | PARX charging is 60 kW | 342670 | 342671 |
| Minimum SOC | Keep at least 36 kWh (15% of 240) throughout | 342672 | 342673 |
| Smaller battery A | Every bus has 236.44 kWh | 342674 | 342675 |
| Smaller battery B | Every bus has 239.01 kWh | 342676 | 342677 |
| Separate vehicle groups | No route mixes 18E1 and 18E2 trips; battery remains 240 kWh | 342678 / 342680 | 342679 / 342681 |

Both smaller-battery arms use a homogeneous fleet. Group segregation is a separate structural experiment; it is not silently bundled with battery capacity. The unchanged 2.5 kWh SOC grid conservatively rounds energy states. Because 236.44 and 239.01 are not grid-aligned, these measure sensitivity of the discretized model, including rounding, rather than a pure continuous-capacity effect.

All arms retain set covering, flat electricity prices, a $5 charging-start fee, $100,000 fleet coefficient, full starting SOC and no separate terminal-energy target. Station capacity, fleet-type availability limits, 65% terminal SOC, driver rules, minimum charging duration and nonlinear charging are not included. This is not a claim of matching every GIRO constraint.

## What is running, and why there are dependencies

Preparation job **342539** creates 125 common sequence shards. Replay array **342540_[0–749%50]** processes all six arms with **50 concurrent tasks total**. A task reads only its small shard, avoiding 750 copies of the original large pool. Each arm attempts every sequence; a timeout or error stays **unknown**, never “infeasible.”

Graph-cache jobs **342655–342661** run independently of replay. Each arm's assembly job **342662–342667** depends on exactly that arm's 125 replay elements. Each CG job then depends on its own completed graph and assembled pool; each MIP depends only on its CG. The segregation arm shares one assembly and solves its two disjoint vehicle groups separately. This is valid here because there are no shared station-capacity constraints.

All jobs use `default_partition`, exclude `scaglione-compute-01`, and preserve held historical jobs. Replay requests 1 CPU / 4 GB / 6 hours. Full 716-trip graph and CG jobs request 96 GB; group components request 32 and 64 GB. Graph construction gets 40 hours separately: the original 716-trip graph had about 1.177 billion arcs, 18.83 GB of packed arrays and an approximately 11.1-hour initial build. We do not promise these finish overnight.

The nominal solver budget is **four hours of CG plus one hour of MIP per arm**, excluding replay and graph preparation. The segregation budget is split proportionally: CG 3,459 + 10,941 seconds and MIP 865 + 2,735 seconds. MIP spends half its allowance minimizing fleet, then minimizes charging costs subject to fleet **≤ the first-stage incumbent**. A first-stage incumbent is not called proved unless its bound supports that claim.

Budgets reset on a requeued attempt. Retain all attempt times when comparing cumulative computation; equal nominal budgets do not imply equal total work after preemption. Replay checkpoints every completed sequence. CG checkpoints after completed iterations, with a 300-second interval; a single long pricing or LP call can lose more than five minutes.

## Evidence and interpretation

- `manifest.json` freezes source hashes, six physical settings, replay scope and resources. `continuation_manifest.json` freezes graph/CG/MIP budgets and tooling.
- `replay_jobs.json` and `continuation_jobs.json` contain exact submission commands, true dependencies, timestamps and Slurm exclusion receipts.
- Execution commit: **be624eb486a7c3a4137f8b0fe660f9bb273ddf98**. `code.diff.patch` and `code.bundle` preserve the implementation; source pool came from commit `a0e0bb7681c8451e3cbbbfa06aef390026d9af4b`.
- Source sequence SHA256: `cdaae9f1f92fee3d2fac216917ca3a99478111e5c5c3a8ea7d9c9d6d843a69a4`. All 254,068 source columns had distinct ordered sequences. Input SHA256: `83585a8fbcba73f0262ba2e32e389bd014874c7f2e51aea65183e6dbed6321df`.
- Independent tests cover 600 flat-price charging windows, 35 fixed-sequence physical cases, 225 reduced-cost comparisons, group-subset event lattices, capacity rejection, 11 replay-recovery cases and four atomic-copy cases. See `independent_review/`.
- `stage_pipeline_test.json` records a 4.64-second integration test: graph → physical seed → certified CG → one-bus MIP, plus safe completed-cache retry.

The compact graph and flat-tariff shortcut preserve each fixed sequence's cheapest feasible realization in these no-capacity experiments; they do not retain every dominated station/time realization. They are not enabled for capacity pricing. Large outputs publish through validated atomic pointers; interrupted replay and CG progress are retained.

**Any control sequence reported infeasible or unknown is an audit anomaly.** It must be investigated rather than explained as a tighter-physics loss. The stricter arms may legitimately lose sequences. Assembly explicitly records missing trip coverage and any added physical singleton routes. No missing or timed-out replay is silently declared impossible.

Initial columns are neither a new full-model lower bound nor an integer solution. Keep six questions separate: did the scheduler finish; were all source sequences attempted; is the selected solution physically valid; did CG obtain a pricing certificate; did the finite-pool MIP prove its answer; did the fleet match 31? The full scientific comparisons remain pending until those results exist.

Native root: `/home/nc437/ladder-lite/action3_full_20260916`; large replay results: `/share/scaglione/nc437/evsp-dr/action3_full_20260916`. Use `MONITORING.md` for a read-only current snapshot.
