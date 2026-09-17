# Full-Partille CG: 12-hour Scaglione replacement

**CG job343119 is queued on Scaglione, waiting for existing graph job341404_0.** It uses the tested checkpoint-recovery wrapper. The superseded held CG341405 was cancelled only after its held downstream MIP341406 was rewired to343119. MIP341406 remains held; the graph and all other jobs were left unchanged.

Submission: **16 September2026,21:23 EDT /17 September01:23 UTC**. Effective scheduler settings and before/after states are in `submission_receipt.json`.

| Setting | Effective value |
|---|---|
| Slurm partition | `scaglione` |
| Scheduler wall allowance | **12:00:00 exactly** |
| CG application allowance | **11h45m =42,300 seconds** |
| Child-process watchdog | At most11h50m; reduced by wrapper preparation time when needed |
| Shutdown reserve | Watchdog reserves120s inside an11h59m wrapper budget; existing60s SIGTERM-to-SIGKILL escalation retained |
| Resources | **8 CPUs,120G RAM** |
| Excluded node | `scaglione-compute-01` |
| True prerequisite | `afterok:341404_0` |
| Requeue | Enabled; checkpoint identity validation and older-candidate fallback |
| Input |948 trips from the frozen40-duty C1 instance; SHA904070ec8919dd11bf431eab62e1396ad4882d95f7d5917bcf83f34d1e20e607 |
| Solver |a0e0bb7681c8451e3cbbbfa06aef390026d9af4b |
| Initialization |Fresh singletons, no GIRO or inherited routes |

The **only solver-argument change** is the CG time cap, from172,800 to42,300 seconds. Battery, charging power, objective, covering master, grid, tolerance, pricing batch, graph and inputs are unchanged. The graph was already running and must finish before CG starts; this submission does not make that prerequisite disappear. The resulting bound/certificate depends on how CG actually stops.

## Why120G is a defensible estimate

The prior206–232GiB Slurm MaxRSS values came from k32 jobs using **eight inheritance workers**. They are process-tree measurements and can count shared pages multiple times. They were not silently divided by eight or treated as a measured fresh-run requirement.

The six solver parent-process peaks were36.78–42.34GiB for731–779 trips. Applying quadratic graph-size scaling to948 trips gives a tightly grouped **61.86–62.95GiB projection**. This is an empirical model of graph-dominated memory, not a proven peak bound. Those parent processes also handled large inherited pools; this run starts fresh and has no inheritance-process fan-out. Gurobi's solver threads share the solver process.

120G leaves about57GiB above the largest projection, roughly1.9times that estimate. Each eligible Scaglione CPU node advertises128,350MiB;120G requests122,880MiB and fits, leaving5,470MiB unrequested. The old128G request did not fit those nodes. The partial current graph build was around10GiB at the captured snapshot, but it was unfinished and is **not** used as a measured full-CG peak.

The first actual full40CG peak still needs measurement. Transient loading, graph shape and column growth can exceed a projection; this choice does not certify a maximum of120G. `resource_rationale.json` retains every historical source path/hash, trip count, packed-arc size and projection. The direct compute-node SSH probe failed host-key verification; Unicorn login access remained available. No PSS/cgroup memory value was invented.

## Recovery and shutdown

The unchanged tested helper stages complete private copies, validates native input/model identity and journal consistency, then atomically publishes the checkpoint directory. It rejects invalid newer candidates and tries older ones; if all previous candidates fail, it stops rather than silently starting empty. The original helper's11 tests remain applicable because both helper files are byte-identical. New static tests verify that graph arguments are unchanged, only the CG time cap differs, and the watchdog leaves its shutdown reserve after example preparation delays.

Native cumulative **recorded** CG time survives resume. In-flight work since the last durable timestamp and wrapper overhead still require external attempt accounting. The LP basis and an in-progress pricing calculation are not restored.

The six historical k32 ordinary wall-limit stops spent **6.63–24.90seconds** constructing and resolving their final LP. Two final LP attempts hit their time limits. These observations are not a guaranteed signal-exit time. Actual tiny SIGTERM/SIGUSR1 tests confirmed another post-signal LP solve and about0.208s total signal-to-exit on four trips; they do not predict full40 shutdown. This replacement preserves the signal path and reserves time around it; it does not change the algorithm to skip that final solve. Detailed evidence remains in `../advisor_sequence_20260916/checkpoint_fix/k32_final_resolve.json` and `signal_timing.json`.

## Files and live locations

- New immutable wrapper/manifest: `/home/nc437/ladder-lite/review_full40_20260916/scaglione_12h_v1/`.
- Source code and data stay under the original campaign's `code/`.
- Existing graph cache: original `cases/c1_full40/network.pkl`; native identity/hash checks remain required.
- CG attempt outputs: original `cases/c1_full40/cg/343119_r<restart>/`; a resumed attempt uses its `resume/cg.json` subdirectory. Successful completion publishes the original case-level `cg.json` link.
- Logs: original campaign `logs/dr40_12hCG_343119.out` and `.err`.
- Held MIP341406 still uses its original wrapper and MIP settings; only its dependency was rewired. Its future source status/pool are checked at execution.

`deployment_receipt.json` records current policy text/hash, code cleanliness, deployed file hashes and original job states. `submission_receipt.json` records the exact scheduler command, dependency rewire, held-MIP verification and old-CG cancellation. `validation.json` records argument/time-budget checks. `ledger_addition.json` is supplied for the parent's register/monitor update; shared ledgers and Google Docs were not edited here.

**Scientific status remains unverified:** no full40CG result, pricing certificate, integer proof, physical dispatch validation or target attainment has been produced by this scheduling action.
