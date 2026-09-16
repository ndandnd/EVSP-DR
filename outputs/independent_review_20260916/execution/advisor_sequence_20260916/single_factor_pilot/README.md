# Single-factor charging replay: timing pilot

**Control job 342321 passed; the five 20-sequence sensitivity pilots are now submitted.** No full replay, new CG or MIP is authorized by this campaign.

| Control result | Measured value |
|---|---:|
| Physically validated feasible sequences | 20/20 |
| Timeouts / errors / unknowns | 0 |
| Source-pool extraction and validation | 70.24 seconds |
| Charging replay, including common graph/data setup | 43.66 seconds |
| Total worker / scheduler elapsed | 118.22 / 119 seconds |
| Extraction peak RSS | 4,440,848 KiB (4.24 GiB) |
| Replay peak RSS | 192,732 KiB (0.184 GiB) |
| Scheduler sampled MaxRSS | 4,432,852 KiB |
| Trips appearing in feasible pilot sequences | 347 of 716; not full coverage |

Source and pilot artifacts are saved locally under `control/`. F4 control fixed-sequence feasibility for the sampled 20 is **VERIFIED**; full-pool and fleet conclusions remain unresolved.

| Sensitivity pilot | Job |
|---|---:|
| Depot 60 kW only | 342380 |
| 15% SOC reserve only | 342381 |
| 236.44 kWh battery only | 342382 |
| 239.01 kWh battery only | 342383 |
| Group separation only | 342384 |

The `pass` field is the **control-to-next-five authorization gate**, not an acceptance test for the physical sensitivity arms. For other arms, inspect `outcome_counts`, unknowns, resource metrics and worker status; stricter physics can legitimately make a sequence infeasible or structurally excluded.

| Control job | Request |
|---|---|
| Partition | `default_partition` |
| CPU / RAM | 1 CPU / 16 GB |
| Allocation | 90 minutes; no automatic requeue |
| Excluded node | `scaglione-compute-01` (verified in scheduler receipt) |
| Initial state | Pending for priority; no dependency |

The job reconstructs the **same 254,068-column chain-5 k31 pool**, checks its source/ordered-pool hashes, and deduplicates chronological trip sequences. It then chooses 20 deterministic ranks across sequence length and original charge-start counts. This avoids testing only the early singleton columns. Charging is reoptimized under the unchanged 240-kWh/240-kW control with the full instance's event times. The pool and selected sequences have 716 possible source trips; **the pilot's trip coverage is not full-instance coverage**.

The gate for the next five pilots requires all 20 control sequences to be physically replayed and feasible, no timeout/error/validation-unknown outcomes, verified sampling/source hashes, successful scheduler completion, each measured stage's peak memory below 12.8 GiB (80% of allocation), and elapsed time below 75 minutes. A no-path result is limited to the fixed-sequence event model. It fails this control gate; it is not silently discarded.

If the gate passes, `submit.py --remaining` may submit exactly the other five **20-sequence pilots**: depot60, reserve15%, battery236.44, battery239.01, and segregation-only. They reuse the identical selected sequences. No automatic continuation into a full replay is present. Gated submission is idempotent and records unresolved intents before scheduler calls.

Source definitions and ten local preparation tests are preserved in `prepared/`. `pilot_manifest.json` records authorization, resources and acceptance thresholds. `jobs.json` holds submission receipts. Native records:

- Campaign: `/home/nc437/ladder-lite/advisor_sequence_20260916/single_factor_pilot`
- Data/attempts: `/share/scaglione/nc437/evsp-dr/advisor_sequence_20260916/single_factor_pilot/runs/<arm>/<job>_r0/`
- Successful first-job evidence: `control_pass.json`; per-attempt `pilot_result.json`, `execution.json`, stage timing/RSS logs, and replay outcomes.

F4 scientific sensitivity results remain unresolved until pilot outputs are inspected; a successful timing pilot alone does not establish a fleet-size result.
