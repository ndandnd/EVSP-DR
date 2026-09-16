# Which physical constraint changes the result?

**Prepared only. No jobs submitted. Launch waits for the user's cluster-load confirmation.** This tests F4 / review item 11 using the original chain-5, k31 pool, rather than changing several assumptions together.

| Arm | Battery | Minimum SOC | PARX power | Trip-group restriction |
|---|---:|---:|---:|---|
| Control | 240 kWh | 0 | 240 kW | None |
| Depot power only | 240 kWh | 0 | **60 kW** | None |
| SOC reserve only | 240 kWh | **36 kWh (15%)** | 240 kW | None |
| Smaller battery A only | **236.44 kWh** | 0 | 240 kW | None |
| Smaller battery B only | **239.01 kWh** | 0 | 240 kW | None |
| Group separation only | 240 kWh | 0 | 240 kW | **A route cannot mix 18E1/18E2 trips** |

Every arm uses the same 716 trips, flat electricity tariff, 5-per-charge-start fee, set covering, full initial battery and free ending SOC above its reserve. Other chargers remain 240 kW. There are no shared charging-capacity limits, vehicle-class quotas or separate 65% ending-SOC target. The 15% floor necessarily applies at route end too. These are sensitivity experiments, not a complete GIRO replication.

The two battery arms are homogeneous fleets. Assigning a specific battery according to a trip's original GIRO group would also impose a routing restriction, so it would not isolate battery size. Offering both battery sizes without compatibility restrictions or quotas gives no advantage over offering the larger one in the physical model. The original combined strict campaign remains a separate experiment.

## What is reused?

The exact original MIP pool has **254,068 columns**: source status `.../chain_extension_31_32_20260915/cases/w5_k31/cg/228610_r0/cg.json`, ordered-pool SHA256 `a5a44b5c708286f914c6f6eec4680a8a7081c6015ee8e9c6f687e1c8e69eb5c6`. Full paths, source hashes, code pin and input hashes are in `manifest.json`.

`extract_sequences.py` first reproduces that identical cheapest-per-trip-incidence pool and verifies its count/hash. It then removes duplicate **ordered trip sequences**, taking order from route nodes. Each sequence retains its source pool indices and route hashes. We have inspected metadata and one journal record; the approximately 985 MB production journal has **not** been fully scanned or replayed during preparation. The number of distinct sequences remains unmeasured.

For each sequence, `replay_chunk.py` optimizes charging anew under the selected arm. It uses the frozen strict driver's transitions and the full instance's event times, while restricting service to that sequence. It does not merely replay the old charging times. The control is reoptimized too. The group-separation arm explicitly excludes mixed-group sequences.

## What the output means

Each sequence receives one outcome: feasible with physical replay; no path in this fixed-sequence event graph; excluded by group separation; or **unknown** due to timeout, error or failed physical validation. No-path certification is limited to the event representation (2.5 kWh SOC step, 5-minute event construction), not continuous-time physical infeasibility. The optimized objective is the expanded-grid route cost; continuous realized cost remains separately recorded. None of these outcomes is a full-CG pricing certificate.

`assemble.py` requires every source sequence to have a completed attempt. It retains feasible routes, records unresolved exclusions, and attempts new singleton routes only for uncovered trips. Missing physical coverage blocks the seed-ready flag. Artificial variables used later by CG are never counted as buses or physical coverage. Importing a seed with unresolved sequences requires an explicit flag and qualification.

`continue_cg.py` physically validates every imported route and prepares a checkpoint for the pinned driver; it does **not** run CG unless `--run-cg` is supplied. Group separation uses two independent class components, valid here because capacity and class availability do not couple them. A seed pool is not an integer solution, and its restricted LP value is not a full-model lower bound. New CG certificates and finite-pool MIP proofs must be reported separately.

## Proposed execution after confirmation

1. Extract once; run a deterministic 20-sequence pilot per arm, recording sequence length and charging demand. Use six independent 1-CPU/8-GB jobs; extraction gets 1 CPU/16 GB.
2. Use measured runtime and memory to choose replay chunk sizes. Initial proposal: 128 sequences/chunk, 120-second limit per sequence, 6-hour allocation, default-partition concurrency 50. Preserve unfinished attempts and retry into new directories; never silently replace an attempt.
3. Assemble all chunks, inspect outcome counts and physical coverage, then import the survivor pools. Run matched 4-hour CG arms and 1-hour two-stage MIPs (30-minute fleet minimization, remaining time minimizing charging subject to fleet **≤** the first incumbent). Exclude `scaglione-compute-01` throughout.

Full replay may be substantial: at one second per sequence, six arms and 254,068 sequences would consume about 423 CPU-hours before deduplication; at ten seconds, about 4,234 CPU-hours. The pilot determines whether that cost is justified. Report extraction, charging reoptimization, CG and MIP time separately and cumulatively. Do not compare runtime directly with the older CG driver: the pinned strict driver adds one best column per iteration, whereas the historical baseline added up to 30.

Local validation covers all six factor definitions, ordered-sequence preservation, agreement with the full tiny event graph, a charging-required example, explicit timeout/error classification, coverage fallback, and checkpoint import. No production solver run or cluster submission has been made.

## Scripts

The scripts take explicit paths and never call `sbatch`. Intended sequence: `extract_sequences.py` → `replay_chunk.py` for each arm/chunk → `assemble.py` → `continue_cg.py`. `--help` shows arguments. `prepare.py` regenerates local inputs/manifest only. Tiny checks:

```sh
python3 test_sequence_replay.py
python3 test_workflow.py
```
