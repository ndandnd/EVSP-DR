# F4 — depot rate and GIRO replay

**P0 item 3 completed.** This audit distinguishes copying GIRO's original charging schedule from keeping its trips and finding a new charging schedule.

| Finding tested | Verdict | Evidence |
|---|---|---|
| Baseline chain code gives PARX 240 kW, rather than GIRO's stated 60 kW depot limit | **VERIFIED** | Execution commit `a0e0bb7681c8451e3cbbbfa06aef390026d9af4b` uses one `charge_kw` for every station. Campaign manifests set it to 240. |
| That baseline code also models KEX at 240 kW | **REFUTED** | KEX is absent from the Partille station list. Its rate is not represented by this campaign. |
| Chain baseline has no 15% reserve or group-specific batteries | **VERIFIED** | Manifest records 240 kWh, initial 240, reserve 0, and homogeneous physics. Source requirements remain separately indexed in `outputs/model_fairness_audit_20260913/giro_requirements_audit.md`. |
| “12 of 40 original GIRO schedules fail replay” describes the chain baseline | **REFUTED** | That earlier screen was **240 kWh / 350 kW**, and its 28/40 passes reproduce exactly. Under chain **240/240**, 0/40 pass unchanged. Including both service-day variants gives 0/42. |
| These replay failures mean the GIRO trip sequences themselves cannot be covered with one bus each under baseline physics | **REFUTED for the continuous model** | Fresh fixed-trip charging optimization and full physical replay succeeds for **42/42** sequences. Every chain's k duties therefore supplies a continuous physical k-bus schedule under the stated relaxed physics. |
| The feasible continuous schedules are representable on the production 2.5-kWh/5-minute event lattice | **UNRESOLVED by this test** | No event-lattice route construction was attempted. Continuous feasibility is a distinct claim. |
| Raising depot power explains LP=k−1 in particular chains | **UNRESOLVED** | The code difference is verified; its causal effect requires a matched rerun. |

The unchanged-original test does not repair charging windows, clip energy, or change charge amounts. At 240 kW, **39/42** duties violate at least one recorded power/window limit. The remaining three are rejected for small over-capacity discrepancies in recorded energy and/or production deadhead accounting. For example duty 13307 passes recorded-activity replay but its production replay reaches 240.4 kWh at PARX against a 240-kWh cap. These are checks of the copied schedule, not proofs that no alternative charging plan exists.

The 42 independent fixed-trip optimizations took **4.34 seconds total** locally. All returned feasible, replay-validated, certified fixed-duty results. Their objective values match the old 42-row certificate ledger within **4.9e−10**. Every returned schedule and replay is retained in `optimized_fixed_duties/`; these are new witnesses, not merely copied summary flags. They use 240-kWh battery, uniform 240-kW charging, reserve 0, free ending SOC, no shared capacity and a charge-start fee of 5. Their optimality scope is charging on that fixed trip sequence.

## Files

- `chain_replay_counts.csv`: one row for every frozen chain prefix k=2..40; original schedule counts, reoptimized continuous counts, and the actual duty variants. Values above k=32 are **input-membership audits**, not completed CG experiments. Every row has 0 unchanged-production passes and k reoptimized continuous passes.
- `duty_replay.json` / `.csv`: all 42 duty variants, at both 240 and 350 kW, with exact failure reasons.
- `prior350_reproduction.json`: exact old/new source and production replay classifications for the prior canonical 40-duty cohort.
- `fixed_duty_rerun.json`: new 42-duty solver/replay summary and result hashes.
- `optimized_fixed_duties/*.json`: full new schedules, energy traces, certificates and runtime.
- `provenance.json`: source hashes, execution pin and replay scope.
- `pinned_charge_rate_trace.txt`: positive call-path evidence.

The source master is the campaign's frozen copy (`outputs/chain_extension_20260913/inputs/sources/Par_VehicleDetails_Updated.csv`, SHA-256 `6b46acce8b0870aff967c73aac372b90873ed32a6e424e55b851e4b8676ab57f`). The workspace's top-level `data/Par_VehicleDetails_Updated.csv` has different bytes and was deliberately not substituted. Each chain retains its actual 13316/13324 service-day variants, rather than using the earlier screen's generic 40-duty selection.

## Code trace and reproduction

Pinned `src/config.py:41` includes PARX and excludes KEX. `EventPricingNetwork.__init__` assigns the scalar power; event arcs and physical realization receive that same scalar (`src/event_pricer_network.py` around lines 168, 506, 703, 790, 810). `src/expanded_path_realization.py` converts energy to duration using `charge_kw` (around lines 305, 555, 559). The manifest supplies 240. Constants of 300 in config defaults do not override that execution argument.

Extract pinned `src/` into this directory's `pinned/` using `git archive a0e0bb76 src`. `compare_original_giro_charging.py` is preserved from the `codex/matched-fleet-tariff-20260908` Git object; its hash is in provenance. Run `python3 audit.py`, then `python3 reoptimize_fixed_duties.py`. The scripts only write here. The `pinned/` source cache and generated `duty_inputs/` are reproducible working files; the execution commit is authoritative. To refresh the combined chain flags after rerunning, run `python3 finalize.py`.
