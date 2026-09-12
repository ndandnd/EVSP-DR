# Fixed-sequence replay: benchmark-only indexed traversal

Exact records and realized action traces match the original implementation in every tested case. Replay speeds vary with graph size and representation; this is not evidence of a whole-CG speedup.

## Paired results

Seconds are medians of six alternating-order repetitions. Every timed call includes original record realization and physical validation. Cold replay clears action/window caches; warm replay primes them using the same method. All timing sections use an exclusive local benchmark flock. Graph generation and index setup are single cold observations, not paired repeated estimates.

| Case | Mode | Arcs | Sequences (feasible) | Baseline warm s | Indexed warm s | Replay speedup | Setup + warm batch speedup | Cold index s | Scan reduction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny_chain | lazy | 158 | 343 (15) | 0.001926 | 0.001769 | 1.09x | 1.07x | 0.000026 | 4.9x |
| tiny_chain | explicit | 158 | 343 (15) | 0.001669 | 0.001814 | 0.92x | 0.91x | 0.000022 | 4.9x |
| tiny_tight_soc | lazy | 1,218 | 33 (2) | 0.000571 | 0.000485 | 1.18x | 1.00x | 0.000084 | 2.8x |
| tiny_tight_soc | explicit | 1,218 | 33 (2) | 0.000408 | 0.000482 | 0.85x | 0.75x | 0.000059 | 2.8x |
| tiny_tariff_station_ties | lazy | 330 | 33 (2) | 0.000402 | 0.000370 | 1.09x | 1.00x | 0.000031 | 2.8x |
| tiny_tariff_station_ties | explicit | 330 | 33 (2) | 0.000323 | 0.000370 | 0.87x | 0.82x | 0.000025 | 2.8x |
| real_first_12_soc15 | lazy | 2,631 | 161 (35) | 0.005251 | 0.004276 | 1.23x | 1.18x | 0.000191 | 13.4x |
| real_first_12_soc15 | explicit | 2,631 | 161 (35) | 0.004442 | 0.004358 | 1.02x | 0.99x | 0.000138 | 13.4x |
| real_first_48_soc15 | lazy | 122,137 | 210 (118) | 0.050261 | 0.026510 | 1.90x | 1.38x | 0.009979 | 38.0x |
| real_first_48_soc15 | explicit | 122,137 | 210 (118) | 0.033925 | 0.026792 | 1.27x | 1.08x | 0.004628 | 38.0x |
| real_first_48_soc5 | lazy | 1,068,874 | 210 (118) | 0.180733 | 0.034374 | 5.26x | 1.40x | 0.095147 | 38.5x |
| real_first_48_soc2.5 | lazy | 4,225,616 | 210 (118) | 0.565551 | 0.058025 | 9.75x | 1.53x | 0.310747 | 38.8x |

## Setup and scope

The following graph-build-inclusive totals are arithmetic combinations of a single measured graph build and the median cold-cache replay batch; they are not separately observed complete execution times. Indexed totals include the full sortedness/contiguity validation pass.

| Case/mode | Build s | Baseline build + cold replay s | Indexed build + setup + cold replay s | Warm-batch setup amortization |
|---|---:|---:|---:|---:|
| real_first_12_soc15/lazy | 0.1262 | 0.1319 | 0.1312 | 1 batch(es) |
| real_first_12_soc15/explicit | 0.1221 | 0.1266 | 0.1269 | 2 batch(es) |
| real_first_48_soc15/lazy | 3.4466 | 3.5010 | 3.4875 | 1 batch(es) |
| real_first_48_soc15/explicit | 3.4340 | 3.4683 | 3.4656 | 1 batch(es) |
| real_first_48_soc5/lazy | 23.5043 | 23.6879 | 23.6397 | 1 batch(es) |
| real_first_48_soc2.5/lazy | 88.4710 | 89.0293 | 88.8426 | 1 batch(es) |

## Prototype and correctness

`benchmark_replay.py` imports the immutable pinned source and creates a standalone subclass. The original fixed_sequence_record source is copied at runtime and only its two outgoing-iterator expressions are replaced. The dynamic program, edge-list tie comparison, action lookup, record realization, cost recomputation, and physical validation stay unchanged. Binary search selects the complete half-open node-ID interval for the required successor trip (all SOC targets) or the sink. Source rows retain original order and all arcs retained by original graph construction, including its existing station-alternative deduplication/tie decision. There is no new per-arc Python index. Persistent extra storage is two bounds per trip; validation scans all nodes/arcs once and has temporary O(nodes) storage. Explicit mode bisects existing rows by target; lazy mode bisects existing packed integer arrays.

Tiny cases enumerate every sequence of lengths 0–4, including repeated/backwards trips, plus unknown IDs. Cases cover a four-trip chain; 191.2-kWh initial trip with 2.5-kWh SOC rounding; fractional arrival (59.75 minutes), an exact-hour deadline, unequal hourly tariffs, and two equal station alternatives. Every baseline/new full record and action list is compared with exact Python equality, including all rejections; serialized aggregate hashes are recorded. Real cases use the earliest 12 or 48 trips of the frozen k05/r2 input, preserving their times, energy and restricted-graph deadhead arcs. Deterministic seed 20260912+size chooses chronological sequences of lengths 2, 3, 4, 6, supplemented by all singletons and invalid sequences. These are induced subgraphs, not whole campaign instances.

The inherited-pool importer and its capped selected-sequence list were not exercised. This prototype leaves selection unchanged by operating on an immutable explicit sequence list. It makes no claim to complete WO-2 acceptance, production readiness, new route quality, CG certification, finite-pool MIP proof, or GIRO attainment. It does not compare independent physical validators or a continuous/global optimum; equivalence is to the pinned method and its existing validator.

An initial harness smoke run stopped while hashing mixed integer/string dictionary keys; canonical key normalization fixed the harness before valid measurements. No old/new correctness mismatch occurred. Subsequent exploratory timings were superseded by the final locked run.

## Physics, provenance, and reproducibility

Battery and initial energy 240 kWh; charge power 240 kW; reserve 0 kWh; terminal requirement is the same reserve; event time; 5-minute event lattice supplementation; SOC step shown in each JSON case (tiny chain 15, tight case 2.5, station-tie case 5; real cases 15, 5, or 2.5 kWh). Shared station capacity is absent. Existing objective coefficients: bus 100000, charge-start 5, tariff multiplier 1; costs are recomputed by original realization code. Tariffs: flat 0.1 for first tiny cases, specified 0.3/0.05/0.4 hourly boundary case, and peak12_alpha_1p0_h26.csv for real induced graphs. This is route replay, so no RMP master sense or dual is involved. No cluster job, importer policy, register, Google Doc, Slides, or production solver source was changed.

Pinned execution source: `a29992196acb74d02b8c7891be4061718889999f`; module SHA256 `d939fe981cb5b3f40293996e8f3a2430da8a365b392a62d6c21959316cb03de6`; harness SHA256 `8276b1ea763bc8ed19a50cc98d6d08ab3940c9492baaa4548df7871fb3ce6fc1`.

Interpreter: /Library/Frameworks/Python.framework/Versions/3.12/bin/python3; 3.12.2 (v3.12.2:6abddd9f6a, Feb  6 2024, 17:02:06) [Clang 13.0.0 (clang-1300.0.29.30)]; host macOS-26.5.2-arm64-arm-64bit; arm64. OpenBLAS/OpenMP threads: {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}. Peak RSS is process high-water RSS, not incremental graph/index memory; macOS values are bytes.

Run from any directory with the pinned checkout and local Python dependencies available:

```sh
python3 /Users/nadan/Documents/projects/demandresponse/outputs/algorithm_benchmarks_20260912/replay/benchmark_replay.py
python3 /Users/nadan/Documents/projects/demandresponse/outputs/algorithm_benchmarks_20260912/replay/make_report.py
```

`results.json` is the authoritative run: all raw replay samples, order, cold setup/build observations, full-record/action hashes, exact selected sequence lists, graph/event-lattice metrics, physics/tariff identity, input hashes, interpreter and host. Per-case JSONs and run.log are supporting files. The earlier exploratory outputs with names lacking _soc were superseded and removed. Source CSV/deadhead/tariff hashes are included in aggregate results. No long historical replay was run; the historical 42732-sequence run must not be multiplied by these speedups.
