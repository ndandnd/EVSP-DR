# Capacity selector benchmark — 2026-09-12

The optimization is implementable and gives measurable local gains, particularly when physical window keys repeat. It is **not production-ready**: it exactly reproduces a source pricing/returned-record cost inconsistency discovered by independent reduced-cost replay. No hard k3 workload or end-to-end CG run was measured.

## Measured timings

Six alternating reference/prototype repetitions, exclusive shared timing lock, single BLAS/OpenMP thread. Medians below include prefix construction, configuration identity construction, memo initialization and all per-call work. Imported Python modules and synthetic input generation are outside timings. Source graph build is separate.

| Workload | Reference ms | Prototype ms | Ratio reference/prototype |
|---|---:|---:|---:|
| Synthetic `single_first_use` | 0.850 | 0.981 | 0.87× |
| Synthetic `unique_12` | 24.107 | 15.893 | 1.52× |
| Synthetic `repeated_120` | 235.785 | 16.129 | 14.62× |
| Synthetic `repeated_120_x3_dual_changes` | 716.658 | 29.225 | 24.52× |
| Synthetic two-trip full pricing, cold selector | 14.813 | 8.285 | 1.79× |
| Synthetic two-trip full pricing, repeated iterations | 15.161 | 4.089 | 3.71× |

The single first-use call is slower. The repeated-window gains are dominated by reuse: 120 calls share 12 keys, giving 108 memo hits (90%). The 12 unique keys require 3,802 candidate interval queries per iteration. Across three changed-dual iterations the best-window memo is cleared each time; 12 immutable option lists are reused in each later iteration. The single-window workload uses 181 candidates and zero memo hits. Near-tie exact-sum fallback counts are recorded (706 for the first 12-key workload), so the measured implementation does not assume floating prefix subtraction preserves every tie.

The full-network fixture is synthetic, copied from the source two-trip unit fixture and run through the production graph builder, DAG pricing and physical realization: 16 nodes, 45 arcs. It is not operational instance data. Graph construction took 5.822 ms and is excluded from the full-pricing table. The repeated-network series includes its first cold selector call, followed by warm immutable-option reuse and a new dual generation on every call.

## Correctness and a failed independent gate

The 3,600 per-arc oracle comparisons had identical selected action dictionaries and zero adjusted-cost error. They cover sparse nonpositive duals, missing/zero duals, sites absent/present/unrestricted, 1/5-minute grids, fractional arrivals and durations, near-minute/hour endpoints, and 60/240-kW station power. Eight extra configuration/in-place-dual-change checks pass. In 1,000 direct row-occupancy checks, the maximum prefix sum error was 3.979e-13. Cost tolerance is 1e-8 absolute and 1e-12 relative; selected intervals and metadata must match exactly. Three expired-deadline checks cover setup, cold lookup and memo-hit entry.

Six full-network calls match the unchanged source reduced cost, trip list and entire event record exactly; the source physical validator reports valid realization. Exhaustive enumeration of all 10 DAG paths under two dual vectors also matches the priced minimum exactly.

**Failed independent master-row reduced-cost check:** returned record cost minus trip duals and occupied-row duals differs from both source/prototype pricing by 0.12000000001 and 4.80000000000 on two variable-tariff fixture cases. This is preserved in `extra_validation.json` as `master_row_replay_pass: false`. Exact agreement with the reference therefore establishes implementation equivalence only; it does not validate the current source pricing-to-master contract. This must be resolved before production adoption or certification claims. No solver certificate, finite-pool proof, GIRO attainment or experiment result is asserted.

## Isolated source record-cost mismatch

`reproduce_record_mismatch.py` is a baseline-only reproducer: it uses the unchanged production class, not the prefix prototype. `record_mismatch.json` stores all eight control cases, exact trip/capacity duals, adjusted path cost, source static arcs, selected actions and power, occupied rows, expanded/continuous record costs and returned tariff blocks.

The source `expanded_path_realization.py:762–775` event branch passes global `mapping["charge_kw"]` to `blocks_from_continuous_stops` and also uses it for realized-energy allocation, omitting the station-specific power. The source pricer correctly uses 60 kW for this station, while tariff-record construction uses the 240-kW default. A selected 60-kWh window `[61,121]` costs 59 kWh at 0.17 plus 1 kWh at 0.29; the record reallocates all 60 kWh into `[61,76]` at 0.17, losing 0.12. A selected `[100,160]` similarly loses 4.8. Record semantics explicitly say `expanded_grid_cost`; expanded and realized energy are equal in this fixture, so this is not the documented continuous/grid distinction.

Both variable-tariff/60-kW cases fail the exact capacity runner's `math.isclose(..., abs_tol=1e-5)` check at `run_capacity_speed_event_cg.py:399–409`. Both flat-tariff/60-kW controls, both variable-tariff/default-240-kW controls, and both flat/default controls pass with zero residual. The driver would reject these fixture results; this reproducer does not establish that a particular saved production run encountered this condition. Source diagnosis was independently checked by the profile-audit agent. This pre-existing failure blocks a production-readiness claim for station-specific-power/variable-tariff cases, even though the selector optimization is equivalent to the source.

## Implementation and limitations

`benchmark.py` subclasses the pinned network in memory. All capacity boundaries, duration-shifted boundaries, tariff boundaries, tariff integration and the source tuple ordering remain. Consecutive half-open occupied rows use a prefix difference with the exact source tolerance predicates. Penalties remain nonprorated. Exact original frozenset sums resolve numerically close comparisons and recompute the returned winning cost. Each normal interval query is O(1); full window enumeration is still linear in candidate count, with sorting retained and potential expensive near-tie fallback. No range-min structure is implemented.

Every pricing call invokes `begin_iteration`: copies duals, clears best-window memo, rebuilds station prefixes and compares a configuration identity covering events, tariff, station/base power mapping, default power, grid and charging-start fee. The physical-window key includes station, arrival, deadline, energy and power. Caller must not mutate configuration during an iteration. Actions retain their original metadata on memo hits. The prototype preserves source objective handling; full-network checks use combined-cost only. Prefix storage is linear in the station row span; a benchmark-only 100,000-row safety cap prevents unreasonable allocation. Deadline checks are included in generation and evaluation loops, but no checkpoint implementation is changed.

Unmeasured: actual hard-k3 saved dual/graph, real-network unique-key distribution, station span/cache memory at scale, cold import/process startup, complete CG, LP behavior, and production deadlines under long workloads. Synthetic hit rate cannot be assumed for k3. Near-tie fallback means a universal O(1) evaluation claim would be false. Six subsecond repetitions give an initial local measurement, not a robust cross-machine performance estimate.

## Provenance and reproduction

Pinned source: `253588e9b22d68fcbc67cb56bc3eb30cbb0e16b6` at `/Users/nadan/Documents/projects/demandresponse/.codex-work/capacity-timeout-checkpoint-20260911`. All imported source `.py` hashes, benchmark hash, input hashes, per-repetition timings/counters, platform/interpreter and graph physics are in `results.json`. Hardware: Apple M3 Pro; peak whole-process RSS 94.9 MiB, including imports and all fixtures, not isolated incremental memory. Python hash seed 0. Physics: 240-kWh battery, 15-kWh SOC step, 5-minute event support, 0 reserve, explicit graph, 60 kW at first station and 240 kW default, combined-cost objective, synthetic varying hourly tariffs. No cluster access, submissions, git mutations, production edits, document/register writes or existing artifacts were performed.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 python3 outputs/algorithm_benchmarks_20260912/capacity/benchmark.py
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 python3 outputs/algorithm_benchmarks_20260912/capacity/validate_extra.py
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 python3 outputs/algorithm_benchmarks_20260912/capacity/reproduce_record_mismatch.py
```

Files: `benchmark.py`, `results.json`, `validate_extra.py`, `extra_validation.json`, `REPORT.md`, `reproduce_record_mismatch.py`, `record_mismatch.json`, `SHA256SUMS`. Reruns overwrite this isolated benchmark’s JSON results.
