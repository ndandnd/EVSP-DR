# Exact-CG performance audit

This work is diagnostic only.  It does not change master/pricing semantics,
defaults, stopping rules, resume identity, or live cluster jobs.

## Opt-in durable phase telemetry

Add one explicit sidecar path to an otherwise unchanged exact-CG command:

```bash
python -u src/exact_pricer_expanded.py \
  ...existing arguments... \
  --phase-telemetry /absolute/path/to/run.phases.jsonl
```

Default is off.  The telemetry path is excluded from model/resume provenance.
The append-only sidecar records network build, incidence construction, every
master attempt, exact shortest-path pricing, extra-column extraction, route
insertion, journal/iteration fsyncs, status checkpoints, snapshots, pool size,
incidence nonzeros, network nodes/arcs, and process peak RSS.  Every JSONL row
is fsynced.  On reopen, only an interrupted final row is repairable and the
session identity must match.

## Read-only frozen-pool prefix profiler

```bash
python -u src/profile_exact_pool_prefixes.py \
  --result /absolute/path/to/immutable.snapshot.json \
  --prefixes 1000,5000,10000,25000,50000 \
  --methods highs,highs-ds,highs-ipm \
  --out /separate/path/prefix_profile.json
```

The profiler reconstructs first-reached unique-incidence prefixes, builds the
incidence matrix once per prefix, and cold-solves each requested SciPy/HiGHS
method.  It records incidence time/size/nonzeros, total and backend solve time,
LP values/violations, peak RSS, and errors.  Status, journal, instance, and
tariff hashes are checked before and after; source files are never repaired or
opened for writing.

## Deterministic pricing microbenchmark

```bash
python -u src/benchmark_exact_pricing.py \
  --csv Practice_Custom_TwoDuty_13301_13302.csv \
  --prices_csv hourly_prices_flat.csv \
  --soc-step 15 --block-min 10 --columns 30 \
  --warmup 1 --repeat 3 \
  --out /separate/path/pricing_benchmark.json
```

The benchmark builds one network, prices against a fixed dual vector, and
hashes all returned route incidences, ordering, reduced costs, route nodes, and
charging realizations.  Commit B's micro-optimizations are acceptable only if
the route hash and best route remain identical case-by-case.

## Interpretation

Telemetry identifies where live CG iterations spend time.  Prefix profiling
isolates cold incidence/RMP scaling.  The pricing benchmark measures only the
expanded-network pass and existing candidate extraction.  None is a pricing
certificate or an integer-schedule result.

## Commit-B local benchmark result

`analysis/exact_cg_performance_20260814/benchmark_comparison.json` contains
the exact before/after repetitions and route hashes.

| Case | Before median | After median | Reduction | Route hash |
|---|---:|---:|---:|---|
| Two-duty pair (86 trips) | 0.05549 s | 0.04353 s | 21.6% | unchanged |
| Reproducible synthetic k8 (206 trips) | 0.11293 s | 0.08833 s | 21.8% | unchanged |

These are local microbenchmarks, not cluster throughput claims.  The synthetic
k8 input is generated with seed `20260802`, size 8, replicate 1 and is not
committed as a dataset.
