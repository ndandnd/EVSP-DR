# Event graph startup optimization — 12 September 2026

This package validates a minimal optimization to `EventExpandedNetwork` at
commit `28681fa108cd89d25abe50aa0e425dbb4fb8caf6`, based directly on source
`a29992196acb74d02b8c7891be4061718889999f`.

The original builder serialized every candidate action with
`json.dumps(..., sort_keys=True)` before it knew whether cost alone decided the
retained arc. The implementation now serializes actions only when two
candidates for the same `(target, dual)` have equal cost. Final arc sorting
likewise serializes only an exact `(target, cost)` tie group. Candidate methods
allocate a new scalar-valued action dictionary for each yield and never mutate
it afterward, so deferring its canonical representation is safe.

The winner rule remains exactly `(cost, canonical action JSON)`. Adversarial
tests cover cheaper, dearer and equal-cost candidates and lexicographically
earlier and later actions. A complete graph comparison using the first 12 real
`d00_g3` trips checks both explicit and production lazy modes: node metadata,
topological order, every explicit arc/action or packed target/cost/recipe,
slice boundaries, sink arcs, metrics and a realized reduced-cost route are
identical to an embedded copy of the eager implementation.

The focused regression command completed 18 tests in 61.329 seconds:
`python3 -m unittest tests.test_event_pricer_tie_keys
tests.test_event_pricer_network tests.test_event_pricer_gates`.

## Bounded measurement

The local benchmark used the first 24 of 325 `d00_g3` trips, retaining all
induced trip, station and depot arcs. Its production settings were event time,
lazy packed arcs, 2.5-kWh SOC, 5-minute blocks, 240-kWh battery, 240-kW charge
power, zero reserve and the flat tariff. The completed graph has 2,158 nodes
and 492,090 arcs.

Three alternating unprofiled builds measured:

| Builder | Samples (s) | Median (s) |
|---|---:|---:|
| Original eager keys | 11.417, 11.466, 11.937 | 11.466 |
| Deferred keys | 4.650, 4.611, 6.545 | 4.650 |

The observed median improvement is **2.47×**, or **59.45% less constructor
time**. Canonical JSON calls fell from 2,201,750 to 368,254 (**83.27%**).
Synchronous cProfile measured 17.016 seconds for the eager oracle, including
9.599 cumulative seconds in JSON encoding, and 8.325 seconds for the deferred
builder, including 1.617 cumulative seconds in JSON encoding. No asynchronous
timer or background sampler was used.

These timings are a bounded local diagnostic on an Apple M3 Pro with Python
3.12.2. They do not measure a full 325-trip `d00_g3` build, cluster behavior,
CG, an LP certificate or a MIP speedup. The raw measurements and all source
input hashes are in `benchmark.json`.

## Launch recommendation

Cherry-pick `28681fa108cd89d25abe50aa0e425dbb4fb8caf6` onto the independently
validated execution/recovery pin, then build fresh caches under that exact
combined commit. Prepare exactly one cache for `d00_g3` and one cache for the
shared 32-duty parent graph. The ten parent decomposition runs must all consume
the same completed parent cache with `--event-network-cache-mode require`.
Give each cache preparation its own explicitly extended construction budget,
retain the original 14,400-second `d00_g3` result as censored, and make every CG
consumer depend on successful cache completion. Downstream MIPs must retain
their true previous-k/valid-CG dependencies.

For both CPU-only cache jobs, use the default partition, two CPUs and 32 GiB as
in the reviewed diagnostic, exclude `scaglione-compute-01`, and do not use the
failed asynchronous sampler. Record ordinary completed-source progress and
synchronous profiling only if further diagnostics are needed. This package did
not submit, cancel or alter any cluster job.
