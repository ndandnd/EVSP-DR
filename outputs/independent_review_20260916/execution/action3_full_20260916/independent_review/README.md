# Independent review of six-arm physical sensitivities

Review scope: F4 attribution to one physical factor at a time, compact graph equivalence, cache identity, complete original-pool coverage and replay failure classification. No jobs submitted and no production code edited by this reviewer. A separate atomic pool-copy helper was contributed locally under explicit coordination with the implementation owner.

## Mathematical changes — verified within the stated scope

With a constant station tariff, charging `e` kWh at power `p` costs the same at every feasible start time. The earliest start is therefore an optimal representative whenever `arrival + 60e/p <= deadline`. The shortcut retains the charge-start fee in the route arc and uses the existing charging-cost multiplier. It does not remove battery, reserve, travel or charging-time constraints.

Without shared station capacity or other constraints coupling charging decisions across routes, a more expensive transition to the same next-trip/SOC state is dominated. The compact graph retains a cheapest transition. It preserves achievable ordered trip sequences and their cheapest charging objective; it does **not** retain every alternative station or charging schedule. Compact mode explicitly rejects capacity-coupled arms.

Independent tests passed against original charge-window enumeration:

- 600 charging-window cases, including both power levels, infeasible windows, zero and negative flat prices.
- 35 fixed-sequence cases across control, depot60, reserve36, battery236.44 and battery239.01; original explicit, shortcut explicit and compact graphs agreed.
- 225 pricing comparisons with varied trip duals and combined-cost, fleet-only and charging-cost objectives agreed.
- Five group-subset comparisons agreed despite changed event sets; no capacity coupling and flat tariffs are essential to this interpretation.
- Compact cache round trips retained five route results. Fourteen altered input/model identity fields were rejected; corrupted cache bytes were rejected before unpickling; existing caches could not be overwritten.

Tests and source hashes are in `equivalence_tests.json` and `cache_tests.json`. These are bounded equivalence checks, not a proof of production convergence or final fleet attainment.

## Single-factor design

All arms keep 716 trips, the flat tariff, 5-per-charge-start fee, covering master, full initial battery, free ending SOC above reserve and no shared station capacity. The reserve-only arm uses a 36-kWh floor with a 240-kWh battery. Each smaller-battery arm is a homogeneous fleet with zero reserve; these are not vehicle-type assignments. Group separation changes only which trips may coexist in a route, with two independent components. Because no station capacity or fleet-type quota couples these components, their fleet minima can be added; full-model claims require both components' certificates.

The independent replay receipt from the completed pilot preprocessing confirms **254,068 original pool columns and 254,068 unique ordered sequences**, full-source scope and sequence hash `cdaae9f1f92fee3d2fac216917ca3a99478111e5c5c3a8ea7d9c9d6d843a69a4`. Proposed chunks of 2,048 produce 125 shards (124 full shards and a 116-sequence tail), hence 750 arm/shard tasks. Array mapping `arm = task % 6`, `shard = task // 6` covers every arm/shard pair exactly once for tasks 0–749.

A timeout, exception or failed physical replay remains **unknown**, not physical infeasibility. Exhausted fixed-sequence event search proves no path only in that representation. Assembly must account for every source sequence, including structural exclusions and unknowns, and must verify trip coverage. A prepared survivor pool is neither a fleet solution nor a full-model lower bound.

## Operational review — replay worker approved after correction

Initial replay-worker review found recovery defects and reported them before launch: interior journal corruption was silently truncated; valid final JSON without a newline was not repaired before appending; completion files were non-atomic and checked only for existence; the group-map input hash was not checked. The implementation owner corrected these before submission. Eleven independent two-record restart/completion checks passed, including fail-closed handling of interior corruption, checksum mismatch, wrong source order, malformed completion and changed output hashes. The corrected worker checks the group-map hash. `recovery_tests.json` binds the exact reviewed worker/helper/sharding hashes. Replay sign-off was followed by the final continuation check described below.

Graph cache workers must publish a validated completed cache from a unique attempt directory. A killed unpublished build has no internal checkpoint and must rebuild; the cache writer deliberately refuses to overwrite a partial existing canonical path. Five-minute CG pool checkpointing is checked after completed iterations, so a long pricing/LP call can lose more than five minutes. It is not a strict bound on lost work.

Group input mapping was independently checked against every source row: 172 trips in 18E1 and 544 in 18E2, disjoint and covering all 716. CSV `count_trip_id` remains the original identifier; the native problem builder resets dataframe indices and uses consecutive local indices. The manifest remapping matches that runtime convention. See `group_mapping_check.json`.

A coordinated `atomic_pool_copy.py` helper now publishes only a complete, fsynced, hash-matched private pool copy. Four tests passed: complete copying, preservation of an existing destination, interruption before publication and source mutation during copying. The native strict driver still validates every resumed pool record and fails closed on malformed records or mismatched identities; it does not silently omit a damaged route. See `atomic_pool_copy_tests.json`.

## Final continuation review — approved for the recorded hashes

`continuation_signoff.json` verifies the frozen replay/prepared manifests, code/tooling hashes, every case input and all dependency mappings. Seven graph-cache stages and seven CG/MIP components serve six arms; segregation shares one assembly and splits into 172/544-trip components. Each assembly waits on its own 125 replay tasks. Each CG waits on its own graph and assembly; each MIP waits on its CG.

Nominal budgets sum to 14,400 CG seconds and 3,600 MIP seconds **per arm**. Segregation splits these into 3,459 + 10,941 CG seconds and 865 + 2,735 MIP seconds. Graph construction has a separate 40-hour allocation. CG budgets reset on requeue, so comparisons must include all attempts; these are not guaranteed four-hour total-computation limits. The implementation records that qualification.

The final stage worker publishes an atomic cache JSON pointer to complete attempt files, avoiding partial publication of two canonical symlinks. Assembly likewise uses a unique output directory and a ready receipt. Copied CG pools are published atomically and remain subject to the native identity and physical replay checks. Completed-stage paths are checked before reuse. The final integration smoke is performed separately by the implementation owner; this reviewer did not submit production jobs.

F4 scientific conclusions remain **UNRESOLVED** until full replay and subsequent solver outputs arrive. A passing implementation review does not verify which physical constraint explains the observed fleet gap. Baseline replay failures must be flagged as anomalies, not attributed to tightened physics. Smaller-battery effects are measured in the fixed 2.5-kWh conservative SOC model, so sub-grid battery differences should not be described as a pure continuous-physics sensitivity.
