# Six-chain input extension, 13 September 2026

`python3 outputs/chain_extension_20260913/prepare_inputs.py` reproduces the frozen inputs using only Python's standard library. The script refuses to replace an existing CSV with different bytes. It creates 150 cases, k16–40 for six chains; `inputs/manifest.json` marks only the 60 k16–25 cases as staged. k26–40 are prepared membership for later continuation, not job submissions. The launcher records its own execution commit and resource policy separately.

## Parent preservation and selection

The six exact k15 CSVs, original `chain_order.csv`, `selection_manifest.csv`, `input_plan.json`, continuous-duty evidence, source master, and historical builder were copied read-only from `/home/nc437/ladder-lite/full_pool_recovery_20260912/code`. Copies are retained under `inputs/sources/`. The successful campaign manifest and dated integer-result table are also frozen there. All six parent hashes match both successful campaign and result records. Original membership and addition order match all k15 trip IDs and every field except the merged positional `count_trip_id`.

The master has 42 duty variants across 40 numeric base duties. The two variant groups are `13316m/13316uwt` and `13324muw/13324t`. The variant policy creates an **inferred day-compatible scenario**, based on a nonempty intersection of suffix characters. No supplied weekday legend was located or assumed; the labels are not expanded to weekday names. All parent variants are preserved. Where multiple pairs remain possible, select the lexicographically first compatible pair and freeze it for that chain:

| Chain | Frozen 13316 variant | Frozen 13324 variant | Pair uniquely implied by parent? |
|---|---|---|---|
| C1 | 13316m | 13324muw | No |
| C2 | 13316uwt | 13324t | Yes |
| C3 | 13316uwt | 13324t | Yes |
| C4 | 13316m | 13324muw | No |
| C5 | 13316uwt | 13324muw | No |
| C6 | 13316uwt | 13324t | Yes |

For each chain, sort the 25 unused numeric base duties, then shuffle with `random.Random(20260913 + chain)`. Append exactly one duty per k using its frozen variant. This order is selected without solver outcomes. C1/C4, C2/C3/C6, and C5 consequently converge to three distinct k40 variant scenarios. These are extensions of six existing successful chains, not six newly independent experimental samples.

## Validation and schema

Every emitted case asserts exactly k unique base duties, globally unique stable `Ordered_Trip_ID` values, strict nesting from the immediately previous case, no alternate variant of an already selected duty, and unchanged prior trip attributes. Rows use the original event-solver nine-column schema and chronological `(Start1 minutes, Ordered_Trip_ID)` ordering. Only `count_trip_id`, the positional merged row index, is recomputed. Stable trip IDs are never renumbered. Each added duty includes all its Regular trips from the frozen master.

`inputs/manifest.json` records source hashes, preparation source revision and script hash, exact parent paths and hashes, prior CG payload hashes, original order, candidate universe, ambiguous variant options, resolved pairs, random seeds/orders, per-case additions and hashes, and previous-case dependencies. `inputs/membership.csv` is the compact addition ledger. `inputs/SHA256SUMS` covers all frozen and generated files. The archived historical continuous-duty evidence establishes no new event-lattice certificate. Input preparation claims no CG result, fleet proof, physical validation, or GIRO target attainment.
