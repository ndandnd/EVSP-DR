# Verified update — 14 September, collection completed 14:59 EDT

The warm-start comparison has 25 certified CG endpoints out of 36. Six MIPs are published; five match their target. The first informative contrast is chain 5 at target 8: seeds from the previous integer solution produce 8 buses, while seeds selected by largest LP weight produce 9. Both fleet minima are proved within their respective pools and both CGs certify the same weighted LP objective (800,431.772270). Each method selected 7 prior sequences, covering 156 versus 100 parent trips. Thus this compares selection strategies; it does not isolate integrality from coverage.

Integer-route seeds also match 10 buses on chains 5 and 6. Their LP-weight MIPs are pending. Chain 6 target 8 matches under both seed methods.

Original larger-chain results now include 55 CG endpoints (45 certified, 10 capped) and 53 one-hour MIPs (31 target matches, 22 misses). Separate longer searches recover 15 original misses, leaving 7 unresolved. New caps are C1k23 and C6k25; their last reported minimum reduced costs are −0.073378 and −0.063090. C4k24 MIP finds 25 buses with bound 24 and no fleet proof. Fresh C1k15 with 200 columns per iteration finds 18 buses after certified 765-minute CG and a one-hour MIP; bound 15 remains open.

The collector succeeded in 264seconds. No new execution failure, confirmed preemption or invalid dependency was observed. Its queue contains 105 running EVSP allocations and 72 dependency waits (including 18 operational graph gates); held historical jobs remain excluded and unchanged. No new campaigns, retries, source solver edits or scheduler mutations were made during this check.

Register: 2,935 records across 63 source groups, with all 6 existing supplements retained. These are artifact/stage records, not independent experimental samples. Full source snapshot SHA-256: 167a7198a772d21580b9384d290741bd6d343f2649399c9a61c70ff60d314757.

[Seed and pool-union results](README.md). [Decomposition treatments](DECOMPOSITION_STATUS.md). [Current chain results](../../cumulative_budget_20260913/status_20260914T185520Z/README.md).
