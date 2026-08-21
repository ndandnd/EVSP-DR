# Provisional findings: `cursor/event-based-pricer-2969`

These labels are branch-local only. They are not authoritative bug or decision
IDs; a records curator may assign canonical identifiers.

## LOCAL-1 — Coarse SOC does not preserve the event result

Claim: Event timing alone is insufficient at coarse SOC across the three k02
cells. At 15 kWh only k02_s2 reaches 2.0; at 10 kWh k02_s2 is 2.090909 and
k02_s3 is 2.081081. All three reach 2.0 at 2.5 kWh.

Evidence: `analysis/event_based_pricer_v2_20260821/coarse_soc_evidence.csv`

Producing commit: `92dd6c1d46d49de304f80e69e3a3fc955a8db462`

## LOCAL-2 — An opt-in event realization contract closes the alignment gap

Claim: Irregular event windows can be replayed without snapping while
preserving continuous timing, SOC, reserve, capacity, charger power,
tariff-hour validation, route nodes, trip order, and unchanged master cost
semantics. The corrected strict audits reject zero of 15,882 k02_s3 records and
zero of 16,253 k05_s2 records.

Evidence:

- `analysis/event_based_pricer_v2_20260821/artifacts/k02_s3.physical.json`
- `analysis/event_based_pricer_v2_20260821/artifacts/k05_s2.physical.json`

Producing commit: `f7baa6221f2ab87074c57964c7f1de4f13d7900a`

## LOCAL-3 — Packed factorized arcs clear the stated memory gates

Claim: Packed target/cost/recipe arrays with lazy physical-action
reconstruction preserve the logical event graph while avoiding per-arc Python
objects. k02_s3 certifies in 557.18 s at 671.07 MiB peak RSS. The frozen k05_s2
canary certifies in 1,013.33 s at 1,069.18 MiB, versus 26,814.64 MiB for the
prior explicit 22,161,911-arc build.

Evidence:

- `analysis/event_based_pricer_v2_20260821/results.csv`
- `analysis/event_based_pricer_v2_20260821/artifacts/k02_s3.status.json`
- `analysis/event_based_pricer_v2_20260821/artifacts/k05_s2.status.json`

Producing commits:

- k02_s3: `ba9c17c8bb1ba0604bebe6f8f9e0291c1971140d`
- k05_s2: `f0b482cf0390ae51ea091871785e70e38ff68671`

## LOCAL-4 — Sink batching is not k-shortest-path enumeration

Claim: The enrichment pass returns the exact minimum-reduced-cost route plus
at most one best prefix per sink predecessor. Its public name is now
`sink_predecessor_route_batch`; documentation and CLI help state that it is a
heuristic, not k-shortest-path enumeration.

Evidence:

- `src/exact_pricer_expanded.py`
- `src/event_pricer_network.py`
- `analysis/event_based_pricer_v2_20260821/REPORT.md`

Producing commit: `eee73d312b99b6dc5f675d39ae2fd104cb3d7880`

## LOCAL-5 — Event tariff fallback needed one identity policy

Claim: The first k5 audit rejected 52 late-horizon records because generation
labeled fallback-price blocks with requested hour 25 while replay labeled the
same price with source hour 24. Shared event tariff normalization through the
full horizon removes the identity discrepancy; the corrected audit accepts all
16,253 records.

Evidence:

- `analysis/event_based_pricer_v2_20260821/artifacts/k05_s2.physical.pre_normalization.json`
- `analysis/event_based_pricer_v2_20260821/artifacts/k05_s2.physical.json`

Producing commit: `f7baa6221f2ab87074c57964c7f1de4f13d7900a`
