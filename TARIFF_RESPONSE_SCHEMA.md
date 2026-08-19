# Tariff-response experiment schema

## Scientific tiers

- `TIER0_GIRO_ORIGINAL` preserves the selected 40 literal GIRO duties and all
  recorded recharge rows. It performs no optimization. A scalar tariff cost is
  available only when each recorded window has complete tariff coverage and
  one unambiguous price; no uniform-power split or hour fallback is invented.
- `TIER1_FIXED_GIRO_OPTIMIZED_CHARGING` fixes each ordered trip sequence and
  solves an exhaustive acyclic dynamic program over 15 kWh SOC levels,
  10-minute time blocks and modeled stations. Its certificate proves the
  minimum expanded-grid cost for that sequence under 300 kWh, 300 kW,
  zero reserve, and depot-arrival SOC at least reserve.
- `TIER2_RAW_ROUTE_CHARGING`,
  `TIER2_GIRO_AUGMENTED_ROUTE_CHARGING` (k5/k8), and the separately gated
  `TIER2_GIRO40_AUGMENTED_ROUTE_CHARGING` (future k40 MIP) permit route
  replacement through exact expanded-network pricing. RAW and augmented
  journals are distinct finite column pools. Tier-1 seed costs are recomputed
  and hash-bound to every tariff before injection. This pilot prepares k40 CG
  pools but deliberately contains no k40 MIP.

No tier claims that the continuously replayed charging cost is pricing-optimal.

The required `alpha=2` extrapolation contains explicitly labeled negative
prices. The manifest policy is `allow_feasible_consumption_no_export`; extra
energy consumption is therefore a modeled response, not energy resale.
Terminal SOC minima/maxima and charged kWh are mandatory outputs so this
behavior cannot be hidden inside a cost comparison. Because no common terminal
SOC equality or terminal-energy salvage value is available, alpha=2 is labeled
`negative_price_stress`, excluded from all primary savings and elasticity
tables/curves, and rendered only in a separate stress figure.

## Cost decomposition

For matching instance and tariff identities:

```text
charging_only_savings = Tier0 - Tier1
rerouting_increment = Tier1 - matching Tier2 augmented treatment
total_price_aware_savings = Tier0 - matching Tier2 augmented treatment
```

Each formula is calculated separately for expanded-grid and continuous-replay
costs. It is emitted only when all three tiers use the same fleet count and
terminal policy; fleet changes are reported separately. If any term is
unavailable, the result remains null.

## Route response

Route comparisons never use bus labels or MIP column indices. Duties are
matched by maximum trip overlap. Reported metrics include changed duty
assignment, predecessor/successor changes, trip-adjacency and co-assignment
Jaccard similarities, intact duties, split/merged duties, and retained/new
columns. These are called route response or route sensitivity, not elasticity.

Elasticity is reserved for adjacent positive price-amplitude levels and
quantities such as peak-window kWh or charging cost.

## Real-data limitation

The selected source contains 344 recorded recharge windows. Seventy-five cross
an hourly tariff boundary without a within-window energy trace, two extend
beyond hour 24 where most pilot tariffs stop, and 86 imply more than 300 kW
after whole-minute rounding. Tier 0 therefore preserves the decisions but
marks affected tariff costs, hourly allocations, terminal-policy comparability
and physical replay unavailable. The pipeline must not repair or infer them.

## Required outputs

`tariff_manifest.csv`, `giro40_duty_manifest.csv`,
`charging_blocks_long.csv`, `tariff_response_summary.csv`,
`route_change_summary.csv`, `fixed_duty_certificate_summary.csv`,
`cg_iteration_long.csv`, `mip_checkpoint_long.csv`,
`artifact_inventory.csv`, `data_dictionary.csv`, and `provenance.json` are
immutable and hash-indexed. Figures are accompanied by their plotting CSVs.

Cluster evidence is accepted only with the v2 worker-completion identity:
plan SHA, job key, campaign-independent execution digest, phase/treatment/
analysis role, scale, tariff and instance identities, numeric Slurm job ID,
and the exact artifact hash map. Scheduler receipts and worker completions are
separate required provenance layers.

`main_k5_k8_pilot` and `k40_preparation_only` are separate scheduler and
reservation identity domains even though they share one plan SHA. Gate names,
comments, roles, receipts, reservation paths/transactions, worker
completions, reconciliation, and consumers all bind the exact scope.

The current matrix is planning-only: deterministic primary-grid membership
preflight records exact nonrepresentable duties/reasons and sets
`submission_blocked=true`. No consumer may interpret the planned 111 main
outputs as available evidence.
