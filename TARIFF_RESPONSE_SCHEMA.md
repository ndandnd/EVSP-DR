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
- `TIER2_RAW_ROUTE_CHARGING` and
  `TIER2_GIRO40_AUGMENTED_ROUTE_CHARGING` permit route replacement through
  exact expanded-network pricing. RAW and augmented journals are distinct
  finite column pools. Tier-1 seed costs are recomputed and hash-bound to every
  tariff before injection.

No tier claims that the continuously replayed charging cost is pricing-optimal.

## Cost decomposition

For matching instance and tariff identities:

```text
charging_only_savings = Tier0 - Tier1
rerouting_increment = Tier1 - Tier2_GIRO40_AUGMENTED
total_price_aware_savings = Tier0 - Tier2_GIRO40_AUGMENTED
```

Each formula is calculated separately for expanded-grid and continuous-replay
costs. If any term is unavailable, the result remains null.

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
