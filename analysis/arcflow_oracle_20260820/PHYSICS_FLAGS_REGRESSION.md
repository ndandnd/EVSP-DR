# Arc-flow physics flags regression

Date: 2026-08-21. Producing code commit:
`2715b90b78f343b16d705031f47e4e2543cb463f`.

All commands ran locally. No cluster jobs were submitted.

## Configuration contract

`src/arcflow_oracle.py` now accepts:

- `--g-kwh` (default `300`)
- `--charge-kw` (default `300`)
- `--min-soc-frac` (default `0`)
- `--soc-step`
- `--block-min`

New grids must divide battery capacity and the 1560-minute horizon, and one
charging block must add an integer number of SOC steps. The historical
300 kWh / 300 kW / 15 kWh / 10 minute configuration remains an explicit
compatibility exception because its 50 kWh block is intentionally floored to
45 kWh by the existing model.

## Required legacy regression first

Instance: `k02_s2`, 300 kWh / 300 kW / reserve 0, 15 kWh / 10 minutes.

| solve | prior | with explicit physics flags | result |
|---|---:|---:|---|
| fleet LP | 2.1875 | 2.1875 | identical |
| integer fleet | 3 | 3 | identical |
| integer objective | 300070.592 | 300070.592 | identical |
| integer charging cost | 70.592 | 70.592 | identical |

The deterministic scientific projections of the old and new JSON artifacts
are byte-identical:

- LP SHA256:
  `a3b75e37e2b8933b8149bd51a414853794b504604b96720ae50974859b013d66`
- Integer SHA256:
  `bff53b87db0a39a42e93345d9b921f713c63fb417125646160ba40d5cadbb3be`

The projection contains instance, grid, network dimensions, every solve field
except elapsed runtime, and all decomposed routes. Full files necessarily
differ because `solve_s` is nondeterministic and the new artifact records
physics and journal-identity provenance.

Both G1 and G2 passed. The integer result was optimal, every returned arc was
integral, and G4 replayed three routes covering all 29 trips exactly once.

## Frozen-physics smoke

Instance: `k02_s2`, 240 kWh / 240 kW / reserve 0, 10 kWh / 10 minutes.

- G1: 23,982 nodes and 306,963 arcs matched the exact pricer; 162,685 arcs
  survived exact source/sink reachability presolve.
- G2: a charged 14-trip journal route mapped to 29 oracle arcs at the identical
  stored expanded-grid cost of 100,070.552.
- Fleet LP: **2.4**, optimal in 7.288 seconds.
- Integer witness: **3 buses**, objective 300,110.352, with every arc integral.
- G4: three physically replayed routes covered all 29 trips exactly once.
- The LP lower bound `2.4` and three-bus integral witness prove the frozen
  discretized model's integer fleet optimum is **3** for this smoke instance.
