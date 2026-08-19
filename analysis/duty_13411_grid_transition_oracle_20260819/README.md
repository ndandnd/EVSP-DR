# Duty 13411 grid-transition oracle

Read-only post-hoc current-code diagnostic. Counterfactuals are not feasibility or pricing certificates and do not change production physics or the running ladder.

## Grid outcomes

- 15 kWh/10 min: local 46→53, ordered 106→119, cause `interaction`
- 5 kWh/10 min: local 46→53, ordered 106→119, cause `interaction`
- 2.5 kWh/10 min: local 53→59, ordered 119→132, cause `accumulated SOC flooring`
- 1 kWh/10 min: local 53→59, ordered 119→132, cause `accumulated SOC flooring`
- 1 kWh/5 min: local 73→77, ordered 158→167, cause `accumulated SOC flooring`

## Artifact hashes

- `oracle.json`: `6d212cc9140aff0c16ab05e4fb1a8e2a2ee9e01f4b476a7926af13cda9c59807`
- `transition_candidates.csv`: `a0a2c2b346ab874c80d176a3af3d8be6eeb3eca27fa032634b1e6a19e5e2c705`
- `frontier_states.csv`: `876fb4d17479cc3c3e40b03a0e1d823c8458e1a38c1994e11e129aeb6a310c1c`
- `counterfactuals.csv`: `4ad528d73f3d532934b0990c208514aabc3395ace135a4cb2539f0b5ed0759c7`
