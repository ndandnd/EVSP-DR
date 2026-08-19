# Duty 13411 grid-transition oracle

Read-only post-hoc current-code diagnostic. Counterfactuals are not feasibility or pricing certificates and do not change production physics or the running ladder.

No-floor counterfactuals replay the same production prefix timing, stations, actions, and grid charge gains while retaining SOC residuals; they do not substitute the separately optimized continuous witness state.

## Grid outcomes

- 15 kWh/10 min: local 46→53, ordered 106→119, cause `interaction`
- 5 kWh/10 min: local 46→53, ordered 106→119, cause `interaction`
- 2.5 kWh/10 min: local 53→59, ordered 119→132, cause `unresolved`
- 1 kWh/10 min: local 53→59, ordered 119→132, cause `unresolved`
- 1 kWh/5 min: local 73→77, ordered 158→167, cause `unresolved`

## Artifact hashes

- `oracle.json`: `b0d23755a6035082e5d00f488d132a889cf37f66a0a16b15436ed1d408a85dfe`
- `transition_candidates.csv`: `a0a2c2b346ab874c80d176a3af3d8be6eeb3eca27fa032634b1e6a19e5e2c705`
- `frontier_states.csv`: `876fb4d17479cc3c3e40b03a0e1d823c8458e1a38c1994e11e129aeb6a310c1c`
- `counterfactuals.csv`: `d0651693371deb482ff82fcbddea0b620937bd67da524164d2080f13c09ccb7e`
