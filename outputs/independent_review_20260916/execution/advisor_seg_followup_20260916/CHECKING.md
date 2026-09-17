# Compare the reviewer predictions without changing the experiments

From the project root:

```sh
python3 outputs/independent_review_20260916/execution/advisor_seg_followup_20260916/check_predictions.py --refresh
```

The command takes a read-only cluster snapshot, then saves a dated JSON report and short table in `comparisons/`. To use an existing snapshot, omit `--refresh`, or provide `--snapshot /absolute/path/status.json`. It never submits, requeues, changes concurrency, or calls a solver.

`predictions.json` and `reviewer_source/README.md` remain unchanged. Their hashes are included in each comparison. The original source's broader causal claims are not adopted by the checker.

A certified numeric LP mismatch is marked refuted; an uncertified endpoint is unresolved, even if its number agrees. The segregation prediction requires both groups to have consistent pricing certificates and weights 12 and 19 separately; a sum of 31 alone is insufficient. The objective minimized includes charging costs, so fractional route weight is not automatically a fleet-only lower bound, and a weighted optimum can have multiple optimal solutions.

MIP incumbents, finite-pool proofs, trip coverage and capacity checks are reported separately. A timed incumbent is not a proof. A finite-pool proof supports only a prediction about that saved pool; it does not prove full-model fleet optimality. The production driver checks coverage and capacity where imposed and relies on individual event-route construction. This is explicitly distinguished from an independent complete physical replay or duplicate-coverage cleanup.

Missing endpoints are pending. Missing provenance or mismatched settings produce unresolved results. Replay statistics cover completed shards only, and therefore undercount partial work. Any nonfeasible control sequence is an anomaly to investigate.

Nine tests cover certified contradictions, uncertified mismatches, inconsistent certificate flags, missing groups, required group splits, MIP proof scope, incumbent-only outcomes, and invalid coverage. See `checker_tests.json`.
