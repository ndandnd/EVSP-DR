# Read-only hourly monitoring

Run from the local project root:

```sh
python3 outputs/independent_review_20260916/execution/action3_full_20260916/collect.py
```

This reads only this campaign's registered jobs and output directories. Slurm accounting is filtered to user `nc437` and dates from 2026-09-16 to avoid recycled historical job IDs. It saves a dated `snapshots/<UTC>/status.json`; it does not submit, requeue or alter cluster files. No other project or SE3 data is collected. If SSH fails, notify the user promptly.

Interpret the fields carefully:

- `replay`: completed shards and their explicit outcome counts, out of 125 shards / 254,068 sequences per arm. Partial shards are not counted as completed. Baseline nonfeasible outcomes are flagged separately as unexpected.
- `stages`: existing canonical graph, assembly, CG and MIP endpoint scalar fields, plus assembly coverage. Missing endpoints mean pending or interrupted; consult scheduler and attempts.
- `attempts`: every recorded replay/stage attempt, including interrupted ones. Use scheduler elapsed times for killed attempts lacking an end timestamp. Sum attempts before making time-matched claims.
- `squeue`, `sacct`: scheduler state and registered-job accounting, separate from scientific correctness. `errors` contains explicit read failures and nonempty stderr; inspect warnings before classifying them as fatal.

Monitor the seven CG components: control 342668, PARX60 342670, reserve15 342672, battery236.44 342674, battery239.01 342676, segregation18E1 342678 and segregation18E2 342680. Their MIPs are the next integer job IDs. Graph jobs 342655–342661 can be long; assembly 342662–342667 waits only for the relevant 125 replay elements of 342540, not the entire array.

For new results, verify canonical output hashes against native receipts before adding numerical research claims to the register/Doc. Collect physical checks, CG stop reason/certificate, finite-pool MIP bound/gap, integer fleet and target attainment separately. The compact collector does not rehash every large pool. Do not call an RMP value a full-model lower bound without certified pricing.

Submission is complete. Do not rerun submitters or launch replacements merely because jobs are pending. Interrupted replay tasks can requeue with their journals; CG resumes atomic saved pools. Any new repair/submission should be justified by an actual failure, preserve all attempts and stay inside user-authorized scope.
