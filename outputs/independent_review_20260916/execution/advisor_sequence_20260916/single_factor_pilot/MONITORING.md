# Monitor only the six authorized pilots

Read-only local command:

```sh
python3 /Users/nadan/Documents/projects/demandresponse/outputs/independent_review_20260916/execution/advisor_sequence_20260916/single_factor_pilot/monitor_pilot.py
```

The dated `snapshots/<UTC>/status.json` contains scheduler states/MaxRSS, worker attempts, gate evidence and error logs if a worker failed. `latest_snapshot.txt` points to it. Only these registered pilot job IDs are queried. Notify the user promptly if SSH access fails. An hour is sufficient for regular monitoring; a short follow-up after the first job completes can release the five small pilots.

If and only if `conditional_pilots_authorized_to_submit` is true, the user's conditional authorization permits this exact command:

```sh
ssh -S /Users/nadan/.ssh/evsp-unicorn.sock -o BatchMode=yes nc437@unicorn-login-01.coecis.cornell.edu '/home/nc437/evsp_env/bin/python /home/nc437/ladder-lite/advisor_sequence_20260916/single_factor_pilot/submit.py --remaining'
```

The native submitter rechecks the gate and successful control scheduler completion immediately before submission. It is idempotent, records intents, and can submit only the five remaining **20-sequence pilots**. Save its returned jobs ledger locally and collect once again. A false condition is not approval to bypass the gate. On failure, report the reason and preserve artifacts; do not automatically retry, enlarge the test, or start full-pool replay, CG or MIP.

Stay quiet on unchanged healthy/pending status. Notify on a completed or failed pilot and its measured time/memory; distinguish fixed-sequence feasibility from fleet optimality. The 20 sequences never represent full-pool coverage, and other physical arms may legitimately have infeasible or structurally excluded sequences. Any timeout/error remains unknown.
