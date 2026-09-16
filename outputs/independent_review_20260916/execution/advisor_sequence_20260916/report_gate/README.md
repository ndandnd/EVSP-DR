# Four-result reporting gate

This is a read-only report generator. It does not submit, retry, requeue, cancel, or move jobs. A ready report does not authorize more runs.

Wait for all **37 solver cells**: 18 fresh k15 pool-MIP trials, one C5 k31 long MIP, 12 k32 seed trials (including four reused seed-zero trials), and six constrained k5 F6 solver arms. The three original GIRO invoice rows are required too. Until every cell has an endpoint or an explicitly terminal scheduler failure, the output contains readiness only: `items: null`. A terminal failure is censored, never a target miss; an active requeue remains pending.

When ready, exactly four items are emitted:

1. Fresh k15 target hits, all 18 seed outcomes, and any censored trials. These reuse six saved CG pools; the MIP fleet allowance is three hours.
2. C5 k31 outcome after a 12-hour fleet allowance and 30-minute charging allowance.
3. Original invoice interval, fixed-duty cost, and fresh-CG cost for each synthetic peak. Cost differences require two physically validated five-bus optimized outcomes. A common minimum ending energy is distinguished from equal achieved ending energy.
4. All three k32 seeds for each chain1/3/4/5. Report descriptive population variance (divisor3), sample variance (divisor2), and range. Variance is suppressed for censored seeds or unmatched ordered pools. The observed three-hour fleet and 30-minute charging allowances, pool, start, physics, seed, and code controls are audited before reporting.

Route physical replay, duplicate cleanup, finite-pool fleet proof, and a full-model certificate are separate. A missed target without a excluding bound leaves target feasibility unresolved. Seed trials over one pool are not independent instance replications.

After the hourly focused collector:

```sh
python3 outputs/independent_review_20260916/execution/advisor_sequence_20260916/report_gate/four_numbers.py --snapshot outputs/independent_review_20260916/execution/monitor/<UTC stamp>/snapshot.json --out outputs/independent_review_20260916/execution/advisor_sequence_20260916/report_gate/snapshots/<UTC stamp>
```

The snapshot path comes from `execution/monitor/latest_path.txt`. `readiness.json` hashes the snapshot, both frozen manifests, and report-generator source. The focused collector now includes the existing P1 endpoint audit and registers all four reused seed-zero jobs even before their results exist. Real SE3 outcomes are still excluded.

Validation: nine unit tests cover completion, missing cells, scheduler terminal failures, live requeues, wrong budgets, wrong manifests, censored variance, unmatched F6 fleets, and invalid physical audits. First native-backed gate at2026-09-16T23:34:36Z: nine endpoints,28pending,zero terminal failures,zero integrity errors. Four results deliberately withheld.
