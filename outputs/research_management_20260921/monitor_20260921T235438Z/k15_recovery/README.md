# Recovery of the approved k15 continuation

Gate668797 exited before scientific validation: its Slurm accounting query returned exit1 on the compute node. The exception and empty original receipt were preserved remotely; no k15 job ledger existed. The exact underlying accounting-service error was not logged by the old script.

At23:57UTC the unchanged idempotent submitter was rerun from the login node, after all sixteen k8 jobs completed. The native gate passed seven independent handoff receipts (seven repeated runs on four cases). Six preapproved paired k15 jobs were submitted:704510–704515, C1/C3/C5 × control/treatment, seed20260921. All six were subsequently verified RUNNING. No failed validation was bypassed and no scientific setting changed.

Resources:8CPU,32G,3h,default_partition,requeue,scaglione-compute-01 excluded; all six independent allocations submitted without a smaller throttle. Scientific budget remains7200s shared dive-wall plus MIP-solver allowance, max5400s dive. True source prerequisites were already complete. Original held jobs and other projects were untouched.

The remote resource policy was read before submission. jobs.tsv, source manifest, native gate receipt, original failure log, immutable source scripts, submission records and live scheduler checks are retained here and hashed in the parent k15_recovery.json. The source paths remain /home/nc437/ladder-lite/integer_columns_k15_20260921. This is a scheduler recovery, not a k15 result.
