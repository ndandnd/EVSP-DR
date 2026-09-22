# Compute-worker launch preparation; NOT SUBMITTED

Stage run_salvage.py and salvage_plan.json under the root below without changing the pinned code checkout. Create its logs/ directory. Recheck live case jobs and the submission ledger under a separate submission lock before sbatch; worker lock only prevents simultaneous execution. Both cells are independent, so no failed-job afterok dependency.

Suggested allocation: default_partition,8CPU,32G,01:30:00,exclude scaglione-compute-01,--no-requeue. Scientific limits remain1852/3266solver seconds; allocation headroom covers physical preparation and I/O. No concurrency throttle needed for two cases.

## c1_k15

Worker command after native environment setup:
```sh
/home/nc437/evsp_env/bin/python /home/nc437/ladder-lite/integer_columns_k15_20260921/salvage_20260922/run_salvage.py --case c1_k15 --plan /home/nc437/ladder-lite/integer_columns_k15_20260921/salvage_20260922/salvage_plan.json --out-root /home/nc437/ladder-lite/integer_columns_k15_20260921/salvage_20260922/results
```

## c5_k15

Worker command after native environment setup:
```sh
/home/nc437/evsp_env/bin/python /home/nc437/ladder-lite/integer_columns_k15_20260921/salvage_20260922/run_salvage.py --case c5_k15 --plan /home/nc437/ladder-lite/integer_columns_k15_20260921/salvage_20260922/salvage_plan.json --out-root /home/nc437/ladder-lite/integer_columns_k15_20260921/salvage_20260922/results
```
