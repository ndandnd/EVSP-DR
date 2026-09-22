# Exact native MIP invocation templates (not executed)

Expand the output-directory job/restart template only inside a validated allocation wrapper; create the directory with exist_ok=False. The wrapper must set the environment from salvage_plan.json and run the native replay gate before launching the solve. No initial-partition/witness argument is supplied.

## c1_k15

```sh
/home/nc437/evsp_env/bin/python /home/nc437/ladder-lite/integer_columns_k15_20260921/code/src/run_exact_pool_mip.py --result /home/nc437/ladder-lite/integer_columns_k15_20260921/results/c1_k15/treatment_s20260921/704511_r0/dive/cg.json --data-dir /home/nc437/ladder-lite/integer_columns_k15_20260921/code/data --reference-data-dir /home/nc437/ladder-lite/integer_columns_k15_20260921/code/data --cover --two-stage --timelimit 1852 --stage1-timelimit 926 --threads 8 --seed 20260921 --mipgap 0.0001 --gurobi-log '/home/nc437/ladder-lite/integer_columns_k15_20260921/salvage_20260922/c1_k15/job_${SLURM_JOB_ID}_r${SLURM_RESTART_COUNT:-0}/mip_gurobi.log' --out '/home/nc437/ladder-lite/integer_columns_k15_20260921/salvage_20260922/c1_k15/job_${SLURM_JOB_ID}_r${SLURM_RESTART_COUNT:-0}/result.json'
```

## c5_k15

```sh
/home/nc437/evsp_env/bin/python /home/nc437/ladder-lite/integer_columns_k15_20260921/code/src/run_exact_pool_mip.py --result /home/nc437/ladder-lite/integer_columns_k15_20260921/results/c5_k15/treatment_s20260921/704515_r0/dive/cg.json --data-dir /home/nc437/ladder-lite/integer_columns_k15_20260921/code/data --reference-data-dir /home/nc437/ladder-lite/integer_columns_k15_20260921/code/data --cover --two-stage --timelimit 3266 --stage1-timelimit 1633 --threads 8 --seed 20260921 --mipgap 0.0001 --gurobi-log '/home/nc437/ladder-lite/integer_columns_k15_20260921/salvage_20260922/c5_k15/job_${SLURM_JOB_ID}_r${SLURM_RESTART_COUNT:-0}/mip_gurobi.log' --out '/home/nc437/ladder-lite/integer_columns_k15_20260921/salvage_20260922/c5_k15/job_${SLURM_JOB_ID}_r${SLURM_RESTART_COUNT:-0}/result.json'
```
