# Ladder-lite commands

Run the smoke in a disposable campaign:

1. `LL_ROOT="$HOME/ladder-lite-smoke" LL_CAMPAIGN=ll_smoke LL_PYTHON="/home/nc437/evsp_env/bin/python3.12" bash scripts/ladder_lite/plan.sh`
2. `SLURM_ARRAY_TASK_ID=0 LL_ROOT="$HOME/ladder-lite-smoke" LL_PYTHON="/home/nc437/evsp_env/bin/python3.12" bash scripts/ladder_lite/run_cell.sh "$HOME/ladder-lite-smoke/campaign/approved-plan.json" PREFLIGHT`
3. `SLURM_ARRAY_TASK_ID=0 LL_ROOT="$HOME/ladder-lite-smoke" LL_PYTHON="/home/nc437/evsp_env/bin/python3.12" LL_BUDGET_OVERRIDE_S=180 bash scripts/ladder_lite/run_cell.sh "$HOME/ladder-lite-smoke/campaign/approved-plan.json" CG`

Then start the production campaign:

4. `export LL_ROOT="$HOME/ladder-lite" LL_PYTHON="/home/nc437/evsp_env/bin/python3.12"`
5. `bash scripts/ladder_lite/plan.sh`
6. `bash scripts/ladder_lite/submit.sh PREFLIGHT`
7. `bash scripts/ladder_lite/status.sh PREFLIGHT`
8. After a PREFLIGHT output appears: `bash scripts/ladder_lite/submit.sh SEED`
9. `bash scripts/ladder_lite/submit.sh CG --scales 2,3,5 --dry-run`
10. `bash scripts/ladder_lite/submit.sh CG --scales 2,3,5`
11. `bash scripts/ladder_lite/submit.sh CG_SENSITIVITY --scales 2,3,5`
12. `bash scripts/ladder_lite/submit.sh CG --scales 8,13,20,30,40`
13. After dependencies finish: `bash scripts/ladder_lite/submit.sh MIP_RAW`
14. `bash scripts/ladder_lite/submit.sh MIP_KNOWN`
15. `bash scripts/ladder_lite/status.sh`
16. `bash scripts/ladder_lite/normalize.sh`
17. `RUN_ID=ll_$(date -u +%Y%m%d) bash scripts/ladder_lite/record_results.sh "$RUN_ID"`

For a confirmed exit 137 or `Exceeded job memory limit`, resubmit that group
with `--mem 64G`. Do not automate memory escalation.
