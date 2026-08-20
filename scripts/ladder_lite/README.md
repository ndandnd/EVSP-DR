# Ladder-lite commands

Set `LL_ROOT` and `LL_PYTHON` once in the shell:

`export LL_ROOT="$HOME/ladder-lite" LL_PYTHON="/home/nc437/evsp_env/bin/python3.12"`

1. `bash scripts/ladder_lite/plan.sh`
2. `SLURM_ARRAY_TASK_ID=0 bash scripts/ladder_lite/run_cell.sh "$LL_ROOT/campaign/approved-plan.json" PREFLIGHT`
3. `SLURM_ARRAY_TASK_ID=0 LL_BUDGET_OVERRIDE_S=180 bash scripts/ladder_lite/run_cell.sh "$LL_ROOT/campaign/approved-plan.json" CG`
4. `bash scripts/ladder_lite/submit.sh PREFLIGHT`
5. `bash scripts/ladder_lite/status.sh PREFLIGHT`
6. After a PREFLIGHT output appears: `bash scripts/ladder_lite/submit.sh SEED`
7. `bash scripts/ladder_lite/submit.sh CG --scales 2,3,5`
8. `bash scripts/ladder_lite/submit.sh CG_SENSITIVITY --scales 2,3,5`
9. `bash scripts/ladder_lite/submit.sh CG --scales 8,13,20,30,40`
10. Resubmit blocked diagnostics after dependencies finish: `bash scripts/ladder_lite/submit.sh MIP_RAW`
11. `bash scripts/ladder_lite/submit.sh MIP_KNOWN`
12. `bash scripts/ladder_lite/status.sh`
13. `bash scripts/ladder_lite/normalize.sh`
14. `bash scripts/ladder_lite/record_results.sh <RUN_ID>`

Inspect every dry run before submission:

`bash scripts/ladder_lite/submit.sh CG --scales 2,3,5 --dry-run`

For a confirmed exit 137 or `Exceeded job memory limit`, resubmit that group
with `--mem 64G`. Do not automate memory escalation.
