#!/bin/bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LM_LICENSE_FILE
export GRB_LICENSE_FILE=/share/apps/software/gurobi/gurobi.lic PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export EVSP_EXPECTED_COMMIT=a3392e2c8f6722da9538d88a34bab7ef85f7cc72 EVSP_REQUIRE_DETACHED=1
export EVSP_MIP_EXPECTED_RESULT_SHA256=8343ffa5b0d585e39034f68396f14cd857626e97e29188c0ba525659699673d8 EVSP_MIP_EXPECTED_JOURNAL_SHA256=f4ce8dc44cc62a733449fa34cd74be93512edb4ecdb1edbcb9dd776a39dbc254 EVSP_MIP_EXPECTED_INITIAL_PARTITION_SHA256=c7179fc7034786e8e918d59e0923a1fbfe85dbcea705e067c89f12c75473eb3c
OUT=/home/nc437/ladder-lite/research_execution_20260921/c1_followup/${SLURM_JOB_ID}_r${SLURM_RESTART_COUNT:-0}
mkdir -p "$OUT"
cd /home/nc437/ladder-lite/diving_pricing_20260919/code
exec /home/nc437/evsp_env/bin/python src/run_exact_pool_mip.py --result /home/nc437/ladder-lite/diving_pricing_20260919/results/c1_k08/treatment/586633_r0/dive/cg.json --data-dir /home/nc437/ladder-lite/diving_pricing_20260919/code/data --reference-data-dir /home/nc437/ladder-lite/diving_pricing_20260919/code/data --initial-partition-routes /home/nc437/ladder-lite/research_execution_20260921/c1_dive_start.json --verified-expanded-initial-partition --cover --two-stage --timelimit 3600 --stage1-timelimit 1800 --threads 8 --mipgap 0.0001 --seed 20260919 --gurobi-log "$OUT/gurobi.log" --out "$OUT/result.json"
