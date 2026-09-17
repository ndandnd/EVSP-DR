#!/bin/bash
set -euo pipefail
campaign_root=${1:?root}
stage=${2:?stage}
shift 2
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LM_LICENSE_FILE
export PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GRB_LICENSE_FILE=/share/apps/software/gurobi/gurobi.lic
if [[ "$stage" == "prepare" ]]; then
 exec /home/nc437/evsp_env/bin/python "$campaign_root/prepare_shards.py" --root "$campaign_root"
elif [[ "$stage" == "replay" ]]; then
 exec /home/nc437/evsp_env/bin/python "$campaign_root/replay_worker.py" "$campaign_root"
else
 exec /home/nc437/evsp_env/bin/python "$campaign_root/stage_worker.py" --root "$campaign_root" --stage "$stage" "$@"
fi
