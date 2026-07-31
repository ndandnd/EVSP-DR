#!/usr/bin/env bash
set -euo pipefail

REMOTE=${REMOTE:-nc437@unicorn-login-01.coecis.cornell.edu}
PROJECT_DIR=${PROJECT_DIR:-/Users/nadan/Documents/projects/demandresponse}
LOCAL_ROOT=${LOCAL_ROOT:-/Users/nadan/Downloads/evsp_final_results_stag999997_fullsoc}
POLL_SECONDS=${POLL_SECONDS:-300}
WAIT=${WAIT:-1}
REMOTE_SQUEUE=${REMOTE_SQUEUE:-/usr/local/slurm/current/bin/squeue}

echo "[MAC] Remote      : ${REMOTE}"
echo "[MAC] Project dir : ${PROJECT_DIR}"
echo "[MAC] Local root  : ${LOCAL_ROOT}"

if [ "$WAIT" = "1" ]; then
  echo "[MAC] Waiting for cluster jobs named CG12HFS or MIP40FS to finish..."
  ssh "${REMOTE}" "while true; do remaining=\$(${REMOTE_SQUEUE} -h -u nc437 -n CG12HFS,MIP40FS | wc -l | tr -d ' '); echo \"[CLUSTER] \$(date): remaining jobs = \${remaining}\"; if [ \"\${remaining}\" = \"0\" ]; then break; fi; sleep ${POLL_SECONDS}; done"
fi

mkdir -p "${LOCAL_ROOT}"

echo "[MAC] Pulling only stag999997 full-SOC result folders..."
rsync -av --prune-empty-dirs \
  --include='*/' \
  --include='Inst_10B_RND00[1-4]_*_stag999997_imp-2.0_peak??_fullsoc_g300_*/***' \
  --include='Inst_15B_RND00[1-4]_*_stag999997_imp-2.0_peak??_fullsoc_g300_*/***' \
  --exclude='*' \
  "${REMOTE}:~/demandresponse/src/results/" \
  "${LOCAL_ROOT}/"

echo "[MAC] Rendering Gantt plots..."
cd "${PROJECT_DIR}"
MPLCONFIGDIR=/private/tmp/mplconfig python3 src/plot_charging_gantt_from_solutions.py \
  --results-root "${LOCAL_ROOT}" \
  --output-dir "${LOCAL_ROOT}/charging_gantt_plots" \
  --verbose

echo
echo "[MAC] Done."
echo "[MAC] Plots are in: ${LOCAL_ROOT}/charging_gantt_plots"
