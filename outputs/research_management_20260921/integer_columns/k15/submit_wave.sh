#!/bin/bash
source /etc/profile >/dev/null 2>&1
set -euo pipefail
WORK=/home/nc437/ladder-lite/integer_columns_k15_20260921
exec 9>"$WORK/submission.lock"
flock -n 9 || { echo 'Another manager owns the submission lock'; exit 0; }
/home/nc437/evsp_env/bin/python "$WORK/check_native_gate.py" > "$WORK/native_gate_receipt.json"
mkdir -p "$WORK/slurm"
if [ ! -e "$WORK/jobs.tsv" ]; then printf 'case\tarm\tseed\tjob_id\n' > "$WORK/jobs.tsv"; fi
for CASE in c1_k15 c3_k15 c5_k15; do
 for SEED in 20260921; do
  for ARM in control treatment; do
   EXISTING=$(awk -F '\t' -v c="$CASE" -v a="$ARM" -v s="$SEED" 'NR>1 && $1==c && $2==a && $3==s {print $4}' "$WORK/jobs.tsv")
   if [ -n "$EXISTING" ]; then continue; fi
   JOB=$(sbatch --parsable --partition=default_partition --requeue --exclude=scaglione-compute-01 --cpus-per-task=8 --mem=32G --time=03:00:00 --job-name="intcol_${CASE}_${ARM}_${SEED}" --output="$WORK/slurm/%j.out" --error="$WORK/slurm/%j.err" "$WORK/run.sub" "$CASE" "$ARM" "$SEED")
   printf '%s\t%s\t%s\t%s\n' "$CASE" "$ARM" "$SEED" "$JOB" >> "$WORK/jobs.tsv"
   scontrol show job "$JOB" > "$WORK/slurm/${JOB}_submission.txt"
   grep -q 'ExcNodeList=scaglione-compute-01' "$WORK/slurm/${JOB}_submission.txt"
  done
 done
done
cat "$WORK/jobs.tsv"
