#!/bin/bash
source /etc/profile >/dev/null 2>&1
set -euo pipefail
WORK=/home/nc437/ladder-lite/integer_columns_20260921
if [ -e "$WORK/jobs.tsv" ]; then echo 'Refusing duplicate submission: ledger exists' >&2; exit 1; fi
mkdir -p "$WORK/slurm"
printf 'case\tarm\tseed\tjob_id\n' > "$WORK/jobs.tsv"
for CASE in c1_k08 c3_k08 c4_k08 c5_k08; do
 for SEED in 20260921 20260922; do
  for ARM in control treatment; do
   JOB=$(sbatch --parsable --partition=default_partition --requeue --exclude=scaglione-compute-01 --cpus-per-task=8 --mem=16G --time=02:00:00 --job-name="intcol_${CASE}_${ARM}_${SEED}" --output="$WORK/slurm/%j.out" --error="$WORK/slurm/%j.err" "$WORK/run.sub" "$CASE" "$ARM" "$SEED")
   printf '%s\t%s\t%s\t%s\n' "$CASE" "$ARM" "$SEED" "$JOB" >> "$WORK/jobs.tsv"
   scontrol show job "$JOB" > "$WORK/slurm/${JOB}_submission.txt"
   grep -q 'ExcNodeList=scaglione-compute-01' "$WORK/slurm/${JOB}_submission.txt"
  done
 done
done
cat "$WORK/jobs.tsv"
