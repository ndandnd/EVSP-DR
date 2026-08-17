# Cross-generation evidence collection and rebuild

The collector is read-only with respect to computational artifacts. It hashes
matched files and writes one new manifest. It does not repair tails, copy
results, invoke solvers, or submit jobs.

## Paste-ready Unicorn Slurm launcher

```bash
cd "$HOME/EVSP-DR-cross-generation-evidence"
REVIEWED_EVIDENCE_COMMIT="${REVIEWED_EVIDENCE_COMMIT:-}"
RAW_K40_CAMPAIGN="${RAW_K40_CAMPAIGN:-}"
RAW_K40_JOB_IDS="${RAW_K40_JOB_IDS:-}"
REVIEWED_MANIFEST="$HOME/evsp-evidence-manifests/cross-generation-reviewed.json"
LOG_DIR="$HOME/evsp-evidence-logs"

if [[ -z "$REVIEWED_EVIDENCE_COMMIT" || -z "$RAW_K40_CAMPAIGN" || -z "$RAW_K40_JOB_IDS" ]]; then
  echo "Set REVIEWED_EVIDENCE_COMMIT, RAW_K40_CAMPAIGN, and space-separated RAW_K40_JOB_IDS." >&2
elif ! git checkout --detach "$REVIEWED_EVIDENCE_COMMIT"; then
  echo "Detached checkout failed; evidence collection not submitted." >&2
elif [[ -n "$(git status --porcelain)" ]]; then
  echo "Checkout is not tracked-clean; evidence collection not submitted." >&2
else
  dependency_args=()
  for job_id in $RAW_K40_JOB_IDS; do
    dependency_args+=(--after-job-id "$job_id")
  done
  python -u src/launch_cross_generation_evidence.py \
    --phase collect \
    --template CROSS_GENERATION_EVIDENCE_INPUT_MANIFEST_20260816.json \
    --root "current_heuristic=$HOME/EVSP-DR-current/src/results" \
    --root "repool_small=$HOME/EVSP-DR-k40mip-f40b120/src/results/repool_small" \
    --root "exact_big=$HOME/EVSP-DR-k40mip-f40b120/src/results/exact_big" \
    --root "k40_factorial=$HOME/EVSP-DR-k40fx-eb85ca0/src/results/k40_factorial" \
    --root "mip_campaign=$HOME/EVSP-DR-k40mip-f40b120/src/results/mip_statistics" \
    --root "releases=$HOME/EVSP-DR-releases" \
    --campaign "raw_k40=$RAW_K40_CAMPAIGN" \
    "${dependency_args[@]}" \
    --manifest "$REVIEWED_MANIFEST" \
    --log-dir "$LOG_DIR" \
    --expected-commit "$REVIEWED_EVIDENCE_COMMIT" \
    --submit
fi
```

This submits only `run_cross_generation_evidence_job.py` to a Scaglione
compute node with `afterany` dependencies. After dependency release the worker
uses a zero-second default wait and fails closed unless the four-cell RAW
campaign, every scheduled checkpoint, source status/journal, and final output
are complete and valid. It never submits CG or MIP solves.

## Rebuild

The build job publishes a create-only evidence directory and a separate
completion-marker archive directory containing `evidence.tar`, its SHA-256
sidecar, `ARCHIVE_MANIFEST.json`, and `completion.json`.

## Inputs still expected from Unicorn/releases

- Current instrumented heuristic-DP pricing CSVs and endpoint/checkpoint JSONs.
- Exact `repool_small`, `exact_big`, and k40 factorial `.iters.csv`, status,
  journal and optional phase telemetry.
- MIP campaign checkpoint/final artifacts with RAW/MATCHING/GIRO metadata.
- Verified single-duty, exact-pair and small-union archive manifests.
- Instance, trip-set and tariff hashes from immutable status/release manifests.
- Git/worktree identity, solver/backend versions, resources and stopping rules.

The tracked historical CSVs are already enumerated in the template manifest.
They provide real trajectories but incomplete provenance; the output must retain
those provenance fields as null.
