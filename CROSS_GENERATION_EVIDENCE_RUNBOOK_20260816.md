# Cross-generation evidence collection and rebuild

The collector is read-only with respect to computational artifacts. It hashes
matched files and writes one new manifest. It does not repair tails, copy
results, invoke solvers, or submit jobs.

## Paste-ready Unicorn Slurm launcher

```bash
cd "$HOME/EVSP-DR-cross-generation-evidence"
CURRENT_MIP_CAMPAIGN="${CURRENT_MIP_CAMPAIGN:-}"
RAW_K40_CAMPAIGN="${RAW_K40_CAMPAIGN:-}"
REVIEWED_MANIFEST="$HOME/evsp-evidence-manifests/cross-generation-reviewed.json"
APPROVED_MANIFEST_SHA256="${APPROVED_MANIFEST_SHA256:-}"
BUILD_OUT="$HOME/evsp-evidence-builds/cross-generation-final"
ARCHIVE_OUT="$HOME/evsp-evidence-archives/cross-generation-final"
LOG_DIR="$HOME/evsp-evidence-logs"
EVIDENCE_PHASE="${EVIDENCE_PHASE:-collect}"

if [[ -z "$CURRENT_MIP_CAMPAIGN" || -z "$RAW_K40_CAMPAIGN" ]]; then
  echo "Set CURRENT_MIP_CAMPAIGN and RAW_K40_CAMPAIGN to explicit campaign directories." >&2
elif [[ "$EVIDENCE_PHASE" == "collect" ]]; then
  python -u src/launch_cross_generation_evidence.py \
    --phase collect \
    --template CROSS_GENERATION_EVIDENCE_INPUT_MANIFEST_20260816.json \
    --root "current_heuristic=$HOME/EVSP-DR-current/src/results" \
    --root "repool_small=$HOME/EVSP-DR-k40mip-f40b120/src/results/repool_small" \
    --root "exact_big=$HOME/EVSP-DR-k40mip-f40b120/src/results/exact_big" \
    --root "k40_factorial=$HOME/EVSP-DR-k40fx-eb85ca0/src/results/k40_factorial" \
    --root "mip_campaign=$HOME/EVSP-DR-k40mip-f40b120/src/results/mip_statistics" \
    --root "releases=$HOME/EVSP-DR-releases" \
    --current-mip-campaign-root "$CURRENT_MIP_CAMPAIGN" \
    --current-mip-mode "${CURRENT_MIP_MODE:-pilot}" \
    --raw-k40-campaign-root "$RAW_K40_CAMPAIGN" \
    --manifest "$REVIEWED_MANIFEST" \
    --log-dir "$LOG_DIR" \
    --expected-commit "$(git rev-parse HEAD)" \
    --submit
elif [[ "$EVIDENCE_PHASE" == "build" && -n "$APPROVED_MANIFEST_SHA256" && -f "$REVIEWED_MANIFEST" ]]; then
  python -u src/launch_cross_generation_evidence.py \
    --phase build \
    --template CROSS_GENERATION_EVIDENCE_INPUT_MANIFEST_20260816.json \
    --root "current_heuristic=$HOME/EVSP-DR-current/src/results" \
    --root "repool_small=$HOME/EVSP-DR-k40mip-f40b120/src/results/repool_small" \
    --root "exact_big=$HOME/EVSP-DR-k40mip-f40b120/src/results/exact_big" \
    --root "k40_factorial=$HOME/EVSP-DR-k40fx-eb85ca0/src/results/k40_factorial" \
    --root "mip_campaign=$HOME/EVSP-DR-k40mip-f40b120/src/results/mip_statistics" \
    --root "releases=$HOME/EVSP-DR-releases" \
    --current-mip-campaign-root "$CURRENT_MIP_CAMPAIGN" \
    --current-mip-mode "${CURRENT_MIP_MODE:-pilot}" \
    --raw-k40-campaign-root "$RAW_K40_CAMPAIGN" \
    --manifest "$REVIEWED_MANIFEST" \
    --approved-manifest-sha256 "$APPROVED_MANIFEST_SHA256" \
    --build-out "$BUILD_OUT" \
    --archive-out "$ARCHIVE_OUT" \
    --log-dir "$LOG_DIR" \
    --expected-commit "$(git rev-parse HEAD)" \
    --submit
else
  echo "Nothing submitted. Use EVIDENCE_PHASE=collect, or review the manifest and use EVIDENCE_PHASE=build with APPROVED_MANIFEST_SHA256."
fi
```

Both invocations submit only `run_cross_generation_evidence_job.py` to a
Scaglione compute node. The worker waits for both named MIP campaigns to have
validated outputs and `progress/final.json` for every cell; timeout, missing
cells, or provenance errors fail closed. It never submits CG or MIP solves.

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
