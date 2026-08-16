# Unicorn portable publication and k40 recovery

Unicorn rejected `renameat2(RENAME_NOREPLACE)` with `EINVAL`. The replacement
protocol reserves a destination with `mkdir`, writes and fsyncs members through
same-directory temporary files, and publishes `completion.json` last. A bundle
is complete only when that marker and every committed hash validate.

The failed campaign’s raw results and temporary directories are inputs, not
trash. Recovery never deletes, renames, or rewrites them and never invokes
Gurobi.

Do not claim cluster readiness until the compute-node probe below succeeds and
the recovered bundles pass the read-only monitor. The block intentionally
contains no MIP submission:

```bash
set -euo pipefail

# Run only inside a human-created Scaglione compute allocation after reviewing
# this branch. The first action is the tiny publication capability probe.
[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo "Refusing: enter a reviewed compute-node allocation first." >&2
  exit 2
}
[[ "${EVSP_PORTABLE_RECOVERY_APPROVED:-0}" == "1" ]] || {
  echo "Set EVSP_PORTABLE_RECOVERY_APPROVED=1 after code review." >&2
  exit 2
}

RECOVERY_ROOT="$HOME/EVSP-DR-mip-publication-recovery"
SOURCE_ROOT="$HOME/EVSP-DR-k40mip-f40b120"
CAMPAIGN="$SOURCE_ROOT/src/results/k40_factorial_mip/k40fx_mip2h_20260816T035618Z"
PROBE_ROOT="$SOURCE_ROOT/src/results/publication_probe_20260816"
REVIEW_ROOT="$RECOVERY_ROOT/review/k40fx_mip2h_20260816T035618Z"
mkdir -p "$PROBE_ROOT" "$REVIEW_ROOT"

if [[ ! -f "$REVIEW_ROOT/filesystem-capability.json" ]]; then
  python -u src/probe_portable_publication.py \
    --directory "$PROBE_ROOT" \
    --out "$REVIEW_ROOT/filesystem-capability.json"
fi
python - "$REVIEW_ROOT/filesystem-capability.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d["portable_protocol"] == "complete_valid"
PY

[[ -n "${APPROVED_SOURCE_CAMPAIGN_SHA256:-}" ]] || {
  OBSERVED_SOURCE_SHA=$(sha256sum "$CAMPAIGN/campaign.json" | awk '{print $1}')
  echo "Set APPROVED_SOURCE_CAMPAIGN_SHA256=$OBSERVED_SOURCE_SHA after review." >&2
  exit 2
}
[[ "$(sha256sum "$CAMPAIGN/campaign.json" | awk '{print $1}')" \
   == "$APPROVED_SOURCE_CAMPAIGN_SHA256" ]] || {
  echo "campaign.json differs from out-of-band approval." >&2
  exit 2
}

if [[ ! -f "$REVIEW_ROOT/recovery-plan.json" ]]; then
  python -u src/recover_k40_factorial_mip_campaign.py \
    --campaign-root "$CAMPAIGN" \
    --source-campaign-sha256 "$APPROVED_SOURCE_CAMPAIGN_SHA256" \
    --plan-out "$REVIEW_ROOT/recovery-plan.json"
else
  python -u src/recover_k40_factorial_mip_campaign.py \
    --campaign-root "$CAMPAIGN" \
    --source-campaign-sha256 "$APPROVED_SOURCE_CAMPAIGN_SHA256"
fi

OBSERVED_RECOVERY_SHA=$(sha256sum "$REVIEW_ROOT/recovery-plan.json" | awk '{print $1}')
[[ -n "${APPROVED_RECOVERY_PLAN_SHA256:-}" ]] || {
  echo "Set APPROVED_RECOVERY_PLAN_SHA256=$OBSERVED_RECOVERY_SHA after review." >&2
  exit 2
}
[[ "$OBSERVED_RECOVERY_SHA" == "$APPROVED_RECOVERY_PLAN_SHA256" ]] || {
  echo "Recovery plan changed after approval." >&2
  exit 2
}

python -u src/recover_k40_factorial_mip_campaign.py \
  --campaign-root "$CAMPAIGN" \
  --source-campaign-sha256 "$APPROVED_SOURCE_CAMPAIGN_SHA256" \
  --apply \
  --approved-plan-sha256 "$APPROVED_RECOVERY_PLAN_SHA256"

python -u src/launch_mip_statistics_campaign.py \
  --mode inventory \
  --root "repool_small=$SOURCE_ROOT/src/results/repool_small" \
  --root "exact_big=$SOURCE_ROOT/src/results/exact_big" \
  --root "k40_factorial=$SOURCE_ROOT/src/results/k40_factorial" \
  --root "bigtar_snapshots=$SOURCE_ROOT/src/results/bigtar_snapshots" \
  --root "fresh_preparation=$SOURCE_ROOT/src/results/mip_statistics_prep" \
  --data-root "$SOURCE_ROOT/data" \
  --inventory-out "$REVIEW_ROOT/inventory.json"

# Dry run only: one k40 RAW cell and the identical pool plus GIRO columns/start.
# The 1,800-second limit gives checkpoint marks 0, 5, 15, and 30 minutes.
python -u src/launch_mip_statistics_campaign.py \
  --mode two-cell \
  --campaign k40-publication-two-cell-review \
  --root "repool_small=$SOURCE_ROOT/src/results/repool_small" \
  --root "exact_big=$SOURCE_ROOT/src/results/exact_big" \
  --root "k40_factorial=$SOURCE_ROOT/src/results/k40_factorial" \
  --root "bigtar_snapshots=$SOURCE_ROOT/src/results/bigtar_snapshots" \
  --root "fresh_preparation=$SOURCE_ROOT/src/results/mip_statistics_prep" \
  --data-root "$SOURCE_ROOT/data" \
  --giro-start "40=$CAMPAIGN/input/common/validated_start.json" \
  --python "$HOME/evsp_env/bin/python" \
  --plan-out "$REVIEW_ROOT/two-cell-pilot-plan.json"
```

Afterward, run the read-only strict check
`monitor_k40_factorial_mip_screen.py --campaign-root "$CAMPAIGN"
--source-campaign-sha256 "$APPROVED_SOURCE_CAMPAIGN_SHA256"
--require-complete`. It reports complete valid outputs, recoverable validated
raw outputs, incomplete publication, and missing/invalid results separately.
