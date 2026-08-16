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

[[ -n "${APPROVED_PORTABLE_BUNDLE_SHA256:-}" ]] || {
  echo "Set APPROVED_PORTABLE_BUNDLE_SHA256 to the reviewed src/portable_bundle.py hash." >&2
  exit 2
}
[[ "$(sha256sum src/portable_bundle.py | awk '{print $1}')" \
   == "$APPROVED_PORTABLE_BUNDLE_SHA256" ]] || {
  echo "Current portable_bundle.py differs from approved hash." >&2
  exit 2
}
PROBE_REPORT="$REVIEW_ROOT/filesystem-capability.${APPROVED_PORTABLE_BUNDLE_SHA256}.json"
if [[ ! -f "$PROBE_REPORT" ]]; then
  python -u src/probe_portable_publication.py \
    --directory "$PROBE_ROOT" \
    --out "$PROBE_REPORT"
fi
python - "$PROBE_REPORT" "$APPROVED_PORTABLE_BUNDLE_SHA256" "$PROBE_ROOT" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
expected = {
  "portable_protocol": "complete_valid",
  "implementation_sha256": sys.argv[2],
  "parent": __import__("os").path.realpath(sys.argv[3]),
  "hardlink_noreplace": True,
  "flock_exclusive": True,
  "ready_for_recovery_probe_only": True,
}
if any(d.get(key) != value for key, value in expected.items()):
    raise SystemExit("publication probe report does not match approval")
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

INVENTORY_OUT="$REVIEW_ROOT/inventory.${APPROVED_RECOVERY_PLAN_SHA256}.json"
if [[ ! -f "$INVENTORY_OUT" ]]; then
python -u src/launch_mip_statistics_campaign.py \
  --mode inventory \
  --root "repool_small=$SOURCE_ROOT/src/results/repool_small" \
  --root "exact_big=$SOURCE_ROOT/src/results/exact_big" \
  --root "k40_factorial=$SOURCE_ROOT/src/results/k40_factorial" \
  --root "bigtar_snapshots=$SOURCE_ROOT/src/results/bigtar_snapshots" \
  --root "fresh_preparation=$SOURCE_ROOT/src/results/mip_statistics_prep" \
  --data-root "$SOURCE_ROOT/data" \
  --inventory-out "$INVENTORY_OUT"
  (cd "$(dirname "$INVENTORY_OUT")" && \
    sha256sum "$(basename "$INVENTORY_OUT")" > "$(basename "$INVENTORY_OUT").sha256")
fi
(cd "$(dirname "$INVENTORY_OUT")" && \
  test -f "$(basename "$INVENTORY_OUT").sha256" && \
  sha256sum -c "$(basename "$INVENTORY_OUT").sha256")

# Dry run only: one k40 RAW cell and the identical pool plus GIRO columns/start.
# The 1,800-second limit gives checkpoint marks 0, 5, 15, and 30 minutes.
PILOT_OUT="$REVIEW_ROOT/two-cell-pilot-plan.${APPROVED_RECOVERY_PLAN_SHA256}.json"
if [[ ! -f "$PILOT_OUT" ]]; then
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
  --plan-out "$PILOT_OUT"
  (cd "$(dirname "$PILOT_OUT")" && \
    sha256sum "$(basename "$PILOT_OUT")" > "$(basename "$PILOT_OUT").sha256")
fi
(cd "$(dirname "$PILOT_OUT")" && \
  test -f "$(basename "$PILOT_OUT").sha256" && \
  sha256sum -c "$(basename "$PILOT_OUT").sha256")
python - "$PILOT_OUT" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
jobs=d.get("jobs")
if (
    d.get("schema") != "evsp-dr-mip-statistics-approved-plan-v1"
    or d.get("mode") != "two-cell"
    or d.get("blocked") is not False
    or not isinstance(jobs, list)
    or len(jobs) != 2
    or {job.get("arm") for job in jobs} != {"RAW", "GIRO"}
    or any(job.get("time_limit_s") != 1800 for job in jobs)
    or any(job.get("partitioning") != "strict_exact_once" for job in jobs)
):
    raise SystemExit("two-cell pilot plan is blocked or malformed")
PY
```

Afterward, run the read-only strict check
`monitor_k40_factorial_mip_screen.py --campaign-root "$CAMPAIGN"
--source-campaign-sha256 "$APPROVED_SOURCE_CAMPAIGN_SHA256"
--require-complete
--approved-recovery-plan-sha256 "$APPROVED_RECOVERY_PLAN_SHA256"`. It reports
complete valid outputs, recoverable validated
raw outputs, incomplete publication, and missing/invalid results separately.
