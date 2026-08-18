# Tariff-response pilot runbook

This package decomposes charging response from route response without treating
either as continuous-price optimality. It is dry-run-only unless the exact
plan SHA is supplied on a second invocation.

The main gate contains:

- one full-GIRO40 Tier-0/Tier-1 job covering all reviewed tariffs;
- 22 k5/k8 tariff-specific fixed-duty seed jobs;
- 44 k5/k8 exact-CG jobs: RAW and GIRO-AUGMENTED;
- 44 corresponding Scaglione finite-pool MIPs.

The separate k40-preparation gate contains 11 fixed-duty seed jobs and 22
RAW/GIRO40-AUGMENTED exact-CG preparations. It contains no k40 MIP and cannot
be selected accidentally by the main submission gate.

## Guarded fetch, dry run, and optional submission

Run from an interactive Unicorn shell. This deliberately does not use
`set -e` and never exits the login shell.

```bash
REVIEWED_COMMIT="${REVIEWED_COMMIT:-}"
CAMPAIGN="${TARIFF_CAMPAIGN:-}"
SUBMIT_SCOPE="${SUBMIT_SCOPE:-none}"
APPROVED_PLAN_SHA256="${APPROVED_PLAN_SHA256:-}"
PYTHON="${EVSP_TARIFF_PYTHON:-$HOME/evsp_env/bin/python3.12}"
RUN_ROOT="$HOME/EVSP-DR-tariff-response"
INPUT_ROOT="$HOME/evsp_tariff_response_inputs"
PLAN_ROOT="$HOME/evsp_tariff_response_plans"
RESERVATIONS="$HOME/evsp_tariff_response_reservations"
K5_PATH="${K5_PATH:-$INPUT_ROOT/k5.csv}"
K8_PATH="${K8_PATH:-$INPUT_ROOT/k8.csv}"
K40_PATH="${K40_PATH:-$INPUT_ROOT/k40.csv}"
K5_SHA256="${K5_SHA256:-}"
K8_SHA256="${K8_SHA256:-}"
K40_SHA256="${K40_SHA256:-}"
K5_URL="${K5_URL:-}"
K8_URL="${K8_URL:-}"
K40_URL="${K40_URL:-}"
PLAN="$PLAN_ROOT/$CAMPAIGN.plan.json"
MATRIX="$PLAN_ROOT/$CAMPAIGN.job-matrix.csv"

if [[ ! "$REVIEWED_COMMIT" =~ ^[0-9a-f]{40}$ || \
      ! "$CAMPAIGN" =~ ^[a-z0-9][a-z0-9._-]{2,79}$ ]]; then
  echo "Set exact REVIEWED_COMMIT and a safe TARIFF_CAMPAIGN." >&2
elif [[ "$SUBMIT_SCOPE" != "none" && "$SUBMIT_SCOPE" != "main" && \
        "$SUBMIT_SCOPE" != "k40-preparation" ]]; then
  echo "SUBMIT_SCOPE must be none, main, or k40-preparation." >&2
elif [[ ! -x "$PYTHON" ]]; then
  echo "Approved Python 3.12 is unavailable." >&2
elif [[ ! -d "$RUN_ROOT/.git" ]] && \
     ! git clone https://github.com/ndandnd/EVSP-DR.git "$RUN_ROOT"; then
  echo "Public clone failed; nothing submitted." >&2
elif ! git -C "$RUN_ROOT" fetch origin "$REVIEWED_COMMIT" || \
     ! git -C "$RUN_ROOT" checkout --detach "$REVIEWED_COMMIT"; then
  echo "Reviewed detached checkout failed; nothing submitted." >&2
elif [[ "$(git -C "$RUN_ROOT" rev-parse HEAD)" != "$REVIEWED_COMMIT" || \
        -n "$(git -C "$RUN_ROOT" status --porcelain)" ]]; then
  echo "Checkout is not the exact clean reviewed commit." >&2
else
  mkdir -p "$INPUT_ROOT" "$PLAN_ROOT" "$RESERVATIONS"
  for SPEC in \
    "k5|$K5_PATH|$K5_URL" \
    "k8|$K8_PATH|$K8_URL" \
    "k40|$K40_PATH|$K40_URL"; do
    IFS='|' read -r LABEL TARGET URL <<<"$SPEC"
    if [[ ! -s "$TARGET" && -n "$URL" ]]; then
      TMP="$TARGET.download.$$"
      if curl -fL --retry 3 --retry-delay 3 "$URL" -o "$TMP"; then
        if ! ln "$TMP" "$TARGET"; then
          echo "$LABEL target appeared concurrently." >&2
        fi
      else
        echo "$LABEL public curl fetch failed." >&2
      fi
      rm -f "$TMP"
    fi
  done

  if ! {
    [[ "$K5_SHA256" =~ ^[0-9a-f]{64}$ ]] &&
    [[ "$K8_SHA256" =~ ^[0-9a-f]{64}$ ]] &&
    [[ "$K40_SHA256" =~ ^[0-9a-f]{64}$ ]] &&
    [[ "$(sha256sum "$K5_PATH" 2>/dev/null | awk '{print $1}')" == "$K5_SHA256" ]] &&
    [[ "$(sha256sum "$K8_PATH" 2>/dev/null | awk '{print $1}')" == "$K8_SHA256" ]] &&
    [[ "$(sha256sum "$K40_PATH" 2>/dev/null | awk '{print $1}')" == "$K40_SHA256" ]]
  }; then
    echo "Explicit k5/k8/k40 paths or hashes are missing/mismatched." >&2
  else
    COMMON=(
      "$PYTHON" -I -B "$RUN_ROOT/src/run_reviewed_python.py"
      "$REVIEWED_COMMIT"
      launch_tariff_response_pilot.py
      --campaign "$CAMPAIGN"
      --python "$PYTHON"
      --reservation-root "$RESERVATIONS"
      --instance "k5=$K5_PATH"
      --instance "k8=$K8_PATH"
      --instance "k40=$K40_PATH"
      --instance-sha256 "k5=$K5_SHA256"
      --instance-sha256 "k8=$K8_SHA256"
      --instance-sha256 "k40=$K40_SHA256"
      --tariff-manifest "$RUN_ROOT/data/tariff_response/tariff_manifest.csv"
    )

    if [[ "$SUBMIT_SCOPE" == "none" && \
          ( -e "$PLAN" || -e "$MATRIX" ) ]]; then
      echo "Dry-run output exists; choose a new campaign." >&2
    elif [[ "$SUBMIT_SCOPE" == "none" ]] && \
         ! "${COMMON[@]}" --plan-out "$PLAN" --matrix-out "$MATRIX"; then
      echo "Dry-run plan generation failed; nothing submitted." >&2
    elif [[ ! -s "$PLAN" || ! -s "$MATRIX" ]]; then
      echo "Reviewed plan/matrix is missing; nothing submitted." >&2
    else
      OBSERVED_PLAN_SHA=$(sha256sum "$PLAN" | awk '{print $1}')
      echo "PLAN: $PLAN"
      echo "MATRIX: $MATRIX"
      echo "PLAN SHA-256: $OBSERVED_PLAN_SHA"
      if [[ "$SUBMIT_SCOPE" != "none" && \
            "$APPROVED_PLAN_SHA256" == "$OBSERVED_PLAN_SHA" ]]; then
        if [[ "$SUBMIT_SCOPE" == "k40-preparation" ]]; then
          "${COMMON[@]}" --approved-plan-sha256 \
            "$APPROVED_PLAN_SHA256" --submit --submit-k40-preparation
        else
          "${COMMON[@]}" --approved-plan-sha256 \
            "$APPROVED_PLAN_SHA256" --submit
        fi
      else
        echo "NOT SUBMITTED. Review the plan/matrix, then provide its exact SHA and one explicit submission scope."
      fi
    fi
  fi
fi
```

## Separate immutable archive/checksum command

Use only after the selected campaign scope is complete:

```bash
ARCHIVE_SCOPE="${ARCHIVE_SCOPE:-main}"
if [[ "$ARCHIVE_SCOPE" == "k40-preparation" ]]; then
  CAMPAIGN_ROOT="$RUN_ROOT/src/results/tariff_response/$CAMPAIGN-k40prep"
else
  CAMPAIGN_ROOT="$RUN_ROOT/src/results/tariff_response/$CAMPAIGN"
fi
ARCHIVE_ROOT="$HOME/evsp_tariff_response_archives"
BUNDLE="$ARCHIVE_ROOT/$CAMPAIGN.bundle"
ARCHIVE_NAME="$CAMPAIGN.tar.gz"

if [[ ! "$REVIEWED_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Set the campaign's exact REVIEWED_COMMIT." >&2
elif [[ -e "$BUNDLE" ]]; then
  echo "Archive bundle exists; refusing overwrite." >&2
elif [[ ! -d "$CAMPAIGN_ROOT" ]]; then
  echo "Campaign root is missing." >&2
elif [[ "$(git -C "$RUN_ROOT" rev-parse HEAD 2>/dev/null)" != \
        "$REVIEWED_COMMIT" || \
        -n "$(git -C "$RUN_ROOT" status --porcelain 2>/dev/null)" ]]; then
  echo "Archive checkout is not the exact clean reviewed commit." >&2
elif ! "$PYTHON" -I -B - "$CAMPAIGN_ROOT" \
       "$REVIEWED_COMMIT" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
expected=sys.argv[2]
plan_raw=(root/"approved-plan.json").read_bytes()
plan=json.loads(plan_raw)
manifest=json.loads((root/"campaign.json").read_text())
if (
    (plan.get("checkout_identity") or {}).get("commit") != expected
    or manifest.get("approval_sha256")
    != hashlib.sha256(plan_raw).hexdigest()
    or not manifest.get("submitted_jobs")
):
    raise SystemExit("campaign approval/commit is incomplete")
jobs={job["job_key"]:job for job in plan["jobs"]}
for submitted in manifest["submitted_jobs"]:
    job=jobs[submitted["job_key"]]
    output=pathlib.Path(job["output"])
    if not output.exists():
        raise SystemExit(f"missing output: {job['job_key']}")
    if job["phase"]=="MIP" and not (
        pathlib.Path(job["progress_dir"])/"final.json"
    ).is_file():
        raise SystemExit(f"missing MIP progress: {job['job_key']}")
PY
then
  echo "Campaign outputs or approval are incomplete." >&2
else
  mkdir -p "$ARCHIVE_ROOT"
  TMP_BUNDLE=$(mktemp -d "$ARCHIVE_ROOT/.bundle.XXXXXXXX")
  if [[ -z "$TMP_BUNDLE" || ! -d "$TMP_BUNDLE" ]] || \
     ! cp -a "$CAMPAIGN_ROOT" "$TMP_BUNDLE/campaign"; then
    echo "Archive staging failed." >&2
  elif ! tar --sort=name --mtime='UTC 1970-01-01' \
       --owner=0 --group=0 --numeric-owner \
       -czf "$TMP_BUNDLE/$ARCHIVE_NAME" \
       -C "$TMP_BUNDLE" campaign; then
    echo "Archive creation failed." >&2
  else
    DIGEST=$(sha256sum "$TMP_BUNDLE/$ARCHIVE_NAME" | awk '{print $1}')
    SIDE_READY=0
    if printf '%s  %s\n' "$DIGEST" "$ARCHIVE_NAME" \
         > "$TMP_BUNDLE/$ARCHIVE_NAME.sha256"; then
      SIDE_READY=1
    fi
    if [[ "$DIGEST" =~ ^[0-9a-f]{64}$ && "$SIDE_READY" == "1" ]] && \
       "$PYTHON" -I -B - "$TMP_BUNDLE" "$BUNDLE" <<'PY'
import ctypes, os, sys
source,target=map(os.fsencode,sys.argv[1:])
libc=ctypes.CDLL(None,use_errno=True)
renameat2=getattr(libc,"renameat2",None)
if renameat2 is None or renameat2(-100,source,-100,target,1) != 0:
    raise SystemExit("atomic no-clobber publication failed")
PY
    then
      TMP_BUNDLE=""
      echo "ARCHIVE: $BUNDLE/$ARCHIVE_NAME"
      echo "CHECKSUM: $BUNDLE/$ARCHIVE_NAME.sha256"
    else
      echo "Archive publication failed." >&2
    fi
  fi
  if [[ -n "$TMP_BUNDLE" && -d "$TMP_BUNDLE" ]]; then
    rm -rf "$TMP_BUNDLE"
  fi
fi
```

Tier-0 scalar costs are expected to remain unavailable for the real source
whenever a recorded window crosses changing tariff hours or lacks tariff
coverage. This is a correctness outcome, not a failed run.
