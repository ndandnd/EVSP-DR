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

Primary response/elasticity uses alpha `0, 0.25, 0.5, 1.0`. Alpha `2.0`
remains scheduled only as a `negative_price_stress` cell; it is excluded from
primary savings and elasticity outputs and reported in a separate terminal-
surplus stress table/figure.

The concrete r2 k5/k8/k40 inputs are tracked in
`data/tariff_response/frozen_instances/frozen_input_manifest.csv`, SHA-256
`5473e8d83c8e7e1f0b6e872125419466bb5044bbbb014df3184254f6a2b601c6`.

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
PLAN_ROOT="$HOME/evsp_tariff_response_plans"
RESERVATIONS="$HOME/evsp_tariff_response_reservations"
FROZEN_ROOT="$RUN_ROOT/data/tariff_response/frozen_instances"
FROZEN_MANIFEST="$FROZEN_ROOT/frozen_input_manifest.csv"
FROZEN_MANIFEST_SHA256="5473e8d83c8e7e1f0b6e872125419466bb5044bbbb014df3184254f6a2b601c6"
K5_PATH="$FROZEN_ROOT/Practice_Custom_DutyUnion_k05_r2.csv"
K8_PATH="$FROZEN_ROOT/Practice_Custom_DutyUnion_k08_r2.csv"
K40_PATH="$FROZEN_ROOT/Practice_Custom_DutyUnion_k40_r2.csv"
K5_SHA256="6ffea0b8cd3a9d15846946f6828705dd3431b7bafc69bd572ca30ed4530d5cb8"
K8_SHA256="0d368920af0c5b14e0907b85977a9f72163a0cea6431c206f992e89aa31eb27f"
K40_SHA256="3508a11f73d1186ae87588656d65ea62812c6e222623ae85488eff26cafb35fd"
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
  mkdir -p "$PLAN_ROOT" "$RESERVATIONS"
  if ! {
    [[ "$(sha256sum "$FROZEN_MANIFEST" 2>/dev/null | awk '{print $1}')" == \
       "$FROZEN_MANIFEST_SHA256" ]] &&
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
      env -u PYTHONPATH PYTHONNOUSERSITE=1
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
BUNDLE="$ARCHIVE_ROOT/$CAMPAIGN-$ARCHIVE_SCOPE.bundle"
ARCHIVE_NAME="$CAMPAIGN-$ARCHIVE_SCOPE.tar.gz"

if [[ "$ARCHIVE_SCOPE" != "main" && \
      "$ARCHIVE_SCOPE" != "k40-preparation" ]]; then
  echo "ARCHIVE_SCOPE must be main or k40-preparation." >&2
elif [[ ! "$REVIEWED_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Set the campaign's exact REVIEWED_COMMIT." >&2
elif [[ -e "$BUNDLE" ]]; then
  echo "Archive bundle exists; refusing overwrite." >&2
elif [[ ! -d "$CAMPAIGN_ROOT" ]]; then
  echo "Campaign root is missing." >&2
elif [[ "$(git -C "$RUN_ROOT" rev-parse HEAD 2>/dev/null)" != \
        "$REVIEWED_COMMIT" || \
        -n "$(git -C "$RUN_ROOT" status --porcelain 2>/dev/null)" ]]; then
  echo "Archive checkout is not the exact clean reviewed commit." >&2
else
  mkdir -p "$ARCHIVE_ROOT"
  TMP_BUNDLE=$(mktemp -d "$ARCHIVE_ROOT/.bundle.XXXXXXXX")
  if [[ -z "$TMP_BUNDLE" || ! -d "$TMP_BUNDLE" ]] || \
     ! cp -a "$CAMPAIGN_ROOT" "$TMP_BUNDLE/campaign"; then
    echo "Archive staging failed." >&2
  elif ! env -u PYTHONPATH PYTHONNOUSERSITE=1 \
       "$PYTHON" -I -B "$RUN_ROOT/src/run_reviewed_python.py" \
       "$REVIEWED_COMMIT" validate_tariff_response_archive.py \
       --campaign-root "$TMP_BUNDLE/campaign" \
       --expected-commit "$REVIEWED_COMMIT" \
       --scope "$ARCHIVE_SCOPE"; then
    echo "Staged campaign archive validation failed." >&2
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
source_path=os.fsdecode(source)
directories=[]
for current,dirs,files in os.walk(source_path):
    directories.append(current)
    for child in files:
        descriptor=os.open(os.path.join(current,child),os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
for current in reversed(directories):
    descriptor=os.open(current,os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
libc=ctypes.CDLL(None,use_errno=True)
renameat2=getattr(libc,"renameat2",None)
if renameat2 is None or renameat2(-100,source,-100,target,1) != 0:
    raise SystemExit("atomic no-clobber publication failed")
parent=os.open(os.path.dirname(os.fsdecode(target)),os.O_RDONLY)
try:
    os.fsync(parent)
finally:
    os.close(parent)
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

If submission stops after creating a campaign root, do not retry under a new
name. The launcher persists an immutable plan-derived gate comment and exact
user/name/partition/role before mutation. Run the reconciler against the same
campaign. It re-observes cached states, boundedly discovers an
accepted-before-record gate by exact comment, and never treats a command return
code as a state transition. States `ambiguous_held_gate`,
`ambiguous_gate_receipt`, `held_release_failed`, and
`reconciliation_unverified` retain reservations and forbid replacement.
Historical manifests without the exact gate specification are preserved as
`legacy_unverified`; they cannot be silently upgraded.

Each scientific child also has a durable pre-`sbatch` intent and an exact
approved-user/ID/name/partition/comment/dependency receipt while the gate is
held. Accepted-before-record children are recovered by that identity; an
unresolved intent prevents gate release. Worker completion schema v2 binds the
job key, execution digest, treatment, role, scale, tariff, inputs, Slurm ID,
and complete artifact hash set. Assembly and archive validation share that
same identity validator.

Reconcile once after release and again after the gate becomes terminal.
Scientific assembly/archive requires both persisted release evidence and exact
terminal `COMPLETED/0:0` evidence. A terminal non-success is persisted as
`terminal_failed` with `submitted=false`, scheduler source, state, and exit
code before the reconciler raises:

```bash
RECONCILE_SCOPE="${RECONCILE_SCOPE:-main}"
if [[ "$RECONCILE_SCOPE" == "k40-preparation" ]]; then
  RECONCILE_ROOT="$RUN_ROOT/src/results/tariff_response/$CAMPAIGN-k40prep"
else
  RECONCILE_ROOT="$RUN_ROOT/src/results/tariff_response/$CAMPAIGN"
fi
if [[ "$RECONCILE_SCOPE" != "main" && \
      "$RECONCILE_SCOPE" != "k40-preparation" ]]; then
  echo "RECONCILE_SCOPE must be main or k40-preparation." >&2
else
  env -u PYTHONPATH PYTHONNOUSERSITE=1 \
    "$PYTHON" -I -B "$RUN_ROOT/src/run_reviewed_python.py" \
    "$REVIEWED_COMMIT" reconcile_tariff_response_gate.py \
    --campaign-root "$RECONCILE_ROOT" \
    --approved-plan-sha256 "$APPROVED_PLAN_SHA256"
fi
```

## Build normalized evidence after the main scope completes

This command fails closed if any of the 111 main outputs is missing or if a
schedule differs from its hash-bound Tier-1 seed/MIP result:

```bash
CAMPAIGN_ROOT="$RUN_ROOT/src/results/tariff_response/$CAMPAIGN"
EVIDENCE_MANIFEST="$CAMPAIGN_ROOT/evidence/experiment-manifest.json"
EVIDENCE_OUTPUT="$CAMPAIGN_ROOT/evidence/normalized"

if [[ ! "$REVIEWED_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Set the exact REVIEWED_COMMIT." >&2
elif [[ -e "$EVIDENCE_OUTPUT" ]]; then
  echo "Evidence output exists; refusing overwrite." >&2
elif ! env -u PYTHONPATH PYTHONNOUSERSITE=1 \
     "$PYTHON" -I -B "$RUN_ROOT/src/run_reviewed_python.py" \
       "$REVIEWED_COMMIT" assemble_tariff_response_campaign.py \
       --campaign-root "$CAMPAIGN_ROOT" \
       --manifest-out "$EVIDENCE_MANIFEST" \
       --evidence-out "$EVIDENCE_OUTPUT"; then
  echo "Normalized evidence build failed." >&2
else
  echo "NORMALIZED EVIDENCE: $EVIDENCE_OUTPUT"
fi
```
