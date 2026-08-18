# Exact-CG scale ladder: guarded Unicorn launch

Primary cells use only the historical flat tariff
`data/hourly_prices_flat.csv`, SHA-256
`1f51f2e1f6ca303838ebaaf6272a28ff2d6bbee97146cb04d330e10f191f8200`.
The tracked input manifest explicitly proves equivalence to `flat_h26.csv` but
the ladder commands continue to use the historical bytes.

The dry-run plan contains exactly:

- 21 known-partition preparation tasks;
- 23 exact-CG tasks;
- 21 RAW MIPs and 21 KNOWN-PARTITION diagnostic MIPs;
- 86 experimental tasks total;
- zero k40 MIP submissions (four reuse-only result slots).

## 1. Dry run

This block does not use `set -e` and cannot exit the login shell.

```bash
REVIEWED_COMMIT="${REVIEWED_COMMIT:-}"
CAMPAIGN="${LADDER_CAMPAIGN:-}"
PYTHON_CANDIDATE="${EVSP_LADDER_PYTHON:-$HOME/evsp_env/bin/python3.12}"
PYTHON=$(readlink -f "$PYTHON_CANDIDATE" 2>/dev/null)
RUN_ROOT="$HOME/EVSP-DR-scale-ladder"
PLAN_ROOT="$HOME/evsp_scale_ladder_plans"
RESERVATIONS="$HOME/evsp_scale_ladder_reservations"
PLAN="$PLAN_ROOT/$CAMPAIGN.plan.json"
MATRIX="$PLAN_ROOT/$CAMPAIGN.tasks.csv"

if [[ ! "$REVIEWED_COMMIT" =~ ^[0-9a-f]{40}$ || \
      ! "$CAMPAIGN" =~ ^[a-z0-9][a-z0-9_-]{2,79}$ ]]; then
  echo "Set exact REVIEWED_COMMIT and a safe LADDER_CAMPAIGN." >&2
elif [[ ! -x "$PYTHON" ]]; then
  echo "Approved Python 3.12 is unavailable." >&2
elif [[ ! -d "$RUN_ROOT/.git" ]] && \
     ! git clone https://github.com/ndandnd/EVSP-DR.git "$RUN_ROOT"; then
  echo "Public clone failed; no plan created." >&2
elif ! git -C "$RUN_ROOT" fetch origin "$REVIEWED_COMMIT" || \
     ! git -C "$RUN_ROOT" checkout --detach "$REVIEWED_COMMIT"; then
  echo "Detached reviewed checkout failed." >&2
elif [[ "$(git -C "$RUN_ROOT" rev-parse HEAD)" != "$REVIEWED_COMMIT" || \
        -n "$(git -C "$RUN_ROOT" status --porcelain)" ]]; then
  echo "Checkout is not exact and clean." >&2
elif [[ -e "$PLAN" || -e "$MATRIX" ]]; then
  echo "Plan/matrix exists; choose a new campaign." >&2
else
  mkdir -p "$PLAN_ROOT" "$RESERVATIONS"
  env -u PYTHONPATH PYTHONNOUSERSITE=1 \
    "$PYTHON" -I -B "$RUN_ROOT/src/run_reviewed_python.py" \
    "$REVIEWED_COMMIT" launch_scale_ladder.py \
    --campaign "$CAMPAIGN" --python "$PYTHON" \
    --reservation-root "$RESERVATIONS" \
    --plan-out "$PLAN" --matrix-out "$MATRIX"
  STATUS=$?
  if [[ "$STATUS" -ne 0 || ! -s "$PLAN" || ! -s "$MATRIX" ]]; then
    echo "Dry-run generation failed; no jobs submitted." >&2
  else
    PLAN_SHA=$(sha256sum "$PLAN" | awk '{print $1}')
    echo "PLAN: $PLAN"
    echo "TASK MATRIX: $MATRIX"
    echo "APPROVAL SHA-256: $PLAN_SHA"
    echo "EXPECTED TASKS: 86 (21 seed + 23 CG + 42 MIP; k40 MIP = 0)"
  fi
fi
```

## 2. Approval and submission

Run only after reviewing the complete JSON plan and task matrix from block 1.

```bash
APPROVED_PLAN_SHA256="${APPROVED_PLAN_SHA256:-}"

if [[ ! "$APPROVED_PLAN_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Set the exact APPROVED_PLAN_SHA256 printed by the dry run." >&2
elif [[ "$(sha256sum "$PLAN" 2>/dev/null | awk '{print $1}')" != \
        "$APPROVED_PLAN_SHA256" ]]; then
  echo "Saved plan differs from the approved SHA; nothing submitted." >&2
else
  env -u PYTHONPATH PYTHONNOUSERSITE=1 \
    "$PYTHON" -I -B "$RUN_ROOT/src/run_reviewed_python.py" \
    "$REVIEWED_COMMIT" launch_scale_ladder.py \
    --campaign "$CAMPAIGN" --python "$PYTHON" \
    --reservation-root "$RESERVATIONS" \
    --approved-plan-sha256 "$APPROVED_PLAN_SHA256" --submit
  STATUS=$?
  if [[ "$STATUS" -ne 0 ]]; then
    echo "Submission did not complete. Do not retry under another name; reconcile the recorded held gate and reservations." >&2
  fi
fi
```

All arrays depend on one held gate. MIP array task `i` uses `aftercorr` on CG
task `i`; KNOWN-PARTITION also depends on seed task `i`. The gate is released
only after all four arrays are accepted.

If submission stops with `gate_state=release_attempting` or
`held_release_failed`, do not resubmit. After `sacct` proves the recorded gate
completed, reconcile it:

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  "$PYTHON" -I -B "$RUN_ROOT/src/run_reviewed_python.py" \
  "$REVIEWED_COMMIT" reconcile_scale_ladder_gate.py \
  --campaign-root "$RUN_ROOT/src/results/scale_ladder/$CAMPAIGN" \
  --approved-plan-sha256 "$APPROVED_PLAN_SHA256"
```

If accounting still shows `PENDING` and bound `scontrol` proves
`Reason=JobHeldUser`, repeat that command with `--release-held-gate`; then run
it once more after `sacct` records the gate as `COMPLETED`.

## Normalize completed outputs

```bash
CAMPAIGN_ROOT="$RUN_ROOT/src/results/scale_ladder/$CAMPAIGN"
SUMMARY_ROOT="$CAMPAIGN_ROOT/summary"
K40_REUSE_MANIFEST="${K40_REUSE_MANIFEST:-}"

if [[ -e "$SUMMARY_ROOT" ]]; then
  echo "Summary exists; refusing overwrite." >&2
elif [[ -n "$K40_REUSE_MANIFEST" ]]; then
  env -u PYTHONPATH PYTHONNOUSERSITE=1 \
    "$PYTHON" -I -B "$RUN_ROOT/src/run_reviewed_python.py" \
    "$REVIEWED_COMMIT" summarize_scale_ladder.py \
    --campaign-root "$CAMPAIGN_ROOT" --out-dir "$SUMMARY_ROOT" \
    --k40-reuse-manifest "$K40_REUSE_MANIFEST"
else
  env -u PYTHONPATH PYTHONNOUSERSITE=1 \
    "$PYTHON" -I -B "$RUN_ROOT/src/run_reviewed_python.py" \
    "$REVIEWED_COMMIT" summarize_scale_ladder.py \
    --campaign-root "$CAMPAIGN_ROOT" --out-dir "$SUMMARY_ROOT"
fi
```

Absent or hash-incompatible k40 reuse artifacts remain explicit missing/censored
rows. They never trigger replacement k40 submissions.
