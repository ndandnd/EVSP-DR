# RAW k40 physical-gate MIP smoke

This package prepares four 30-minute, 8-thread, strict-partition MIP cells.
It never regenerates columns and is dry-run-only unless the exact plan SHA is
supplied again with `SUBMIT_SMOKE=1`.

The MIP runner and continuous realization module must remain byte-identical to
reviewed commit `e2b6939b5a5af7033acabec033f6b3d8dde3af4c`.

## Guarded Unicorn fetch / dry-run / submit

Run from an interactive Unicorn shell. This block deliberately does not use
`set -e`.

```bash
TAG="results-rawk40-mip-80058-physical-audit-20260817"
REVIEWED_SMOKE_COMMIT="${REVIEWED_SMOKE_COMMIT:-}"
SUBMIT_SMOKE="${SUBMIT_SMOKE:-0}"
APPROVED_PLAN_SHA256="${APPROVED_PLAN_SHA256:-}"
PYTHON="${EVSP_MIP_PYTHON:-$HOME/evsp_env/bin/python3.12}"
RUN_ROOT="$HOME/EVSP-DR-rawk40-physical-smoke"
FETCH_ROOT="$HOME/evsp_rawk40_physical_smoke_inputs"
PLAN_ROOT="$HOME/evsp_rawk40_physical_smoke_plans"
RESERVATIONS="$HOME/evsp_rawk40_physical_smoke_reservations"
CAMPAIGN="rawk40_physical_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
PLAN="$PLAN_ROOT/$CAMPAIGN.plan.json"

if [[ -z "$REVIEWED_SMOKE_COMMIT" ]]; then
  echo "Set REVIEWED_SMOKE_COMMIT to the reviewed packaging commit." >&2
elif [[ ! -d "$RUN_ROOT/.git" ]] && \
     ! git clone https://github.com/ndandnd/EVSP-DR.git "$RUN_ROOT"; then
  echo "Clone failed; nothing submitted." >&2
elif ! git -C "$RUN_ROOT" fetch origin "$REVIEWED_SMOKE_COMMIT"; then
  echo "Fetch failed; nothing submitted." >&2
elif ! git -C "$RUN_ROOT" checkout --detach "$REVIEWED_SMOKE_COMMIT"; then
  echo "Detached checkout failed; nothing submitted." >&2
elif [[ -n "$(git -C "$RUN_ROOT" status --porcelain)" ]]; then
  echo "Checkout is not clean; nothing submitted." >&2
else
  mkdir -p "$FETCH_ROOT" "$PLAN_ROOT" "$RESERVATIONS"
  if [[ ! -s "$FETCH_ROOT/rawk40_80058_bc0c386f_20260817T144707Z.tar.gz" ]]; then
    gh release download "$TAG" \
      --repo ndandnd/EVSP-DR \
      --dir "$FETCH_ROOT" \
      --pattern 'rawk40_80058_bc0c386f_20260817T144707Z.tar.gz*'
  fi

  ARCHIVE="$FETCH_ROOT/rawk40_80058_bc0c386f_20260817T144707Z.tar.gz"
  SIDE="$ARCHIVE.sha256"
  EXPECTED=$(awk 'NR==1 {print $1}' "$SIDE")
  ACTUAL=$(sha256sum "$ARCHIVE" | awk '{print $1}')

  if [[ "$EXPECTED" != "$ACTUAL" ]]; then
    echo "Archive checksum mismatch; nothing submitted." >&2
  else
    EXTRACT="$FETCH_ROOT/extracted"
    if [[ ! -d "$EXTRACT/src/results/mip_statistics/rawk40_80058_bc0c386f" ]]; then
      mkdir -p "$EXTRACT"
      tar -xzf "$ARCHIVE" -C "$EXTRACT"
    fi
    SOURCE="$EXTRACT/src/results/mip_statistics/rawk40_80058_bc0c386f"

    R1_CA="$SOURCE/input/k40_r1_ca_raw_m1440/k40r1_flat_CA.m1440.snapshot.json"
    R1_CS="$SOURCE/input/k40_r1_cs_raw_m1440/k40r1_flat_CS.m1440.snapshot.json"
    R2_CA="$SOURCE/input/k40_r2_ca_raw_m1440/k40r1_flat_CA.m1440.snapshot.json"
    R2_CS="$SOURCE/input/k40_r2_cs_raw_m1440/k40r1_flat_CS.m1440.snapshot.json"
    J1_CA="$R1_CA.columns.jsonl"
    J1_CS="$R1_CS.columns.jsonl"
    J2_CA="$R2_CA.columns.jsonl"
    J2_CS="$R2_CS.columns.jsonl"

    COMMON=(
      "$PYTHON" -u "$RUN_ROOT/src/launch_mip_statistics_campaign.py"
      --mode raw_k40_smoke
      --campaign "$CAMPAIGN"
      --python "$PYTHON"
      --reservation-root "$RESERVATIONS"
      --raw-k40-status "R1_CA=$R1_CA"
      --raw-k40-status "R1_CS=$R1_CS"
      --raw-k40-status "R2_CA=$R2_CA"
      --raw-k40-status "R2_CS=$R2_CS"
      --raw-k40-status-sha256 "R1_CA=$(sha256sum "$R1_CA" | awk '{print $1}')"
      --raw-k40-status-sha256 "R1_CS=$(sha256sum "$R1_CS" | awk '{print $1}')"
      --raw-k40-status-sha256 "R2_CA=$(sha256sum "$R2_CA" | awk '{print $1}')"
      --raw-k40-status-sha256 "R2_CS=$(sha256sum "$R2_CS" | awk '{print $1}')"
      --raw-k40-journal "R1_CA=$J1_CA"
      --raw-k40-journal "R1_CS=$J1_CS"
      --raw-k40-journal "R2_CA=$J2_CA"
      --raw-k40-journal "R2_CS=$J2_CS"
      --raw-k40-journal-sha256 "R1_CA=$(sha256sum "$J1_CA" | awk '{print $1}')"
      --raw-k40-journal-sha256 "R1_CS=$(sha256sum "$J1_CS" | awk '{print $1}')"
      --raw-k40-journal-sha256 "R2_CA=$(sha256sum "$J2_CA" | awk '{print $1}')"
      --raw-k40-journal-sha256 "R2_CS=$(sha256sum "$J2_CS" | awk '{print $1}')"
      --data-root "$SOURCE/input/k40_r1_ca_raw_m1440/data"
      --data-root "$SOURCE/input/k40_r1_cs_raw_m1440/data"
      --data-root "$SOURCE/input/k40_r2_ca_raw_m1440/data"
      --data-root "$SOURCE/input/k40_r2_cs_raw_m1440/data"
    )

    if [[ -e "$PLAN" ]]; then
      echo "Plan already exists; refusing to overwrite: $PLAN" >&2
    elif ! "${COMMON[@]}" --plan-out "$PLAN"; then
      echo "Dry-run plan generation failed; nothing submitted." >&2
    elif ! "$PYTHON" "$RUN_ROOT/src/validate_raw_k40_mip_plan.py" \
        "$PLAN" --expected-commit "$REVIEWED_SMOKE_COMMIT" \
        --expected-mode raw_k40_smoke; then
      echo "Plan validation failed; nothing submitted." >&2
    else
      OBSERVED_PLAN_SHA=$(sha256sum "$PLAN" | awk '{print $1}')
      echo "DRY RUN COMPLETE: $PLAN"
      echo "PLAN SHA-256: $OBSERVED_PLAN_SHA"
      if [[ "$SUBMIT_SMOKE" == "1" && \
            "$APPROVED_PLAN_SHA256" == "$OBSERVED_PLAN_SHA" ]]; then
        "${COMMON[@]}" \
          --approved-plan-sha256 "$APPROVED_PLAN_SHA256" \
          --submit
      else
        echo "NOT SUBMITTED. Review the plan, then set SUBMIT_SMOKE=1 and the exact APPROVED_PLAN_SHA256."
      fi
    fi
  fi
fi
```

## Acceptance

- CA `INFEASIBLE` is a valid finite-pool outcome.
- Any incumbent must pass final continuous replay and persisted tariff-block
  validation.
- `physical_pool_audit.rejected_columns` must equal zero.
- Preprocessing and Gurobi wall times must both be finite and separate.
- No continuous-cost pricing certificate may be claimed.
- Checkpoints remain observational; they are not Gurobi tree restarts.
