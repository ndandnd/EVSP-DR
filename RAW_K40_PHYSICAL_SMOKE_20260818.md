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
SMOKE_CAMPAIGN="${SMOKE_CAMPAIGN:-}"
APPROVED_PLAN_SHA256="${APPROVED_PLAN_SHA256:-}"
PYTHON="${EVSP_MIP_PYTHON:-$HOME/evsp_env/bin/python3.12}"
RUN_ROOT="$HOME/EVSP-DR-rawk40-physical-smoke"
FETCH_ROOT="$HOME/evsp_rawk40_physical_smoke_inputs"
PLAN_ROOT="$HOME/evsp_rawk40_physical_smoke_plans"
RESERVATIONS="$HOME/evsp_rawk40_physical_smoke_reservations"
CAMPAIGN="$SMOKE_CAMPAIGN"
PLAN="$PLAN_ROOT/$CAMPAIGN.plan.json"
PINNED_ARCHIVE_SHA256="65dd198b26dcd5c5a512108445bb8089b30ae0821d19ab5fe26bebdaf17f7bb2"

if [[ -z "$REVIEWED_SMOKE_COMMIT" || -z "$SMOKE_CAMPAIGN" ]]; then
  echo "Set REVIEWED_SMOKE_COMMIT and one persistent lowercase SMOKE_CAMPAIGN." >&2
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

  if [[ "$EXPECTED" != "$PINNED_ARCHIVE_SHA256" || \
        "$ACTUAL" != "$PINNED_ARCHIVE_SHA256" ]]; then
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

    R1_CA_SHA="b3579a6f2ce9b0c0d5d41e7503f9c9fa2e1d3714787e8ae3298a1b5be2fc8cd5"
    R1_CS_SHA="04c3d5d9fe701fbb3bc4fd343e58480fabebf27bb18ef2c60e23a34e29b0200b"
    R2_CA_SHA="e8dbe6c2107a7cf810dc4c77317853a5339ea273f89af1e35908d76a97ec7e32"
    R2_CS_SHA="780431fea40763d42576272bd8e9260f3ed2c8541b6d77f751e17f342dfb1202"
    J1_CA_SHA="ab095590c6cd65d38ad37b4c957455fff4e2bafee1a6441960f3ca999a2b9ff4"
    J1_CS_SHA="128e3d841842bba08e4eba2d9a073322caa8a1de5c64c0e9efe2e747a08c01d4"
    J2_CA_SHA="900d9be8ea65f38117e3b8d0c34aba87970910cb9d97d7175ac6f336ba4ea756"
    J2_CS_SHA="8290771a7ca3b6f185070f68a9934e6eaa8894c802ae02ac37f013c25a4b7c31"

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
      --raw-k40-status-sha256 "R1_CA=$R1_CA_SHA"
      --raw-k40-status-sha256 "R1_CS=$R1_CS_SHA"
      --raw-k40-status-sha256 "R2_CA=$R2_CA_SHA"
      --raw-k40-status-sha256 "R2_CS=$R2_CS_SHA"
      --raw-k40-journal "R1_CA=$J1_CA"
      --raw-k40-journal "R1_CS=$J1_CS"
      --raw-k40-journal "R2_CA=$J2_CA"
      --raw-k40-journal "R2_CS=$J2_CS"
      --raw-k40-journal-sha256 "R1_CA=$J1_CA_SHA"
      --raw-k40-journal-sha256 "R1_CS=$J1_CS_SHA"
      --raw-k40-journal-sha256 "R2_CA=$J2_CA_SHA"
      --raw-k40-journal-sha256 "R2_CS=$J2_CS_SHA"
      --data-root "$SOURCE/input/k40_r1_ca_raw_m1440/data"
      --data-root "$SOURCE/input/k40_r1_cs_raw_m1440/data"
      --data-root "$SOURCE/input/k40_r2_ca_raw_m1440/data"
      --data-root "$SOURCE/input/k40_r2_cs_raw_m1440/data"
    )

    if ! (
      [[ "$(sha256sum "$R1_CA" | awk '{print $1}')" == "$R1_CA_SHA" ]] &&
      [[ "$(sha256sum "$R1_CS" | awk '{print $1}')" == "$R1_CS_SHA" ]] &&
      [[ "$(sha256sum "$R2_CA" | awk '{print $1}')" == "$R2_CA_SHA" ]] &&
      [[ "$(sha256sum "$R2_CS" | awk '{print $1}')" == "$R2_CS_SHA" ]] &&
      [[ "$(sha256sum "$J1_CA" | awk '{print $1}')" == "$J1_CA_SHA" ]] &&
      [[ "$(sha256sum "$J1_CS" | awk '{print $1}')" == "$J1_CS_SHA" ]] &&
      [[ "$(sha256sum "$J2_CA" | awk '{print $1}')" == "$J2_CA_SHA" ]] &&
      [[ "$(sha256sum "$J2_CS" | awk '{print $1}')" == "$J2_CS_SHA" ]]
    ); then
      echo "Extracted status/journal hash mismatch; nothing submitted." >&2
    elif [[ "$SUBMIT_SMOKE" != "1" && -e "$PLAN" ]]; then
      echo "Dry-run plan already exists; choose a new SMOKE_CAMPAIGN." >&2
    elif [[ "$SUBMIT_SMOKE" != "1" ]] && \
         ! "${COMMON[@]}" --plan-out "$PLAN"; then
      echo "Dry-run plan generation failed; nothing submitted." >&2
    elif [[ ! -s "$PLAN" ]] || \
         ! "$PYTHON" "$RUN_ROOT/src/validate_raw_k40_mip_plan.py" \
           "$PLAN" --expected-commit "$REVIEWED_SMOKE_COMMIT" \
           --expected-mode raw_k40_smoke; then
      echo "Plan is missing or failed validation; nothing submitted." >&2
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
