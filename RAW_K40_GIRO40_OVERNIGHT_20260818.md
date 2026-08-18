# Controlled k40 RAW versus GIRO40-AUGMENTED campaign

This package contains exactly four strict-partition Scaglione jobs:

| Cell | Treatment | Frozen pool | Gurobi limit | Threads |
|---|---|---|---:|---:|
| R1-CS RAW | `RAW` | R1-CS m1440 | 8 h | 8 |
| R2-CS RAW | `RAW` | R2-CS m1440 | 8 h | 8 |
| R1-CS GIRO40 | `GIRO40-AUGMENTED` | R1-CS m1440 + 40 routes | 2 h | 8 |
| R2-CS GIRO40 | `GIRO40-AUGMENTED` | R2-CS m1440 + 40 routes | 2 h | 8 |

CA is not scheduled. Its finite-pool infeasibility remains historical evidence.
The prior 30-minute smoke is also separate evidence and is not concatenated
with these trajectories.

The GIRO40 pool mixes conservative expanded-grid costs for base columns with
continuous realized costs for injected duties. Its solver stage therefore
compares and may prove fleet count only; it never claims an augmented
route-cost optimum or a continuous-cost pricing certificate.

The reviewed GIRO input is
`analysis/k40_giro40_partition_20260818/giro40_partition.json`, SHA-256
`8f9944f93f26cf0121e9ecab2fa412d573e90a0189b7a38008d3b2535f54d428`.
It has 40 routes, covers all 947 trips exactly once, and excludes weekday
variants `13316m` and `13324muw` in favor of `13316uwt` and `13324t`.

## Guarded public fetch, dry run, and optional submission

Run this in an interactive Unicorn shell. It deliberately does not use
`set -e` and never exits the login shell. The first invocation only writes and
validates an immutable plan. Submission requires a second invocation with the
same campaign name, the exact observed plan hash, and `SUBMIT_OVERNIGHT=1`.

```bash
TAG="results-rawk40-mip-80058-physical-audit-20260817"
REVIEWED_COMMIT="${REVIEWED_COMMIT:-}"
OVERNIGHT_CAMPAIGN="${OVERNIGHT_CAMPAIGN:-}"
APPROVED_PLAN_SHA256="${APPROVED_PLAN_SHA256:-}"
SUBMIT_OVERNIGHT="${SUBMIT_OVERNIGHT:-0}"
PYTHON="${EVSP_MIP_PYTHON:-$HOME/evsp_env/bin/python3.12}"
RUN_ROOT="$HOME/EVSP-DR-k40-cs-overnight"
FETCH_ROOT="$HOME/evsp_k40_cs_overnight_inputs"
PLAN_ROOT="$HOME/evsp_k40_cs_overnight_plans"
RESERVATIONS="$HOME/evsp_k40_cs_overnight_reservations"
ARCHIVE_NAME="rawk40_80058_bc0c386f_20260817T144707Z.tar.gz"
ARCHIVE_URL="https://github.com/ndandnd/EVSP-DR/releases/download/$TAG/$ARCHIVE_NAME"
SIDE_URL="$ARCHIVE_URL.sha256"
PINNED_ARCHIVE_SHA256="65dd198b26dcd5c5a512108445bb8089b30ae0821d19ab5fe26bebdaf17f7bb2"
GIRO40_SHA256="8f9944f93f26cf0121e9ecab2fa412d573e90a0189b7a38008d3b2535f54d428"
CAMPAIGN="$OVERNIGHT_CAMPAIGN"
PLAN="$PLAN_ROOT/$CAMPAIGN.plan.json"

if [[ -z "$REVIEWED_COMMIT" || -z "$OVERNIGHT_CAMPAIGN" ]]; then
  echo "Set REVIEWED_COMMIT and a persistent lowercase OVERNIGHT_CAMPAIGN." >&2
elif [[ ! "$REVIEWED_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "REVIEWED_COMMIT must be one exact 40-character commit." >&2
elif [[ ! "$OVERNIGHT_CAMPAIGN" =~ ^[a-z0-9][a-z0-9._-]{2,79}$ ]]; then
  echo "OVERNIGHT_CAMPAIGN is not a safe campaign identifier." >&2
elif [[ ! -x "$PYTHON" ]]; then
  echo "The approved Python 3.12 executable is unavailable." >&2
elif [[ ! -d "$RUN_ROOT/.git" ]] && \
     ! git clone https://github.com/ndandnd/EVSP-DR.git "$RUN_ROOT"; then
  echo "Public clone failed; nothing submitted." >&2
elif ! git -C "$RUN_ROOT" fetch origin "$REVIEWED_COMMIT"; then
  echo "Reviewed commit fetch failed; nothing submitted." >&2
elif ! git -C "$RUN_ROOT" checkout --detach "$REVIEWED_COMMIT"; then
  echo "Detached checkout failed; nothing submitted." >&2
elif [[ "$(git -C "$RUN_ROOT" rev-parse HEAD)" != "$REVIEWED_COMMIT" ]]; then
  echo "Checkout commit differs; nothing submitted." >&2
elif [[ -n "$(git -C "$RUN_ROOT" status --porcelain)" ]]; then
  echo "Checkout is not tracked-clean; nothing submitted." >&2
else
  mkdir -p "$FETCH_ROOT" "$PLAN_ROOT" "$RESERVATIONS"
  ARCHIVE="$FETCH_ROOT/$ARCHIVE_NAME"
  SIDE="$ARCHIVE.sha256"
  if [[ ! -s "$ARCHIVE" ]]; then
    curl -fL --retry 3 --retry-delay 3 \
      "$ARCHIVE_URL" -o "$ARCHIVE.download"
    CURL_RC=$?
    if [[ "$CURL_RC" -eq 0 ]]; then
      mv -n "$ARCHIVE.download" "$ARCHIVE"
    else
      rm -f "$ARCHIVE.download"
    fi
  fi
  if [[ ! -s "$SIDE" ]]; then
    curl -fL --retry 3 --retry-delay 3 \
      "$SIDE_URL" -o "$SIDE.download"
    SIDE_RC=$?
    if [[ "$SIDE_RC" -eq 0 ]]; then
      mv -n "$SIDE.download" "$SIDE"
    else
      rm -f "$SIDE.download"
    fi
  fi
  EXPECTED=$(awk 'NR==1 {print $1}' "$SIDE" 2>/dev/null)
  ACTUAL=$(sha256sum "$ARCHIVE" 2>/dev/null | awk '{print $1}')
  GIRO40="$RUN_ROOT/analysis/k40_giro40_partition_20260818/giro40_partition.json"
  OBSERVED_GIRO40=$(sha256sum "$GIRO40" 2>/dev/null | awk '{print $1}')

  if [[ "$EXPECTED" != "$PINNED_ARCHIVE_SHA256" || \
        "$ACTUAL" != "$PINNED_ARCHIVE_SHA256" ]]; then
    echo "Release archive checksum mismatch; nothing submitted." >&2
  elif [[ "$OBSERVED_GIRO40" != "$GIRO40_SHA256" ]]; then
    echo "Reviewed GIRO40 partition checksum mismatch; nothing submitted." >&2
  else
    EXTRACT="$FETCH_ROOT/extracted"
    SOURCE="$EXTRACT/src/results/mip_statistics/rawk40_80058_bc0c386f"
    if [[ ! -d "$SOURCE" ]]; then
      TMP_EXTRACT="$FETCH_ROOT/.extract.$$"
      rm -rf "$TMP_EXTRACT"
      mkdir "$TMP_EXTRACT"
      if tar -xzf "$ARCHIVE" -C "$TMP_EXTRACT"; then
        if [[ -e "$EXTRACT" ]]; then
          echo "Extraction target appeared concurrently; nothing submitted." >&2
          rm -rf "$TMP_EXTRACT"
        else
          mv "$TMP_EXTRACT" "$EXTRACT"
        fi
      else
        echo "Archive extraction failed; nothing submitted." >&2
        rm -rf "$TMP_EXTRACT"
      fi
    fi

    R1_CS="$SOURCE/input/k40_r1_cs_raw_m1440/k40r1_flat_CS.m1440.snapshot.json"
    R2_CS="$SOURCE/input/k40_r2_cs_raw_m1440/k40r1_flat_CS.m1440.snapshot.json"
    J1_CS="$R1_CS.columns.jsonl"
    J2_CS="$R2_CS.columns.jsonl"
    R1_CS_SHA="04c3d5d9fe701fbb3bc4fd343e58480fabebf27bb18ef2c60e23a34e29b0200b"
    R2_CS_SHA="780431fea40763d42576272bd8e9260f3ed2c8541b6d77f751e17f342dfb1202"
    J1_CS_SHA="128e3d841842bba08e4eba2d9a073322caa8a1de5c64c0e9efe2e747a08c01d4"
    J2_CS_SHA="8290771a7ca3b6f185070f68a9934e6eaa8894c802ae02ac37f013c25a4b7c31"

    COMMON=(
      "$PYTHON" -B -u "$RUN_ROOT/src/launch_mip_statistics_campaign.py"
      --mode k40_cs_overnight
      --campaign "$CAMPAIGN"
      --python "$PYTHON"
      --reservation-root "$RESERVATIONS"
      --raw-k40-status "R1_CS=$R1_CS"
      --raw-k40-status "R2_CS=$R2_CS"
      --raw-k40-status-sha256 "R1_CS=$R1_CS_SHA"
      --raw-k40-status-sha256 "R2_CS=$R2_CS_SHA"
      --raw-k40-journal "R1_CS=$J1_CS"
      --raw-k40-journal "R2_CS=$J2_CS"
      --raw-k40-journal-sha256 "R1_CS=$J1_CS_SHA"
      --raw-k40-journal-sha256 "R2_CS=$J2_CS_SHA"
      --giro-start "R1_CS=$GIRO40"
      --giro-start "R2_CS=$GIRO40"
      --data-root "$SOURCE/input/k40_r1_cs_raw_m1440/data"
      --data-root "$SOURCE/input/k40_r2_cs_raw_m1440/data"
    )

    if ! {
      [[ "$(sha256sum "$R1_CS" | awk '{print $1}')" == "$R1_CS_SHA" ]] &&
      [[ "$(sha256sum "$R2_CS" | awk '{print $1}')" == "$R2_CS_SHA" ]] &&
      [[ "$(sha256sum "$J1_CS" | awk '{print $1}')" == "$J1_CS_SHA" ]] &&
      [[ "$(sha256sum "$J2_CS" | awk '{print $1}')" == "$J2_CS_SHA" ]]
    }; then
      echo "Extracted status/journal hash mismatch; nothing submitted." >&2
    elif [[ "$SUBMIT_OVERNIGHT" != "1" && -e "$PLAN" ]]; then
      echo "Plan exists; choose a new campaign or review the existing plan." >&2
    elif [[ "$SUBMIT_OVERNIGHT" != "1" ]] && \
         ! "${COMMON[@]}" --plan-out "$PLAN"; then
      echo "Dry-run generation failed; nothing submitted." >&2
    elif [[ ! -s "$PLAN" ]] || \
         ! "$PYTHON" -B "$RUN_ROOT/src/validate_k40_cs_overnight_plan.py" \
           "$PLAN" --expected-commit "$REVIEWED_COMMIT"; then
      echo "Plan is missing or failed validation; nothing submitted." >&2
    else
      OBSERVED_PLAN_SHA=$(sha256sum "$PLAN" | awk '{print $1}')
      echo "DRY RUN COMPLETE: $PLAN"
      echo "PLAN SHA-256: $OBSERVED_PLAN_SHA"
      if [[ "$SUBMIT_OVERNIGHT" == "1" && \
            "$APPROVED_PLAN_SHA256" == "$OBSERVED_PLAN_SHA" ]]; then
        "${COMMON[@]}" \
          --approved-plan-sha256 "$APPROVED_PLAN_SHA256" \
          --submit
      else
        echo "NOT SUBMITTED. Review the plan, then set SUBMIT_OVERNIGHT=1 and the exact APPROVED_PLAN_SHA256."
      fi
    fi
  fi
fi
```

The launcher refuses an existing campaign/output path, an existing execution
reservation, or a matching Slurm execution digest. Job names encode replicate,
treatment, and budget and remain at most 15 characters.

## Post-campaign summary and immutable archive

Only after all four jobs finish successfully, use fresh output names:

```bash
CAMPAIGN_ROOT="$RUN_ROOT/src/results/mip_statistics/$OVERNIGHT_CAMPAIGN"
SUMMARY_ROOT="$HOME/evsp_k40_cs_summaries/$OVERNIGHT_CAMPAIGN"
ARCHIVE_ROOT="$HOME/evsp_k40_cs_archives"
ARCHIVE="$ARCHIVE_ROOT/$OVERNIGHT_CAMPAIGN.tar.gz"

if [[ -e "$SUMMARY_ROOT" || -e "$ARCHIVE" || -e "$ARCHIVE.sha256" ]]; then
  echo "Summary/archive target exists; choose a new immutable destination." >&2
elif ! "$PYTHON" -B "$RUN_ROOT/src/summarize_mip_statistics.py" \
       --campaign-root "$CAMPAIGN_ROOT" --out-dir "$SUMMARY_ROOT"; then
  echo "Summary validation failed; no archive created." >&2
else
  mkdir -p "$ARCHIVE_ROOT"
  TMP_ARCHIVE="$ARCHIVE.tmp.$$"
  ARCHIVE_STAGE="$ARCHIVE_ROOT/.stage.$$"
  ARCHIVE_STAGE_CREATED=0
  if mkdir "$ARCHIVE_STAGE"; then
    ARCHIVE_STAGE_CREATED=1
  fi
  if [[ "$ARCHIVE_STAGE_CREATED" != "1" ]] || \
     ! cp -a "$CAMPAIGN_ROOT" "$ARCHIVE_STAGE/campaign" || \
     ! cp -a "$SUMMARY_ROOT" "$ARCHIVE_STAGE/summary"; then
    echo "Archive staging failed." >&2
  elif tar --sort=name --mtime='UTC 1970-01-01' \
       --owner=0 --group=0 --numeric-owner \
       -czf "$TMP_ARCHIVE" -C "$ARCHIVE_STAGE" campaign summary; then
    ARCHIVE_DIGEST=$(sha256sum "$TMP_ARCHIVE" | awk '{print $1}')
    TMP_SIDE="$ARCHIVE.sha256.tmp.$$"
    printf '%s  %s\n' "$ARCHIVE_DIGEST" "$(basename "$ARCHIVE")" > "$TMP_SIDE"
    ARCHIVE_LINKED=0
    if [[ "$ARCHIVE_DIGEST" =~ ^[0-9a-f]{64}$ ]] && \
       ln "$TMP_ARCHIVE" "$ARCHIVE"; then
      ARCHIVE_LINKED=1
      if ln "$TMP_SIDE" "$ARCHIVE.sha256"; then
        rm -f "$TMP_ARCHIVE" "$TMP_SIDE"
        echo "ARCHIVE: $ARCHIVE"
        echo "CHECKSUM: $ARCHIVE.sha256"
      else
        echo "Checksum target appeared concurrently; archive withdrawn." >&2
        rm -f "$ARCHIVE" "$TMP_ARCHIVE" "$TMP_SIDE"
      fi
    else
      echo "Archive target appeared concurrently; no overwrite performed." >&2
      if [[ "$ARCHIVE_LINKED" == "1" ]]; then
        rm -f "$ARCHIVE"
      fi
      rm -f "$TMP_ARCHIVE" "$TMP_SIDE"
    fi
  else
    echo "Archive creation failed." >&2
    rm -f "$TMP_ARCHIVE"
  fi
  if [[ "$ARCHIVE_STAGE_CREATED" == "1" ]]; then
    rm -rf "$ARCHIVE_STAGE"
  fi
fi
```

The summary produces `mip_checkpoint_long.csv`, `mip_run_summary.csv`,
`artifact_inventory.csv`, `raw_vs_giro40_comparison.csv`, and deterministic
PNG/PDF fleet/bound-versus-time figures. Every proof scope is finite-pool.
No continuous-cost pricing certificate is permitted.
