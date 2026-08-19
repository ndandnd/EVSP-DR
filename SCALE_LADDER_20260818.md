# Exact-CG scale ladder: guarded Unicorn launch

Primary cells use only the historical flat tariff
`data/hourly_prices_flat.csv`, SHA-256
`1f51f2e1f6ca303838ebaaf6272a28ff2d6bbee97146cb04d330e10f191f8200`.
The tracked input manifest proves equivalence to `flat_h26.csv`, but the
campaign continues to bind the historical bytes.

The reviewed plan contains exactly:

- 22 instance-level known-route membership preflights;
- 21 known-partition preparation tasks;
- 23 primary exact-CG tasks;
- 30 k2/k3/k5 sensitivity CG tasks, including three 1-kWh/5-minute k2
  route-space diagnostics;
- 21 RAW MIPs and 21 KNOWN-PARTITION diagnostic MIPs;
- 138 scientific/diagnostic tasks total;
- 3 infrastructure tasks: two environment probes and one activation
  controller;
- zero k40 MIP submissions (four reuse-only result slots).

The held-job false-success incident and the resulting scheduler-state contract
are documented in `SCALE_LADDER_RELEASE_INCIDENT_20260819.md`. Read that note
before changing probe, activation, gate, dependency, or restart logic. In
particular, a successful `scontrol` exit status is never a state-change
postcondition.

The portable-environment gate binds the exact interpreter and package bytes,
versions, architecture, compiler/build dependencies, NumPy compiled SIMD
baseline, and compiled dispatch set. NumPy's host-detected SIMD
`found`/`not found` partition is recorded in each probe's node metadata but is
not a compatibility condition: identical NumPy wheels legitimately dispatch
different kernels on heterogeneous Unicorn CPUs. The compared policy marker is
`numpy-config-v2-runtime-simd-separated`, so an older or missing policy fails
closed and cannot reuse this campaign's approval hash.

## One-block preparation and launch

Set `COMMIT` to the exact reviewed 40-character commit. Keep the campaign name
fixed on every rerun; changing it creates a different campaign. Running this
block with `EVSP_LADDER_SUBMIT=YES` is the explicit launch approval. The wrapper
first performs and records the dry run, checks the printed approval hash against
the saved plan, verifies all task counts/hashes/physics, and only then submits.
It does not enable `set -e` and cannot terminate the surrounding login shell.

```bash
bash <<'BASH'
main() {
  COMMIT="PASTE_THE_REVIEWED_40_CHARACTER_COMMIT_HERE"
  CAMPAIGN="slad_flat_primary_v2"
  SOURCE_ROOT="$HOME/EVSP-DR"
  SCRIPT="$HOME/launch_scale_ladder_probe_first.sh"

  if [[ ! "$COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Replace COMMIT with the exact reviewed 40-character SHA." >&2
    return 1
  fi
  if [[ ! -d "$SOURCE_ROOT/.git" ]]; then
    git clone https://github.com/ndandnd/EVSP-DR.git "$SOURCE_ROOT" || return 1
  fi
  git -C "$SOURCE_ROOT" fetch origin "$COMMIT" || return 1
  git -C "$SOURCE_ROOT" show \
    "$COMMIT:scripts/launch_scale_ladder_probe_first.sh" >"$SCRIPT" || return 1
  chmod 700 "$SCRIPT" || return 1

  REVIEWED_COMMIT="$COMMIT" \
  LADDER_CAMPAIGN="$CAMPAIGN" \
  EVSP_LADDER_SUBMIT=YES \
  EVSP_LADDER_RETRY= \
    bash "$SCRIPT"
  STATUS=$?
  if [[ "$STATUS" -ne 0 ]]; then
    echo "Launch stopped fail-closed; inspect its printed manifest/log paths." >&2
  fi
  return "$STATUS"
}
main
BASH
```

The successful top-level return is `INFRASTRUCTURE_ARMED=true`, not “138 jobs
running.” It proves that two held probes and one held activation controller had
durable numeric IDs and that their exact identity-bound scheduler states were
observed after release; raw `scontrol` exit status is not accepted as proof.
The manifest retains each release observation and command-attempt count. At
that instant scientific work is still absent, unless the controller already
won the harmless post-return race and advanced the manifest. The controller
validates both exact probe artifacts and Slurm identities before creating
reservations, one held scientific gate, and six arrays. It releases the gate
only after all six array IDs are durable and applies the same observed-state
postcondition to that release.

Dependencies are fixed: primary CG and sensitivity CG wait for complete
PREFLIGHT; MIP task `i` uses `aftercorr` on primary-CG task `i`; the
KNOWN-PARTITION MIP also depends on seed task `i`. Scientific task count is 138
and infrastructure count is reported separately as 3.

### One-time closeout of the stalled `ba09d46` campaign

`scripts/replace_stalled_scale_ladder_20260819.sh` is deliberately bound to
the seven zero-runtime jobs `218102`--`218108`, failed probes `218196` and
`218197`, plan SHA `bcea6b9...`, and source commit `ba09d46...`. It is not a
general cancellation utility. Before cancelling anything it verifies the old
plan, manifest, controller identities, job names, partitions, comments, array
ranges, dependencies, states, and failed probe artifacts. It cancels the six
dependent arrays before the gate only after publishing a checksummed
pre-cancel archive containing scheduler captures, external plan/matrix status,
and the manifest-bound reservation files. It then proves every array task and
the gate have zero-runtime `CANCELLED` accounting, atomically publishes a
checksummed receipt bundle, and enters the reviewed launcher with fresh
campaign and reservation paths. Partial prior cancellation is recovered one
exact job at a time; an identity mismatch, an unproved absent task, or an
invalid receipt stops the replacement before a new campaign is submitted.

## Recovery hierarchy

Do not infer a retry from a failed command. First inspect `campaign.json`, the
bound `.out/.err` files, `squeue`, `scontrol`, and `sacct`.

Before any scientific gate/reservation/array exists (`gate_state=not_created`):

- a scheduler-terminal or publication-incomplete probe may be retried only by
  rerunning the same wrapper and campaign with
  `EVSP_LADDER_RETRY=failed_probes`;
- a scheduler-terminal activation controller may be retried only with
  `EVSP_LADDER_RETRY=failed_activation`.

Both retries still require `EVSP_LADDER_SUBMIT=YES`. A valid environment or
identity mismatch is never retryable. An explicitly authorized retry is durable:
rerun the same flag after a client/process interruption. Each invocation either
recovers its exact attempt, advances one terminal attempt, or refuses while the
attempt is live/ambiguous; it never silently creates a new campaign.

There is one deliberately fail-closed crash window: a process can die after a
submission intent is fsynced but before `sbatch` is invoked. That state is
indistinguishable from scheduler acceptance whose job is not yet visible. The
launcher/reconciler performs bounded exact discovery and then refuses; do not
retry automatically. Preserve the manifest and logs, perform an exact operator
Slurm audit, and do not replace anything until every exact held job from the
old intent is cancelled or proven absent. Manually close out the campaign; if a
replacement is approved, use both a new campaign name and an intentionally new
`EVSP_LADDER_RESERVATIONS` directory. Reusing the default reservation path with
a different plan hash correctly fails closed. This sacrifices automatic
liveness rather than risking duplicate jobs.

Example (reuse the same exact values and reviewed wrapper):

```bash
REVIEWED_COMMIT="$COMMIT" LADDER_CAMPAIGN="$CAMPAIGN" \
EVSP_LADDER_SUBMIT=YES EVSP_LADDER_RETRY=failed_probes bash "$SCRIPT"

REVIEWED_COMMIT="$COMMIT" LADDER_CAMPAIGN="$CAMPAIGN" \
EVSP_LADDER_SUBMIT=YES EVSP_LADDER_RETRY=failed_activation bash "$SCRIPT"
```

After scientific submission has begun (a gate ID, reservation, or array is
present), do not use either infrastructure retry. Public gate reconciliation is
serialized by the campaign lock and is the only operator recovery path:

```bash
RUN_ROOT="$HOME/EVSP-DR-scale-ladder-${COMMIT:0:12}"
PLAN="$HOME/evsp_scale_ladder_plans/$CAMPAIGN.plan.json"
PLAN_SHA=$(sha256sum "$PLAN" | awk '{print $1}')
CAMPAIGN_ROOT=$(jq -r '.campaign_root' "$PLAN")
PYTHON=$(readlink -f "$HOME/evsp_env/bin/python3.12")

env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON" -I -B "$RUN_ROOT/src/run_reviewed_python.py" \
  "$COMMIT" reconcile_scale_ladder_gate.py \
  --campaign-root "$CAMPAIGN_ROOT" \
  --approved-plan-sha256 "$PLAN_SHA" \
  --resume-missing-arrays --release-held-gate
```

The reconciler exact-matches the user, parent ID, name, state, partition,
reason, comment, array range, and dependency semantics. Gate completion is
accepted only with an exact `0:0` exit code. Accepted-before-record submission
intents are boundedly rediscovered and fail closed rather than duplicated.
A terminal non-success gate is durably recorded before the command raises:
`gate_state=terminal_failed`, the exact scheduler observation/source/state/exit
code, and `submitted=false`.

## Normalize completed outputs

### Legacy `7937c22` campaign: read-only post-hoc audit first

The already-running `slad_flat_primary_v4_7937c22` predates prospective
receipt fields. Do not edit its manifest or outputs. After all intended worker
completions exist, create a separate no-clobber sidecar:

```bash
LEGACY_ROOT="$HOME/EVSP-DR-scale-ladder-7937c22/src/results/scale_ladder/slad_flat_primary_v4_7937c22"
LEGACY_AUDIT_ROOT="$HOME/evsp_scale_ladder_legacy_audits"
LEGACY_SIDECAR="$LEGACY_AUDIT_ROOT/slad_flat_primary_v4_7937c22.audit.json"
mkdir -p "$LEGACY_AUDIT_ROOT"

env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON" -I -B "$RUN_ROOT/src/run_reviewed_python.py" \
  "$COMMIT" audit_legacy_scale_ladder_campaign.py \
  --campaign-root "$LEGACY_ROOT" \
  --sidecar-out "$LEGACY_SIDECAR" \
  --expected-commit 7937c22fef7771e2f74dd03569ea852cbd805e1c
```

Without a separately captured and validated scheduler JSON, the sidecar is
explicitly `legacy_scheduler_unverified`. Supplying
`--scheduler-capture CAPTURE.json` can yield `legacy_posthoc_audited` only when
the capture uses schema
`evsp-dr-legacy-scale-ladder-raw-scheduler-capture-v2`, includes raw per-task
`scontrol -o` and `sacct -P` records plus approved tool hashes, and the
production parsers prove exact gate/array IDs, names, comments, partitions,
dependencies, task coverage, terminal states, and `0:0` exit codes. Hand-
normalized assertions cannot upgrade the label. The audit writes
`$LEGACY_SIDECAR` and `$LEGACY_SIDECAR.sha256` atomically without replacing
anything. It fails on missing completions, changed journals/checkpoints,
duplicate tasks, incompatible inputs, selected-route/physical evidence
mismatch, mixed old/new scheduler evidence, or any hash change.

Normalization then requires the sidecar explicitly:

```bash
"$PYTHON" -I -B "$RUN_ROOT/src/run_reviewed_python.py" \
  "$COMMIT" summarize_scale_ladder.py \
  --campaign-root "$LEGACY_ROOT" \
  --out-dir "$LEGACY_ROOT/summary-posthoc" \
  --legacy-audit-sidecar "$LEGACY_SIDECAR"
```

The normalized provenance retains the legacy evidence label and never claims
the prospective lifecycle guarantees.

```bash
RUN_ROOT="$HOME/EVSP-DR-scale-ladder-${COMMIT:0:12}"
PLAN="$HOME/evsp_scale_ladder_plans/$CAMPAIGN.plan.json"
CAMPAIGN_ROOT=$(jq -r '.campaign_root' "$PLAN")
SUMMARY_ROOT="$CAMPAIGN_ROOT/summary"
PYTHON=$(readlink -f "$HOME/evsp_env/bin/python3.12")
K40_REUSE_MANIFEST="${K40_REUSE_MANIFEST:-}"

if [[ -e "$SUMMARY_ROOT" ]]; then
  echo "Summary exists; refusing overwrite." >&2
elif [[ -n "$K40_REUSE_MANIFEST" ]]; then
  env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    "$PYTHON" -I -B "$RUN_ROOT/src/run_reviewed_python.py" \
    "$COMMIT" summarize_scale_ladder.py \
    --campaign-root "$CAMPAIGN_ROOT" --out-dir "$SUMMARY_ROOT" \
    --k40-reuse-manifest "$K40_REUSE_MANIFEST"
else
  env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    "$PYTHON" -I -B "$RUN_ROOT/src/run_reviewed_python.py" \
    "$COMMIT" summarize_scale_ladder.py \
    --campaign-root "$CAMPAIGN_ROOT" --out-dir "$SUMMARY_ROOT"
fi
```

Absent or hash-incompatible k40 reuse artifacts remain explicit
missing/censored rows. They never trigger replacement k40 submissions.
Normalized target diagnostics use `target_route_weight_observed`; this is a
combined-cost-master observation, not a certified minimum-fleet bound. When
the known comparator is outside the grid, the interpretation is
`known_comparator_invalid_scaling_unresolved`.

## Local diagnostic launcher

This uses only SEED/membership and exact-CG phases—never Slurm or Gurobi—and
limits concurrent subprocesses to three by default:

```bash
LOCAL_DIAGNOSTIC_ROOT="$HOME/evsp_scale_ladder_local/k2-check"

if [[ -e "$LOCAL_DIAGNOSTIC_ROOT" ]]; then
  echo "Local diagnostic output exists; refusing overwrite." >&2
else
  "$PYTHON" -B "$RUN_ROOT/src/run_scale_ladder_local_diagnostics.py" \
    --scale 2 --max-parallel 3 --budget-s 7200 \
    --out-root "$LOCAL_DIAGNOSTIC_ROOT" \
    --reference-plan "$PLAN"
fi
```

If the local Python/package/code identity differs from the reviewed Unicorn
plan—or no reference plan is supplied—the local manifest records
`diagnostic_only=true`. Its exact-CG status/journal/iteration/telemetry files
and membership CSV/JSON use the normal ladder schemas.
