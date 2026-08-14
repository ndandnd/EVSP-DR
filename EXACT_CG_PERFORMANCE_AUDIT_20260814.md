# Exact-CG performance audit

This work is diagnostic only.  It does not change master/pricing semantics,
defaults, stopping rules, resume identity, or live cluster jobs.

## Opt-in durable phase telemetry

Add one explicit sidecar path to an otherwise unchanged exact-CG command:

```bash
python -u src/exact_pricer_expanded.py \
  ...existing arguments... \
  --phase-telemetry /absolute/path/to/run.phases.jsonl
```

Default is off.  The telemetry path is excluded from model/resume provenance.
The append-only sidecar records network build, incidence construction, every
master attempt, exact shortest-path pricing, extra-column extraction, route
insertion, journal/iteration fsyncs, status checkpoints, snapshots, pool size,
incidence nonzeros, network nodes/arcs, and process peak RSS.  Every JSONL row
is fsynced.  On reopen, only an interrupted final row is repairable and the
session identity must match.  Existing session identity is checked read-only
before any tail repair.  Telemetry I/O overhead is excluded from CG
wall/snapshot/stall clocks; if later telemetry I/O fails, telemetry disables
itself with a warning rather than changing solver fallback or stop reasons.

**Operational limit:** current per-phase telemetry fsyncs every row and is
approved only for short, bounded diagnostic runs.  Do not enable it on the
22--24 hour CA/CS/PA/PS campaigns; thousands of synchronous shared-filesystem
writes can consume substantial real Slurm allocation time even though that
overhead is excluded from CG's logical stopping clocks.  Aggregation is a
future change, not part of this correction pass.

## Read-only frozen-pool prefix profiler

```bash
python -u src/profile_exact_pool_prefixes.py \
  --result /absolute/path/to/immutable.snapshot.json \
  --expected-result-sha256 <64-hex-status-hash> \
  --expected-journal-sha256 <64-hex-journal-hash> \
  --prefixes 1000,5000,10000,25000,50000 \
  --methods highs,highs-ds,highs-ipm \
  --repeat 3 \
  --out /separate/path/prefix_profile.json
```

The profiler reconstructs first-reached unique-incidence prefixes, builds the
incidence matrix once per prefix, and cold-solves each requested SciPy/HiGHS
method.  It records incidence time/size/nonzeros, total and backend solve time,
LP values/violations, peak RSS, and errors.  Status, journal, instance, and
tariff hashes are checked before and after; source files are never repaired or
opened for writing.

### Guarded Unicorn staging

Use a reviewed detached checkout and a compute allocation.  Never switch the
checkout that owns a running campaign.

```bash
set -euo pipefail

: "${EXPECTED_REVIEWED_COMMIT:?export independently reviewed 40-char commit}"
: "${EXPECTED_STATUS_SHA:?export status SHA from campaign manifest/archive}"
: "${EXPECTED_JOURNAL_SHA:?export journal SHA from campaign manifest/archive}"
[[ "$EXPECTED_REVIEWED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$EXPECTED_STATUS_SHA" =~ ^[0-9a-f]{64}$ ]]
[[ "$EXPECTED_JOURNAL_SHA" =~ ^[0-9a-f]{64}$ ]]
test -n "${SLURM_JOB_ID:-}"

PROFILE_ROOT="$HOME/EVSP-DR-exact-profile-${EXPECTED_REVIEWED_COMMIT:0:12}"
SOURCE_ROOT="$HOME/EVSP-DR-legacy-recovery-bab7bfe"
SNAP="/absolute/path/to/immutable.snapshot.json"

test "$(git -C "$PROFILE_ROOT" rev-parse HEAD)" = \
  "$EXPECTED_REVIEWED_COMMIT"
test -z "$(git -C "$PROFILE_ROOT" branch --show-current)"
test -z "$(git -C "$PROFILE_ROOT" status --porcelain --untracked-files=no)"
test "$(sha256sum "$SNAP" | awk '{print $1}')" = "$EXPECTED_STATUS_SHA"

readarray -t INPUTS < <(python3 - "$SNAP" "$PROFILE_ROOT/src" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from run_exact_pool_mip import resolve_pool_journal
source = Path(sys.argv[1])
status = json.load(open(sys.argv[1]))
print(status["csv"])
print(status["prices_csv"])
print(status["provenance"]["instance_sha256"])
print(status["provenance"]["prices_sha256"])
print(resolve_pool_journal(source, status).resolve())
PY
)
CSV=${INPUTS[0]}
PRICES=${INPUTS[1]}
EXPECTED_CSV_SHA=${INPUTS[2]}
EXPECTED_PRICES_SHA=${INPUTS[3]}
JOURNAL=${INPUTS[4]}

test "$(sha256sum "$JOURNAL" | awk '{print $1}')" = "$EXPECTED_JOURNAL_SHA"
[[ "$CSV" != /* && "$CSV" != *".."* ]]
[[ "$PRICES" != /* && "$PRICES" != *".."* ]]

mkdir -p "$PROFILE_ROOT/data/$(dirname "$CSV")"
rsync -a --ignore-existing \
  "$SOURCE_ROOT/data/$CSV" "$PROFILE_ROOT/data/$CSV"
rsync -a --ignore-existing \
  "$SOURCE_ROOT/data/$PRICES" "$PROFILE_ROOT/data/$PRICES"

test "$(sha256sum "$PROFILE_ROOT/data/$CSV" | awk '{print $1}')" = \
  "$EXPECTED_CSV_SHA"
test "$(sha256sum "$PROFILE_ROOT/data/$PRICES" | awk '{print $1}')" = \
  "$EXPECTED_PRICES_SHA"

OUT="$PROFILE_ROOT/src/results/profiles/$(basename "$SNAP").prefix-profile.json"
test ! -e "$OUT"

cd "$PROFILE_ROOT"
python -u src/profile_exact_pool_prefixes.py \
  --result "$SNAP" \
  --expected-result-sha256 "$EXPECTED_STATUS_SHA" \
  --expected-journal-sha256 "$EXPECTED_JOURNAL_SHA" \
  --prefixes 1000,5000,10000,25000,50000 \
  --methods highs,highs-ds,highs-ipm \
  --repeat 3 \
  --out "$OUT"
```

The profiler itself rechecks the copied instance/tariff against status
provenance, reconstructs the complete unique-incidence pool, validates
`status["columns"]` and every positive `final_lp` route, refuses an existing or
concurrently owned output, and rehashes all sources after profiling.
`EXPECTED_STATUS_SHA` and `EXPECTED_JOURNAL_SHA` must come from an independent
campaign manifest/release attestation; never derive the expected journal hash
from the same resolver/path selection being tested.

## Deterministic pricing microbenchmark

```bash
python -u src/benchmark_exact_pricing.py \
  --csv Practice_Custom_TwoDuty_13301_13302.csv \
  --prices_csv hourly_prices_flat.csv \
  --soc-step 15 --block-min 10 --columns 30 \
  --warmup 1 --repeat 3 \
  --out /separate/path/pricing_benchmark.json
```

The benchmark builds one network, prices against a fixed dual vector, and
hashes all returned route incidences, ordering, reduced costs, route nodes, and
charging realizations.  Commit B's micro-optimizations are acceptable only if
the route hash and best route remain identical case-by-case.

## Interpretation

Telemetry identifies where live CG iterations spend time.  Prefix profiling
isolates cold incidence/RMP scaling.  The pricing benchmark measures only the
expanded-network pass and existing candidate extraction.  None is a pricing
certificate or an integer-schedule result.

## Commit-B local benchmark result

`analysis/exact_cg_performance_20260814/benchmark_comparison.json` contains
the exact before/after repetitions and route hashes.

| Case | Before median | After median | Reduction | Route hash |
|---|---:|---:|---:|---|
| Two-duty pair (86 trips) | 0.05549 s | 0.04353 s | 21.6% | unchanged |
| Reproducible synthetic k8 (206 trips) | 0.11293 s | 0.08833 s | 21.8% | unchanged |

These are local microbenchmarks, not cluster throughput claims.  The synthetic
k8 input is generated with seed `20260802`, size 8, replicate 1 and is not
committed as a dataset.
