---
name: ladder-lite-decision
description: "2026-08-19 decision — the frozen 138-task scale ladder is one overnight run; the gated launcher is the sole blocker, so we run cells via plain sbatch arrays reading the same approved-plan.json"
metadata: 
  node_type: memory
  type: project
  originSessionId: 10e2f39f-6fa1-4010-a6fc-f6ccbdbb8259
  modified: 2026-08-19T23:48:47.072Z
---

On 2026-08-19 we diagnosed why the Unicorn cluster has produced zero
scale-ladder rows despite a week of work. The 138-task ladder is only ~325
job-hours (~26 h wall at 20-way concurrency). The blocker is entirely
`src/submit_scale_ladder.sub` + `src/launch_scale_ladder.py` (2822 lines):
13 fail-closed preconditions must all pass in one shot (binary sha256 of
python3/scontrol, worker self-hash, per-file hashes over ~22 sources, portable
NumPy-SIMD identity equality, held probes, held activation with a 6-second
observation window, held gate, no-clobber reservations). Any one failure stops
all 138 tasks.

**Decision: "ladder-lite".** Keep `launch_scale_ladder.py --plan-out` (the
reviewed `build_plan()` freezes all 138 cell definitions and works without
`--submit`); replace only the submission layer with plain `sbatch` arrays that
map array index → `plan["task_groups"][GROUP][idx]` → job and run the same
phase commands. Both paths read the same `approved-plan.json`, so rows are
scientifically identical; label provenance `ladder_lite_direct_array`.

Of the 13 guards we keep exactly two, because they are one command each: a
detached checkout at a named commit with a clean tracked tree, and
`EVSP_EXPECTED_COMMIT` (which `run_exact_pool_mip.py` already enforces under
Slurm, along with `EVSP_MIP_EXPECTED_RESULT_SHA256` /
`_JOURNAL_SHA256` / `_INITIAL_PARTITION_SHA256`). Everything else is deleted
from the code path, not disabled.

**Why:** provenance of nothing is worth nothing. Fail open on liveness, fail
closed on interpretation — run the cell, then label it honestly.

**How to apply:** authoritative directive is
`LADDER_LITE_DIRECTIVE_20260819.md` in the repo root (Cursor work order + exact
operator bash). Give Cursor a line budget and an explicit forbidden list in
every cluster work order; the failure mode is unbounded scope, not
incompetence. Report only executed output, never "ready"/"packaged"/"armed".

Related: [[evsp-dr-project-state]], [[work-delegation-token-budget]],
[[giro-data-provenance]]
