# Event-versus-uniform cluster workflow

These scripts execute and document the event-versus-uniform envelope experiment
defined in `analysis/event_uniform_envelope_20260821/plan.json`.

The operator runs only the top-level shell entry points.  Every Slurm worker
uses a detached checkout pinned to the submitting commit; scripts staged by
Slurm never infer the repository from `BASH_SOURCE`.

## Continue after the 2026-08-24 Panel A run

```bash
bash scripts/event_uniform_envelope/continue_after_panel_a.sh \
  "$HOME/ladder-lite/event_uniform_A_20260824_2dd2b4c"
```

This command:

1. writes `panel_a_summary.csv`, `panel_a_stage_counts.csv`,
   `stderr_inventory.csv`, `slurm_accounting.psv`, and checksums under the
   Panel A root;
2. hashes every immutable CG status and journal before submitting corrected
   RAW-pool MIP and target-feasibility arrays;
3. submits Panel B's 45 matched-wall uniform CG cells and 45 exact wall-boundary
   freezes when all nine Panel A event sources are certified.

It does not append to `records/*.csv`, inject known routes, or submit arc-flow
fallbacks.  Panel B integer jobs are deliberately a later step: their frozen
pool hashes must exist before the MIPs are submitted.

## Statistics contract

The normalized cell table is `panel_a_summary.csv`.  Each row records the
representation, CG certificate and stopping reason, LP value, reduced cost,
iterations, columns, wall time, peak RSS, DAG size, fleet-LP certificate,
finite-pool MIP result, target-feasibility result, and Slurm state/exit/RSS.

`panel_a_stage_counts.csv` is the compact stage-level table.  Raw artifacts
remain authoritative; the CSVs are indexes, not replacements.

## Corrected recovery and matched-wall boundary

The first recovery manifest used the Python CSV default CRLF line ending.
Bash therefore retained a carriage return on the final journal-hash field and
all 54 MIP workers correctly refused the mismatch.  In addition, the original
matched-wall freezer conservatively omitted insertions from the last durable
iteration.  Preserve those failed/v1 artifacts and run the reviewed v2 path:

```bash
bash scripts/event_uniform_envelope/recover_and_refreeze_v2.sh \
  "$HOME/ladder-lite/event_uniform_A_20260824_2dd2b4c" \
  "$HOME/ladder-lite/event_uniform_B_20260824_13596d0"
```

This writes an LF-only immutable manifest, reruns missing Panel A MIPs, retries
only the one missing target result with the reviewed event-tariff replay fix,
and creates new Panel B snapshots under `frozen_v2/`.  It pins solver/freezer
code to reviewed Agent E commit `44b6d5030a78ddca9c74f582d70ad87572e61794`
and accepts a later Agent E tip only when that commit remains an ancestor.

Once all 45 `frozen_v2/` snapshots validate, submit their integer comparison:

```bash
bash scripts/event_uniform_envelope/submit_panel_b_v2_integer.sh \
  "$HOME/ladder-lite/event_uniform_B_20260824_13596d0"
```

This runs the plan-specified Gurobi backend on Unicorn: 1,800-second,
eight-thread two-stage RAW-pool MIPs plus independently parallel target-fleet
feasibility solves.  The scientific solver limit is 1,800 seconds; the Slurm
wrapper allows 75 minutes for source validation, physical replay, and durable
serialization.  No routes are injected.
