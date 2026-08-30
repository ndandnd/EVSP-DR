# EVSP-DR research control tower data

This directory is the machine-readable source for `RESEARCH_CONTROL_TOWER.md`
and the companion workbook. It freezes the research-management view as of
2026-08-30; it does not replace immutable solver artifacts.

Files:

- `goal_status.csv`: publication goals, success rules, evidence gaps, and gates.
- `campaign_register.csv`: every current campaign and the command that audits it.
- `current_queue_snapshot.csv`: the latest queue state supplied by the cluster operator.
- `proof_snapshot.csv`: committed and provisional proof counts, with model scope.
- `stage_definitions.csv`: the evidence ladder used to license every claim.
- `next_actions.csv`: ordered, dependency-aware work queue.
- `large_scale_state.csv`: honest historical/current state at k=8 through k=40.

Update protocol:

1. Never infer completion from an empty `squeue`; run the registered audit.
2. Record Slurm accounting and immutable output hashes before changing a status.
3. A combined-cost CG certificate is Stage 1, not an integer-model proof.
4. Keep physics and representation identifiers in every proof statement.
5. Update committed proof counts only from validated, committed artifacts.
6. Run `python scripts/research/validate_research_control_tower.py` before commit.

The latest live snapshot is intentionally reproducible rather than dynamic. A future
snapshot should be appended or replaced only after retaining the old evidence in git.
