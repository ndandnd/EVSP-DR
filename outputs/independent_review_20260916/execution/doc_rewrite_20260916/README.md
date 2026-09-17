# Current-research replacement — 16 September 2026

Replaced only the current-research tab using the user-supplied DOC_REWRITE.md. Four native editable tables. Figure, CG-curve and history tabs have identical before/after Markdown hashes; all six source/tab footer links remain. Slides untouched.

**doc_verification: PASSED.** [Before](before.md), [after](after.md), [checks and hashes](doc_verification.json), [resolved source](resolved.md), [change record](changes.json).

Counts reproduced from the audited chain table, its 26 already-audited longer searches and 128 replayed schedules: 102 cases, 93 with numerical bound k, 9 with bound k−1; 67 match the bound, 35 remain open. Matching GIRO's count is a separate statistic: 70/102.

Expected dates use ledger-referenced job receipts and live Slurm start times. Times are EDT. Running jobs show start plus allocated wall time; unstarted sequential dependencies use a full-budget projection with zero queue delay. Held full-Partille CG/MIP have no scheduled start. [Per-job calculation](expected_by_job.csv), [scheduler evidence](slurm_timing.txt), [calculation script](calculate_expected.py).

The 102-case cohort is frozen. New seeded experiments do not alter that denominator. Source DOC_REWRITE.md is preserved; corrections are recorded separately. The current HTML and dashboard builder now use this replacement.
