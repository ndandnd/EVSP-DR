# EVSP–DR — independent review for Sol6

Independently assess this electric-bus scheduling and charging research project: what the evidence establishes, what remains uncertain, and which next steps would most improve an operations-research paper. Form your conclusions from source code, inputs and raw results. No target conclusion is prescribed.

Start with primary evidence below and write preliminary findings before opening previous reviews. Then compare them with the existing claims and explain any disagreements. Distinguish a supported finding from a hypothesis; cite executed revisions and exact files, fields or log lines. Use primary papers for any novelty assessment.

## Access and code

- Local root: `/Users/nadan/Documents/projects/demandresponse`.
- GitHub: https://github.com/ndandnd/EVSP-DR
- Evidence branch: `codex/week-evidence-20260921`; starting evidence commit `9a770f39d6dbf245860a2bdcf5721255b28c371a`.
- The root checkout is historical. Executed code differs by campaign; use `execution.json`, manifests and result provenance rather than assuming the current root `src/` produced the results.
- Current source locations: `.codex-work/zero-fee-terminal-cg/` (pricing/CG/pool MIP), `.codex-work/integer-columns-20260921/` (integer-directed pricing), `.codex-work/strict-graph-reuse-20260922/`, `.codex-work/capacity-shortcircuit-20260917/`, `.codex-work/charging-factorial-20260921/`, and `.codex-work/dispatch-cleanup-20260918/`. Read their `src/`, scripts and tests as relevant. [Source snapshot](SOURCE_SNAPSHOT.json) records current HEADs, not execution-time cleanliness.
- **Unicorn: ask Nathan for the exact SSH login command before connecting.** Let him complete authentication interactively if required; do not request passwords or private keys in chat. Remote project/results root: `/home/nc437/ladder-lite/`; local manifests map specific output paths. Report lost access promptly. Saved queue snapshots are dated observations, not live status.

## Primary evidence map

All paths below are relative to the local root. Start selectively; do not read every historical output.

| Area | Files / directories |
|---|---|
| Experiment index and provenance | `outputs/research_register/README.md`; follow its campaign manifests and execution records. Treat narrative summaries as claims to verify. |
| Fresh/sequential comparison | `outputs/research_management_20260921/paper_results/`: figure CSVs, `experiment_settings.json`, `provenance.json`, plotting script and linked raw sources. |
| Native Gurobi evidence | `outputs/week_20260921/evidence/`: full logs, result JSON, execution metadata and hash manifests. |
| Subsequent measurements | `outputs/research_followup_20260921/`: chain comparison, integer-pricing evidence, charging/battery work and MIP matrix audit. |
| Paired pricing and later endpoints | `outputs/research_management_20260921/monitor_20260921T235438Z/` and `monitor_20260922T035403Z/`; `outputs/research_management_20260922/monitor_20260922T235958Z/operations/`. |
| Stricter physics and matrix tests | `outputs/week_20260921/capacity_strict/`; `outputs/research_management_20260922/mip_structure/`; corresponding source worktrees above. |
| Larger chains / decomposition | `outputs/week_20260921/chain_extension_40/`; locate earlier decomposition campaigns through the experiment register. |
| Input data and operating documents | `data/`; `outputs/meeting_20260910/giro_email_sources/` contains raw correspondence/attachments. Read relevant originals, not only the adjacent assumption summaries. Keep correspondence private. |
| Recent bound calculations | `outputs/independent_review_20260922_opus55/followup_response/`: `time_bound_check.py`, `time_bounds.json`, `time_certificates.json`; referenced helper/input hashes identify the calculation. Assess them independently. |

## Published claims and comparison material

Read these after forming initial code/data findings:

- Current Doc: https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit
- Current weekly Slides: https://docs.google.com/presentation/d/1F6udjgkiPH51vMUcku7ZT3PhZPAxsQwUCi3TK9vFRDM/edit
- Paper draft figures/claims: `outputs/research_management_20260921/paper_results/RESULTS_PREVIEW.md`.
- Previous reviews and responses: `outputs/independent_review_20260916/` and `outputs/independent_review_20260922_opus55/` (including `FOLLOWUP_AUDIT.md` and `followup_response/README.md`). These are other reviewers' interpretations, not independent evidence.
- A dated current-Doc export is in `outputs/independent_review_20260922_opus55/followup_response/publication/doc_after.md`. If live Docs/Slides or GitHub access fails, say so; a supplied URL is not proof of access.

## Deliverable and review boundary

Write `outputs/independent_review_20260923_sol6/REVIEW.md`: a short assessment, supported findings with evidence/proof scope, unresolved questions, and at most three ranked next experiments or implementation changes. Identify which sources you actually inspected and any missing access. Preserve your preliminary findings before the prior-review comparison.

This assignment is a review, not a takeover of research operations. You may inspect files/Git, conduct scoped read-only cluster checks after obtaining login instructions, and perform small local checks in your own review directory. Do not submit/cancel jobs, modify production code or other agents' work, alter live Docs/Slides, publish, or contact others. Do not inspect credential/license files. Avoid broad downloads or repeated monitoring; leave V2G and held historical jobs untouched. No new monitoring task or review session has been launched by this handoff.
