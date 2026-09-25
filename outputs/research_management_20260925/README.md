# Research status and maintenance — 25 September 2026

[Current Google Doc](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.79m3d3x4h45m) · [Cluster audit](cluster/README.md) · [Git maintenance](git_audit/SUMMARY.md)

## Current findings

- At 13:30 EDT, 12 EVSP–DR spatial-price jobs were running (five CG and seven dispatch/charging cleanup); five cleanup jobs had genuine dependencies on running predecessors. Other projects and held historical jobs were excluded. No scheduler recovery or duplicate submission was needed. Recurring monitoring was not restarted.
- All 48 sequential extension cases at targets 33–40 finished. None reached its own target, certified pricing, or closed its finite-pool fleet gap. Target-40 integer counts for C1–C6 are **43, 44, 44, 47, 44, 45**. Individual route replay passes; duplicate-trip cleanup and shared charger capacity remain unvalidated.
- All four C1 cap/stopping variants completed. None of the 15-minute dives found an integer incumbent; all four follow-up 45-minute pool MIPs ended at nine buses with bound eight. Own-incumbent transfer was skipped. Keep these results separate from the earlier four-case/two-seed 7/8 comparison.
- The unchanged 24-pair baseline benchmark remains 24/24 target matches sequential versus 6/24 fresh, with certified event-grid LPs in both methods. Matched charging reoptimizations remain saved-assignment results, with the evening zero-fee comparator favoring GIRO assignments.

## Document organization

The current tab now contains three pages of current results and three figure pages, down from 13 pages. Three editable tables cover the baseline benchmark, target-40 endpoints and charging costs. Existing source links, two original figures in the current tab, and all 22 images in the separate Figures tab are preserved. The new [48-case heatmap](figures/full40_fleet_heatmap.png) has an [editable CSV](figures/full40_fleet_results.csv) and [reproducible builder](figures/build_full40.py).

Earlier prose is preserved in [the pre-edit export](document/current_before.docx) and [text with source links](document/current_before.md). Dated research history, CG curves and other reference tabs remain available in the Doc. Publication verification is recorded in [document/doc_verification.json](document/doc_verification.json).

## Next decisions

1. Finish the existing spatial-price comparisons and validate complete five-bus dispatches. Distinguish fresh CG from fallbacks that add fixed-duty routes.
2. Diagnose why the new C1 pilot did not reproduce the earlier pricing success before extending that treatment broadly.
3. Use the completed large-chain and capacity evidence to choose the next capacity-aware implementation test. These baseline extension misses do not establish full-model infeasibility.

## Repository maintenance

Seven redundant local branch names and 44 invalid worktree records were removed. Twenty-four archive tags preserve historical commits and the five older branch tips. Modern solver lineages remain separate because they contain unmerged experimental changes. Main and its pre-existing tracked edits were preserved. Publication uses the isolated `codex/research-maintenance-20260925` branch; no solver code was merged or deployed.

See the Git receipt for exact names, ancestry checks, before/after counts and recovery commands. Large duplicated document exports and the worktree metadata tar are retained locally rather than added to Git.
