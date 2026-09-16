# EVSP–DR independent review handoff

Prepared 16 September 2026. This is a source map, not an assessment of the results. Review read-only initially; do not submit/cancel jobs or modify the research documents without a separate request.

## Review request

Independently assess what the experiments establish, what remains uncertain, and which next experiments would be most informative. Trace important conclusions to inputs, execution code, solver logs and selected schedules rather than accepting previous summaries.

The project studies electric-bus scheduling with charging decisions and time/location-dependent electricity prices, using GIRO/Transdev data. `k` is the number of reference GIRO bus duties whose trips form an instance; it is not the number of trips.

Questions:
1. Can column generation followed by an integer solve recover the GIRO fleet target? How does this depend on instance selection and size?
2. How does sequentially adding a duty and retaining earlier columns compare with a fresh start given the accumulated computation budget?
3. Which algorithm changes have controlled evidence of benefit?
4. With no charging-start fee, does changing duties improve on optimizing charging while holding GIRO duties fixed?
5. Are the physical assumptions, feasibility checks, proof claims and comparisons justified? What evidence is missing for a publishable study?

Please return your own findings with precise source references, distinguish verified facts from hypotheses, and prioritize any required corrections or experiments. No need to agree with the existing interpretations.

## Access and versions

- Local workspace: `/Users/nadan/Documents/projects/demandresponse`.
- GitHub: https://github.com/ndandnd/EVSP-DR (access may require the user's GitHub account).
- Current published research artifacts: branch `codex/parallel-research-20260911`, checkpoint `cecc5365c17b0db8434d7d59065a4bf0ab3b1d01`.
- Local publication checkout: `.codex-work/research-register-20260915`, branch `codex/research-register-20260915`. The main workspace is on `main`, has uncommitted work, and is not the execution version for every experiment. Preserve it.
- Execution code is pinned separately in each experiment manifest. For example, the k31–32 campaign records CG `a0e0bb7681c8451e3cbbbfa06aef390026d9af4b` and MIP `871d057e1067411f09581e37d78f7c1ca43f68bb`. Use recorded execution commits, not whichever branch is currently checked out.
- Zero-fee development checkout: `.codex-work/zero-fee-terminal-cg`, branch `codex/zero-fee-terminal-cg`; its current HEAD is `2c7445ac6449eb7d338c83f8320574da042a65b5`. Different stages have their own recorded commits.

All paths below are relative to the local workspace unless absolute.

## Minimum source set

| Purpose | Location |
|---|---|
| Latest collected source records for this handoff | `outputs/post_meeting_20260910/monitor/20260916T194843Z.json` — collection began 19:48 UTC, finished 19:55 UTC |
| Machine-readable experiment index | `outputs/research_register/register.json` or `register.csv`; use rows to locate manifests, inputs, code and outputs |
| Original chain results and CG endpoints | `outputs/overnight_next_20260914/status_20260916T194843Z/all_chain_extension_results.csv` |
| Separate longer MIP results | Same directory: `longer_gap_results.csv` and `longer_gap_validation.json` |
| Fresh versus accumulated-time experiments | `outputs/cumulative_budget_20260913/manifest.json` and `status_20260916T194843Z/` |
| Algorithm comparisons | `outputs/controlled_comparison_20260913/` — start with its manifest/design and paired result files |
| Zero-fee charging experiment | `outputs/zero_fee_full_cg_20260916/manifest.json`; postprocessing: `outputs/terminal_exact_once_20260916/` and `outputs/terminal_duplicate_cleanup_20260916/`; consolidated data and source hashes: `outputs/zero_fee_validated_comparison_20260916/comparison.json` |
| Original GIRO material | `outputs/meeting_20260910/giro_email_sources/` and `data/` (including `GIRO_soln.docx`, vehicle and deadhead spreadsheets); `outputs/model_fairness_audit_20260913/giro_requirements_audit.md` indexes relevant requirements/sources |

For additional topics: `outputs/strict_capacity_parallel_20260914/`, `outputs/capacity_pricing_boundary_20260914/`, `outputs/decomposition_union_audit_20260914/`, and `outputs/decomposition_lp_support_union_20260914/` contain relevant campaign records. Git history contains the earlier implementations and experiments.

Some READMEs contain accumulated dated updates and stale introductory links. Use timestamps and manifest/source hashes to establish chronology. Summary tables and prior audit narratives are claims to check, not substitutes for primary artifacts. The register contains artifact/stage records, not one independent experiment per row.

## Unicorn: original logs and large artifacts

Login: `ssh nc437@unicorn-login-01.coecis.cornell.edu`.

On the user's Mac, an existing authenticated connection may be available:

```sh
ssh -S /Users/nadan/.ssh/evsp-unicorn.sock -o BatchMode=yes -o ConnectTimeout=8 nc437@unicorn-login-01.coecis.cornell.edu
```

- Campaign roots: `/home/nc437/ladder-lite/<campaign_name>/`.
- Large campaign outputs may be symlinked to `/share/scaglione/nc437/evsp-dr/<campaign_name>/`; resolve symlinks before interpreting missing files.
- Published-artifact mirror: `/home/nc437/ladder-lite/research-register/`.
- Start with each campaign's `manifest.json`, `jobs.json`/`case_jobs.json`, and `cases/`. Manifests and collected records give the exact input, column-journal, selected-route, result and log paths. MIP attempts commonly use `cases/<case>/mip/<job>_r<restart>/`; preserve distinct attempts.
- Relevant chain campaigns include `chain_extension_20260913`, `chain_extension_31_32_20260915`, and `continuation_gaps*`. `continuation_gaps15_20260916` was submitted after the collection above; inspect it separately for newer results.
- Slurm binaries: `/usr/local/slurm/slurm-25.05.5/bin/`. Query only user `nc437` and explicit dates when using accounting; job IDs can recur.
- Resource policy: `/home/nc437/ladder-lite/SCAGLIONE_RESOURCE_POLICY.md`. Leave held jobs and the separate EVSPV2G experiments untouched. Notify the user if cluster access fails.

## Current presentation, to read after the evidence

[Google Doc: current results](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.79m3d3x4h45m). Other tabs retain historical notes and figures. A local export is `outputs/overnight_next_20260914/status_20260916T194843Z/doc_after.md`. Do not edit Slides.

When reporting, distinguish scheduler completion, CG stopping reason/pricing certificate, restricted-pool integer proof, physical schedule validation and reference-target attainment. Check the actual objective, covering/partitioning sense, initialization, charging power/capacity, starting/ending SOC, duplicate-trip handling and runtime accounting for each comparison rather than assuming they are constant across campaigns.
