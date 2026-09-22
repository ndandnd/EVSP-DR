# 22 September overnight check — 07:55 UTC

## What changed

- **All 25 frozen-pool MIP trials finished.** Fifteen finite-pool optima are proved; ten fresh-k15 searches reached 30 minutes with bound 15. Both k8 pools still require9 within their pools; sequential C1k15 proves 15 in every arm. PreSparsify1 improves fresh C3 from 18 to 17, worsens fresh C1 from 18 to 19, and takes 1,779 seconds on sequential versus 104 seconds default. Keep current production defaults; this one-seed pilot identifies no consistently better setting. [Editable fleet/time tables, 738 checks and full logs](mip_structure/README.md).
- **The reserve/depot-power continuation finished at 65 buses, bound 64.** Prefix k19 is331 trips from 11 subgroup duties. Graph building used 4.53 h inside the four-hour CG allowance, so there were zero pricing iterations. The MIP used an unpriced initial pool: 54 new singleton routes plus 11 inherited routes. Duplicate service and omitted charger-capacity conflicts remain. This is not a full-model lower bound or strict-dispatch result. [290 checks, full logs and exact model scope](operations/README.md).
- **35 EVSP–DR jobs were running at 07:57:42 UTC:**32baseline graph builds plus C1/C3/C4 k33 CGs. Twelve of 44 graphs completed; 93 solver tasks have genuine dependencies. No new k33+ integer endpoint, broken dependency or idle-queue recovery. Ten cumulative preempted graph attempts consumed 28 h 37 m 59 s; all have live replacements. No duplicate jobs were submitted.

## Next useful work

1. Add and validate a durable graph export/load path in an isolated branch before another reserve/depot-power CG. The current runner rebuilds even on `--resume`; only routes survived k19. Packed serialization exists, but no saved k19 graph exists. A one-time rebuild/export, exact source/input/physics/event-lattice identity and fixed-dual/route equivalence checks are required. Explicitly validate old-commit route-checkpoint compatibility and report graph preparation separately. [Bounded implementation and native preflight](operations/graph_reuse_readiness.md). Do not automatically queue k20 or mutate running pins.
2. Continue existing baseline k33–40 jobs and collect genuine CG/MIP endpoints. Treat saved RMP values as uncertified progress until pricing finishes. The separate graph-checkpoint native pilot remains relevant to preemption losses; it is not a repair already applied to these jobs.
3. Prioritize useful integer columns and deployable incumbents over a universal sparsification toggle. The completed parameter comparisons do not resolve either fresh15-bus target. Any follow-up should retain a same-pool control and charge the cost of constructing a new incumbent; the offline saved-start arm cannot establish end-to-end gains.

## Publication and monitoring

The current Doc now has the completed 25-cell editable table, refreshed strict/queue status and the older proposed-test paragraph corrected. Existing figures, historical sections and timing tables are preserved. Current weekly Slides 10 and 39 are updated; slides 40 and 41 add editable MIP and capacity-representation tables. Historical decks are untouched. [Publication verification and before/after exports](publication/README.md).

The user's renewed 22 September AGENTS.md authorizes automatic updates to BOTH the current Doc and the current weekly deck. It supersedes older Doc-only records below older continuation headings. Existing four-hour heartbeat updated in place, quiet on unchanged state; all other persisted fields preserved. [Automation before/after receipt](operations/heartbeat_verification.json). SSH was healthy throughout; held 537227, V2G and CPU exclusion policy preserved.
