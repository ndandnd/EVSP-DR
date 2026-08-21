---
name: work-delegation-token-budget
description: "Nathan is token-limited on Claude — delegate heavy coding to Factory/Cursor (Fable 5, ~$2500 credits), lit research to other LLMs; Claude does architecture/decisions/verification/writing, frugally."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 5411361d-3fc9-42c9-8275-f6af541497cb
  modified: 2026-08-13T20:07:02.582Z
---

As of 2026-08-10 Nathan hit his monthly Claude spending cap and is rationing
Claude tokens.

**Why:** He wants Claude for the highest-leverage work only and has ~$2500
of Factory/Cursor credits where Fable 5 is also available.

**How to apply:**
- Claude's role: architecture, modeling decisions, verification/review,
  honest-claims policing, paper writing. Keep turns short; no re-derivation;
  ask for pasted summaries instead of reading raw logs.
- Heavy implementation → Factory/Cursor agents, via precise specs (goal,
  files, interface contract, acceptance test, "run pytest before push");
  they work on branches like `cursor/recovery-audit` — review the diff stat
  + core file only before merging.
- Literature deep-dives → other LLMs using the pre-written briefs in
  `HANDOFF_PRICE_MAKER_20260813.md` §6; use Claude web research only with
  Nathan's explicit go-ahead.
- Cursor agents' work has been high quality (they found the real
  master-LP defect and correctly withdrew overstated claims) — treat their
  corrections seriously, verify before dismissing.

Related: [[price-maker-project]], [[evsp-dr-project-state]].
