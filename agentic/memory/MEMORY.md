# Memory index

- [Project motivation](project-motivation.md) — validate CG pipeline vs GIRO's price-blind solution, then demonstrate savings from time/location-varying electricity prices; charge-on-arrival + 57-min gap limit structurally block the load-shifting story
- [EVSP-DR project state](evsp-dr-project-state.md) — repo layout, 2026-07-31 consolidation, branch scheme, next research tasks
- [EVSP CG literature findings](evsp-cg-literature-findings.md) — de Vos et al. SOC-expanded-network pricing + truncated-CG fixing beats our labeling/price-and-branch; Desaulniers 2023-25 acceleration papers; ranked adoption plan
- [Paper ideas, small venue](paper-ideas-small-venue.md) — savings decomposition (re-timing vs re-routing, no GIRO-parity needed), price-sensitivity surface via parametric CG, demand charges (Wu 2022 caveat), value-of-solar duals
- [GIRO data provenance](giro-data-provenance.md) — where assumptions come from (Transdev emails, sanctioned Ref substitution, dropped GIRO rules, real fleet ≈239kWh/220kW) + 2026-07-31 review-driven fixes (LP ub, final-MIP trip set, q audit, dominance recharge count)
- [Price-maker project (chicken-and-egg)](price-maker-project.md) — fleet charging moves prices; equilibrium of schedule↔price loop; authoritative handoff = HANDOFF_PRICE_MAKER_20260813.md in repo
- [Work delegation & token budget](work-delegation-token-budget.md) — Nathan token-capped; heavy coding → Factory/Cursor (Fable 5, $2500), lit research → other LLMs; Claude = architecture/decisions/writing, frugal
- [Ladder-lite decision](ladder-lite-decision.md) — 2026-08-19: the 138-task ladder is one overnight run; the 13-guard gated launcher is the only blocker, so run cells via plain sbatch arrays over the same approved-plan.json
- [Cluster operating rules](cluster-operating-rules.md) — default_partition vs scaglione rule of thumb, mandatory --mem, and the paste/branch/provenance habits that cost time
