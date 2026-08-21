---
name: paper-ideas-small-venue
description: "Candidate contributions for the smaller conference paper (2026-08-02 brainstorm) — sensitivity/flexibility surface, fixed-routing decomposition, demand charges, value-of-solar duals"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5411361d-3fc9-42c9-8275-f6af541497cb
  modified: 2026-08-03T01:53:16.360Z
---

Context: Goal-1 parity with GIRO not yet reached ([[evsp-dr-project-state]]); user wants a genuine research contribution sized for a smaller conference, methodology feeding a bigger planned paper (not yet described to me). See [[project-motivation]].

Candidate contributions discussed 2026-08-02 (ranked by my recommendation):
1. **Savings decomposition — re-timing vs re-routing** (neutralizes the can't-match-GIRO problem): tier (i) hold GIRO's routes fixed, re-optimize only charging times/stations under TOD prices — a small per-route DP/LP, no CG needed, exact; tier (ii) allow re-routing via CG. Report how much of the price-aware savings each tier captures on real Transdev data. Practical message either way.
2. **Price-sensitivity / flexibility surface** (user's idea, sharpened): route costs are affine in each price p_{station,hour}, so master LP value is piecewise-linear concave in p; compute certified stability intervals = pool-restricted LP ranging + pricing at interval endpoints to certify no external column enters (parametric CG). Deliverable: heat map over (station, hour) of minimum price change that alters the schedule + kWh-shifted-per-premium flexibility envelope. Needs fast trustworthy pricing (expanded-network pricer is the enabler — heuristic 5-min pricing makes parametric studies infeasible).
3. **Demand charges**: add monthly 15-min peak-kW cost (real Transdev tariff, Karl Oct 24 email) — max-load variable in master, one extra dual per time block into pricing (beta plumbing exists). CAVEAT: Wu et al. 2022 did bi-objective E-VSP cost+peak-grid-load branch-and-price (epsilon-constraint, 400 trips) — must differentiate (tariff-form demand charge, industrial data, CG-heuristic setting).
4. **Value-of-solar by station**: duals / perturbation of the price-extended master rank stations by marginal value of free midday energy — infrastructure-siting guidance, ties to the bigger microgrid/V2G paper.
5. **Robustness of savings to price-forecast error**: evaluate schedules under perturbed curves; cheap with existing infra.
6. Fallback: **methodology paper on auditing heuristic CG against an industrial reference** (the GIRO column audit toolkit: incidence certificates, antichain lower bounds, wave analysis) — already built, honest, smaller venue.

Notes: (1) and (2) compose into one paper: decomposition establishes the value, sensitivity surface maps where it lives. Ideas 1, 4, 5 do NOT require winning fleet-size parity with GIRO. Lit-check still owed (small targeted search, not deep research): "parametric column generation", "sensitivity analysis branch-and-price", "electric bus charging price sensitivity", Wu et al. 2022.
