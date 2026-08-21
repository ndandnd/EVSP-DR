---
name: evsp-cg-literature-findings
description: "2026-07-31 literature pass on E-VSP column generation (de Vos et al., Desaulniers group) — design implications for our DP pricer"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 5411361d-3fc9-42c9-8275-f6af541497cb
  modified: 2026-07-31T23:31:12.506Z
---

Literature findings for [[evsp-dr-project-state]] (internalized 2026-07-31, before spend-limit reset; no deep-research launched).

## Anchor paper — de Vos, van Lieshout, Dollevoet, "Electric Vehicle Scheduling in Public Transit with Capacitated Charging Stations" (arXiv:2207.13734; Transportation Science, doi 10.1287/trsc.2022.0253). Read in full (methods + results).
Exactly our problem: E-VSP, partial charging, capacitated stations, set-covering master, CG, 816-trip real instance, Python+CPLEX on a MacBook i5.
- **Master** = our set covering PLUS capacity constraints Σ_p u_{r,b,p} x_p ≤ M_r per (station r, 5-min time block b); duals γ_{r,b} subtracted on charging arcs in pricing. (This is what our STATION_COPIES were gesturing at; principled version is one constraint per (r,b).)
- **Pricing = plain shortest path in an acyclic expanded network — NO labeling, NO dominance.** Nodes: trip×SOC (SOC at trip departure) and (station, time-block, SOC-before-charge); source/sink = depot. SOC is discretized INTO the node, conservative rounding down (F^soc), charging starts at next block boundary. Charging is per-block: consecutive blocks chained by charging-node→charging-node arcs, so partial charging = choice of how many blocks. Fixed charge-start penalty on trip→charging arcs (their c_start = €10; ours $5).
- **Numbers**: pricing solves in 0.2–10 s/iter even at 100k+ nodes / 15M arcs; RMP 0.004–0.6 s; CG runs 100s–20,000+ iterations. 816 trips: 63k nodes/7.5M arcs (6% SOC steps), 20,266 iters, 28 h, gap 3.41%, 53 buses = theoretical minimum.
- **Discretization sweet spot**: 5-min blocks + 3% SOC steps (range 22–100%, 27 values). 1% steps: marginally better, ~3× pricing time. 6% steps: ~2× faster, slightly worse. RULE: SOC step must be ≤ SOC gained per time block, else charging rounds to zero and solutions blow up (their 20% step disaster).
- **Integer solutions**: price-and-branch (CG → MIP over pool, which is OUR approach) gave ~20% gaps on ≥119-trip instances. **Truncated CG with fixing ("diving") gave 0.1–1% gaps**: run CG until <0.01% relative improvement over I=30 iters, fix path vars ≥ θ=0.7 (else the single largest), restart CG, repeat until integral. I=15 too small (degeneracy causes flat stretches → premature stop); I=30–90 fine. **Node removal after each fixing round** (drop covered trips' nodes, saturated (r,b) charge nodes, re-preprocess reachability): 512-trip instance 26 h → 7.2 h AND better solutions.
- **True lower bound independent of discretization**: a "dual network" with optimistic rounding (SOC rounded up, charging may start in the block before arrival); Lagrangian-style bound z_RMP + κ·c̄ with κ = fleet-size upper bound.

## Desaulniers group, recent (GERAD list, 2020–2026)
- Gerbaux, Desaulniers, Cappart (2025), C&OR 173:106848: ML-based CG heuristic for MDEVSP (piecewise-linear charging, capacitated stations). **Reduced-size pricing networks** via greedy arc selection + GNN → CG 3.5× faster, −2.2% quality. Open version: PolyPublie 60348.
- Sabatier Montanaro, Jacquet, Cappart, Desaulniers (CPAIOR 2025): same idea, no ML, similar performance ("A Column Generation Heuristic for Multi-depot Electric Bus Scheduling").
- Morabit, Desaulniers, Lodi (2023), INFORMS J. on Optimization 5(2): ML arc selection for SPPRC pricing in CG.
- Costa, Contardo, Desaulniers, Yarkony (2025), IJOC: stabilized CG via dynamic separation of aggregated rows.
- Lam, Desaulniers, Stuckey (2022), C&OR 145:105870: branch-and-cut-and-price for EVRPTW with piecewise-linear recharging + capacitated stations (bounding-based labeling; ATMOS 2024 follow-up).
- Vendé, Desaulniers, Kergosien, Mendoza (2023), TR-C 156:104360: multi-day e-bus assignment + overnight recharge matheuristics.

## Implications for our repo (ranked)
1. **Replace SPPRC labeling with SOC-expanded DAG shortest path** (van Kooten Niekerk 2017 / de Vos style). Our G=300 kWh, 300 kW: 5-min block charges 25 kWh ≈ 8.3% → SOC step ≤ 8%: use ~10 kWh (or 15 kWh) steps, 5-min blocks. ~200–500 trips × ~30 SOC values + station×block×SOC nodes → well within their demonstrated Python scale. Pricing becomes seconds/iter, thousands of CG iterations in 3 h instead of ~20. This also obsoletes the dominance-vs-time-varying-prices worry.
2. **Switch from price-and-branch to truncated CG + fixing + node removal** (their gaps: 20% → <1%). Our run_final_mip price-and-branch shape is literature-documented to plateau exactly like our 3h/12h results.
3. **Add real station capacity constraints** per (station, block) with γ duals (replaces STATION_COPIES; our beta/gamma plumbing in extract_duals already anticipates this).
4. Early-stop rule: relative improvement <0.01% over 30 iters (replaces stagnation windows of 50/999998).
5. LB via optimistic-rounding dual network → report true gaps at 3h/12h milestones (better paper story than objective curves).
6. Optional later: reduced pricing networks (greedy arc selection, Gerbaux-style — we already hand-prune with max_trip2trip=57), dual stabilization (Costa et al.) if RMP duals oscillate.

Local artifact: full de Vos PDF cached at ~/.claude/projects/-Users-nathan-cho-Documents-demandResponse/5411361d-.../tool-results/webfetch-1785540513251-4pyijg.pdf (re-fetch arXiv:2207.13734 if gone).
