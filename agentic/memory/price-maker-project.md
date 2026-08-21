---
name: price-maker-project
description: "The new \"chicken-and-egg\" thesis project — fleet charging moves electricity prices; equilibrium between EVSP scheduling and price formation. Full handoff in repo."
metadata: 
  node_type: memory
  type: project
  originSessionId: 5411361d-3fc9-42c9-8275-f6af541497cb
  modified: 2026-08-13T20:06:55.505Z
---

**The project (started 2026-08-13):** EVSP-DR is price-taking; a 40-bus fleet
charging at 220–300 kW is a multi-MW load that moves the price it optimizes
against. New thesis: price-MAKER fleet scheduling — equilibria of the
schedule↔price loop, algorithms (damped fixed-point iteration first, ML
surrogates second), and the cost of ignoring the feedback (self-defeating
cheap-hour herding). Three regimes: price-taker equilibrium, benevolent-
dictator co-optimum (convex potential when price impact is increasing
affine), Stackelberg tariff design (thesis-scale, bilevel/MPEC).

**The authoritative handoff is `HANDOFF_PRICE_MAKER_20260813.md` in the
EVSP-DR repo root** (with `CURRENT_RESEARCH_PLAN_20260810.md` for corrected
EVSP-DR claims and `HANDOFF_20260810.md` for infra). It contains the full
TODO (Track A = finish EVSP-DR paper framed as measuring the fleet's
demand-response function; Track B = iteration harness on the re-realization
oracle), pre-written lit-research briefs for delegation, and the delegation
protocol.

**Key technical levers:** (1) `rerealize_routes.py` = seconds-fast
charging-only best-response oracle for the inner loop; (2) column pools are
price-independent in feasibility → "price-parametric CG" (re-cost journal,
re-solve master, incrementally price) = exact best response cheaply across
price iterations; (3) charger concurrency and price impact are the same
congestion coupling — the planned concurrency audit feeds Π.

**Do not re-assert withdrawn EVSP-DR claims** (see
[[evsp-dr-project-state]]): peak-tariff repricing = exposure not savings;
the 0.07% re-timing claim is withdrawn; 39-bus = pool-optimal not global;
k=40 unions are constructed days; costs are no-charger-capacity lower
bounds.

Related: [[project-motivation]], [[paper-ideas-small-venue]],
[[work-delegation-token-budget]].
