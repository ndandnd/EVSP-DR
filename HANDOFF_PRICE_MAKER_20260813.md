# HANDOFF — the "chicken-and-egg" price-maker project (2026-08-13)

Written by Claude (Fable 5) for Claude, to survive a chat clear. Read together
with: `CURRENT_RESEARCH_PLAN_20260810.md` (the authoritative, corrected state
of EVSP-DR — it supersedes older top-line claims), `HANDOFF_20260810.md`
(operational snapshot + infra/conventions), persistent memory files.

## 0. Operating constraints (read first, they shape everything)

- **Nathan is token-limited on Claude.** Claude's role: architecture,
  modeling decisions, verification, honest-claims policing, writing. Keep
  turns short; never re-derive what the repo docs record; ask for pasted
  summaries, not raw logs.
- **Heavy coding goes to Factory/Cursor** — Nathan has ~$2500 of credits
  there and Fable 5 is available on those platforms. Delegate implementation
  by writing precise specs (file, function, contract, test) referencing repo
  conventions (`HANDOFF_20260810.md` §1, §5). Cursor agents already work on
  branches like `cursor/recovery-audit`; review their diffs before merging.
- **Deep literature research**: only with Nathan's explicit go-ahead on
  Claude tokens; otherwise write delegated research briefs for other LLMs
  (queries are pre-written in §6).

## 1. The idea

EVSP-DR treats electricity prices as exogenous: the fleet is a price-TAKER.
But a 40-bus fleet with 240–300 kWh packs charging at 220–300 kW is a
multi-MW controllable load. If the scheduler herds charging into the "cheap"
hours, that aggregate load moves the price — the cheap hour stops being
cheap. Chicken and egg: **the schedule depends on prices; prices depend on
the schedule.** The thesis: fleet scheduling as a price-MAKER — the
interaction between an EVSP-DR-class scheduler and a price-formation
mechanism, its equilibria, algorithms to compute them, and the welfare/cost
consequences of ignoring the feedback ("self-defeating cheap-hour herding").

Two solution families (Nathan's seeds, expanded):

**A. Iteration / fixed point (build first).** Sequence:
p⁰ → schedule S¹ = EVSP(p⁰) → hourly load L¹ = load(S¹) → p¹ = Π(L¹) → ...
where Π is a price-response model of the grid. Nathan's "benevolent
dictator" framing = one agent owns both grid cost and fleet cost. Key
structure to exploit:
- If Π is an increasing (affine) per-hour supply curve, best-response
  iteration is exactly the classic *charging-equilibrium* setting (Ma,
  Callaway & Hiskens valley-filling; Gan, Topcu & Low decentralized EV
  charging). Naive iteration is known to **oscillate** (two cheap hours
  flip-flop); damping/averaging restores convergence. Reproducing that
  oscillation with real bus schedules is itself a compelling figure.
- The dictator problem with Π = marginal cost of a convex generation-cost
  curve C(L) is a SINGLE convex-in-load co-optimization: min fleet-ops cost
  + C(load). The fixed point of the damped iteration should coincide with
  its optimum (potential-function argument). Three regimes to compare:
  (i) price-taker equilibrium (myopic loop limit), (ii) dictator optimum,
  (iii) Stackelberg (grid designs the tariff anticipating fleet response —
  bilevel/MPEC, thesis-scale, not the first paper).
- **Why our pipeline is unusually well suited:** column pools are
  price-INDEPENDENT in feasibility; only column costs change with p. So
  iteration k+1 can re-cost the existing journal, re-solve the master, and
  price only a few new columns — "price-parametric column generation."
  And `rerealize_routes.py` (fixed trip sequences → optimal charging under
  a tariff, seconds per fleet) is a cheap approximate best-response oracle:
  charging-only response for inner-loop iterations, full CG occasionally
  for exact response. Two-fidelity fixed-point iteration is a genuine
  methodological contribution, not just glue.

**B. Learning (second wave).** Options, roughly in order of thesis value:
surrogate response function p ↦ hourly-load-profile trained on Track-A runs
(amortized best response → fast equilibrium search, generalization across
instances); learned price-impact model Π from market data; RL for the
Stackelberg tariff designer. Track A generates the training data for free.

## 2. How EVSP-DR feeds this (and what to trust)

- The **response function** of the fleet to prices is exactly what the
  EVSP-DR final experiment measures (Observed / Fixed-sequence / Joint per
  tariff — see `CURRENT_RESEARCH_PLAN_20260810.md` "Intended final
  experiment"). That experiment's outputs = the first characterization of
  EVSP(p) on real data.
- **Honesty inherited (do not re-assert withdrawn claims):** repricing GIRO
  under a peak tariff measures *exposure*, not savings; the old "0.07%
  re-timing is nearly free" comparison is withdrawn (different tariffs on
  the two sides); 39-bus results are optimal over finite augmented pools,
  not globally; k=40 unions are constructed, not a verified service day;
  all charging results assume **no charger-capacity limit** and are
  optimistic lower bounds.
- The no-capacity caveat is not a footnote here — **herding is a congestion
  phenomenon**. Charger concurrency and price impact are two faces of the
  same coupling. The planned post-hoc concurrency audit is directly
  reusable; a soft concurrency price in Π can stand in for hard capacity.
- Terminal-energy policy and the $5 charge-start term (open decisions in
  the corrected plan) must be fixed once and shared by both projects.

## 3. Publication strategy (Nathan's two goals)

**Goal 1 — a quick conference paper.** Recommendation: finish the EVSP-DR
decomposition paper *framed as measuring the fleet's demand-response
function* — the exact input the equilibrium work needs. Its experiment is
already fully specified in the corrected plan; the framing costs nothing
and makes the follow-on paper's motivation section write itself. Venues to
shortlist (delegate a CFP/deadline scan): ACM e-Energy, IEEE
SmartGridComm/PES-GM, ITSC, TRB, CASE; also fast OR outlets (Operations
Research Letters) if a compact certified-CG story is preferred.
Fallback if Transdev data questions (weekday variants) stall it: a compact
Track-A paper — "price-making fleet charging: oscillation and damped
convergence of schedule–price iteration on real bus operations" — is
publishable at e-Energy/SmartGridComm scale with only the
charging-response oracle (no joint rerouting needed).

**Goal 2 — methods that actually help.** The pieces of paper 1 that carry:
the response-function tables (training/validation data for B), the
re-realization oracle (inner loop), price-parametric CG (outer loop), the
concurrency audit (congestion model), honest-metrics reporting
(electricity, kWh, peak-window kWh, deadhead, visits, SOC endpoints, fleet,
concurrency — reported separately).

## 4. TODO (two tracks; A = quick paper, B = chicken-egg)

Track A — finish EVSP-DR per the corrected plan (owner: Factory/Cursor
agents; Claude reviews):
- A1. Execute the six-step engineering campaign in
  `CURRENT_RESEARCH_PLAN_20260810.md` (raw-residual gate on terminal pools,
  last-valid-LP preservation, fleet-vs-time MIP curves, 72 h no-stall
  controls, hashed archives, actual-walltime analysis).
- A2. Resolve the three modeling decisions: terminal-energy policy,
  $5 charge-start treatment, station-specific vs synthetic-uniform charger
  power. (Claude + Nathan decide; agents implement.)
- A3. Run the final experiment (Observed / Fixed / Joint × tariffs × four
  40-duty variant combos), with the mandatory concurrency audit.
- A4. Email Karl/Transdev: weekday-variant mapping, charger counts & rates.
- A5. Freeze, write, submit (venue from the CFP scan).

Track B — chicken-and-egg:
- B1. One-page formal model note (Claude): agents, timing, Π families
  (affine supply curve; calibrated Nordpool SE3 curve on top of the
  existing hourly SEK data), objectives for regimes (i)–(iii), the
  potential-function convergence claim to prove or cite.
- B2. Delegated lit brief (other LLMs; queries in §6) → annotated
  bibliography with the 10 closest papers and our delta.
- B3. Iteration harness v0 (Factory): loop over
  `rerealize_routes.py`-as-oracle, hourly load aggregation, Π update,
  damping parameter, convergence/oscillation logging. Pure local python;
  no cluster needed. Partille k=8/13 instances first.
- B4. Experiments: undamped oscillation figure; damped convergence;
  price-taker limit vs dictator optimum gap; sensitivity to fleet size
  (k=8→40 — "when does the fleet become a price maker?").
- B5. Price-parametric CG (Factory, bigger): journal re-costing + master
  re-solve + incremental pricing under a new tariff inside
  `exact_pricer_expanded.py`; enables exact best response in the loop.
- B6. Paper-1B skeleton once B3/B4 produce figures.
- B7. Thesis-scale (later): Stackelberg/bilevel tariff design; ML surrogate
  (B family); hard charger capacity in pricing network.

Sequencing: A1–A3 and B1–B3 are parallel (different owners). B4 needs B3
only. The quick paper is whichever of A5 / B6 matures first — decide at
that fork with Nathan.

## 5. Delegation protocol

For every Factory/Cursor task, hand the agent: (1) this file's §0/§5,
(2) `HANDOFF_20260810.md` §1 (environments) and §5 (conventions — tests
before push, no results/ commits, honest labeling, Scaglione-for-MIP,
idempotent .sub patterns), (3) a spec of the form: goal, files to touch,
interface contract, acceptance test, and "run `cd tests && python3 -m
pytest -q` before push". Review diffs on their branch before merging to
`peel-and-price`. Claude-token budget per review: read the diff stat and
the core file only.

## 6. Pre-written research briefs (paste to any capable LLM)

1. "Survey decentralized/aggregate EV charging control where charging load
   affects the electricity price: Ma–Callaway–Hiskens valley-filling,
   Gan–Topcu–Low, mean-field charging games, tâtonnement stability and
   damping/averaging fixes. For each: price model, convergence result,
   gap to a setting where the load comes from a vehicle-SCHEDULING problem
   (routes + charging), not just charging profiles."
2. "Literature on price-making storage/flexible load bidding in electricity
   markets (self-scheduling vs price-impact-aware, MPEC/EPEC bilevel tariff
   design). What price-impact models Π(L) are standard, and what data
   calibrates them for Nordic bidding zone SE3?"
3. "Electric vehicle scheduling problem (E-VSP) with time-of-use or dynamic
   prices: anything where fleet load feeds back into price? Anything using
   column generation parametrically across price scenarios?"
4. "Convergence of best-response dynamics in congestion/potential games
   with a continuum vs finite large agents — conditions under which damped
   iteration between a scheduler and an affine price update converges;
   counterexamples that cycle."

## 7. State of EVSP-DR at handoff (one paragraph, details in repo docs)

Exact SOC×time CG pricer with certificates on discretized route spaces;
re-realization utility making injected schedules physics-valid by
construction; 39-bus pool-optimal schedules on constructed k=40 unions vs
GIRO's 40 duties; corrected-claims regime in force
(`CURRENT_RESEARCH_PLAN_20260810.md`); cursor/recovery-audit branch added
durable IO, legacy-pool migration, snapshot-controlled CG, raw-pool audits,
and found the real large-master defect (primal values zeroed before the
feasibility check — my earlier tolerance diagnosis was only proximate).
Overnight campaigns for the corrected plan are scripted
(`prepare/launch/collect_overnight_correctness`, `submit_*` files) and are
the current cluster workload, alongside the final-experiment prerequisites.
