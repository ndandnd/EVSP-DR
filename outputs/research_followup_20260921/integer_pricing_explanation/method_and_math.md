# Code-backed method and mathematical scope

## The covering LP and why positive reduced costs can help

Let Aᵢᵣ=1 when route r serves trip i, cᵣ=100,000 plus its conservative expanded-grid charging/start cost, and xᵣ be its route weight. The baseline covering relaxation is

    min cᵀx                  max 1ᵀπ
    Ax ≥ 1, x ≥ 0            Aᵀπ ≤ c, π ≥ 0.

A route's reduced cost is rᵣ=cᵣ−aᵣᵀπ. At a certified final LP dual, every admissible route has rᵣ≥0 up to the pricing tolerance; missing routes with positive reduced cost cannot improve that LP value. This is a certificate for the declared expanded-grid weighted objective. Total fractional route weight and a fleet-only full-model lower bound are different quantities.

For any integer covering witness y, with z_LP=1ᵀπ at dual optimality,

    cᵀy − z_LP = (c−Aᵀπ)ᵀy + πᵀ(Ay−1).

The first term sums reduced costs of selected routes. The second values surplus coverage; it vanishes for an exact partition but must be retained for this covering experiment. Thus an integer solution may rationally select positive-RC columns: it accepts a weighted premium above the fractional LP to obtain a compatible collection of whole routes. Ordinary final-dual negative-RC pricing has no incentive to add those columns. This does not establish that all near-zero enrichment methods fail or that a positive-RC route could not be generated earlier under different duals.

Validated C1 diagnostic: 194 trips, 39,940 fresh columns, 99 positive LP routes all fractional, route weight 7.999999999999991. Weighted LP=800,383.6879776365; eight-route witness=800,479.96. Its eight reduced costs sum to 72.1420916178904; the dual-weighted surplus on duplicated trips 76 and 88 is 24.129930744989927. Therefore **96.27202236349694 = 72.1420916178904 + 24.129930744989927**, with residual 6.17×10⁻¹⁰. The fresh pool proves nine buses; adding the eight witness records proves eight. This diagnostic used a sequential solution as a witness; the subsequent dive replication did not import those routes.

## What is fixed and what pricing solves

The pilot retains a covering master and adds fleet cap Σᵣxᵣ≤K, with K=8 explicitly supplied. Artificial aᵢ≥0 covers missing service at M=500,000 per unit. Each chosen route in F gets LB=UB=1; other route variables have LB=0 and no upper bound. Releasing a fixing restores these original bounds. Trip rows remain in the model; overlap is permitted.

Eliminating fixed variables gives the equivalent residual formulation

    min c_F + Σᵣ∉F cᵣxᵣ + MΣᵢaᵢ
    A_free x + a ≥ 1 − A_F1
    Σᵣ∉F xᵣ ≤ K − |F|,  x,a ≥ 0.

Its dual is

    max c_F + (1−A_F1)ᵀπ + (K−|F|)μ
    aᵣᵀπ + μ ≤ cᵣ for free routes;  0≤πᵢ≤M, μ≤0.

New-column reduced cost is **cᵣ−aᵣᵀπ−μ**. Fixing changes residual coverage and fleet slots, hence optimal duals. A route with positive reduced cost at the original optimum can have negative reduced cost under these conditional duals. That is a mechanism, not a claim that every new column was measured positive under the original dual: the current replication does not report such a complete per-column comparison.

The event-network pricer returns up to 30 source-to-sink routes using `sink_predecessor_route_batch(..., selection_mode="reduced_cost")`. It optimizes the same combined route cost minus trip-dual credit. Since μ contributes the same constant to every route, the adapter subtracts μ afterward without changing route ranking. It verifies path reduced cost against the stored route cost and trip-dual sum. Only true reduced costs below −10⁻⁴ are admitted; a new route incidence or a cheaper realization of an existing incidence updates the master. Every admitted record undergoes physical replay. Route incidences are deduplicated by trip set; all generated records are preserved in the appended journal.

## Candidate selection, retreat and stopping

Only routes whose LP value is not approximately zero or one are fixing candidates. The initial ordering favors largest LP value, then more trips, lower total cost and deterministic trip order. Restarts use longest route first, then lowest cost per served trip first, with deterministic tie breaks. The search stores at most three candidates per level. Retreat replaces the most recent fixing with the next alternative; exhausted levels are released. There is no exhaustive xᵣ=0 complement branch. Columns persist across retreats and restarts.

Limits are 400 pricing iterations per node, a 150 s node threshold checked between operations, 40 total nodes, at most two restarts, a 2,400 s dive wall allowance and a 60 s finishing reserve. These are heuristic operation-boundary checks, not hard real-time cancellation. A node with zero artificials can guide another fixing even if pricing has not closed. Surviving artificials trigger heuristic retreat, not an unpenalized infeasibility proof: the finite artificial penalty has no established sufficient-bound argument. An integral certified node, or a collection of fixed routes covering all trips within K, supplies a feasible cover. `global_certificate` remains null.

Concrete C1 seed20260921 trace: ten nodes; node1 has closed pricing and a fractional solution. At node5 (depth4), 0.250994 artificial weight remains at its time threshold. Node6 retains the first three routes, changes the fourth and has zero artificials. A second alternative occurs at depth5. Node10 (depth7) has an integral solution with closed node pricing. These node events support the operational explanation, not a global proof.

## Final MIP handoff and interpretation

The augmented journal retains the byte-identical original prefix and appends physically replayed generated records. An own-dive integer cover is resolved back to complete records already in that journal, with hashes and ordinals, and exported as a validated MIP start. The independent final MIP replays those routes and asserts zero added/replaced pool columns from the start handoff. All seven successful replications have accepted objective-eight starts in their full logs; all 56 exported records match their augmented-journal ordinals in the prior independent audit. The legacy field name `added_giro_route_count` labels a reused importer and does not identify GIRO as the source.

Dive fixings are not imposed on the final MIP. It searches the full augmented pool, first minimizing bus count, then charging cost under the attained fleet cap. The control searches the unmodified fresh pool with the same two-stage settings and seed. The treatment therefore tests **column generation plus an own-dive incumbent handoff** as a package. It is not a columns-only or warm-start-only ablation.

Four cases C1/C3/C4/C5 × seeds20260921/20260922 give eight paired comparisons: treatment reaches and proves eight within its finite pool in 7/8, control reaches eight in 0/8. Five controls prove finite-pool nine; the remaining three are nine/bound eight. The treatment miss C4/20260921 is nine/bound eight after its dive wall limit; neither infeasibility nor optimum nine is established there.

Control receives 3,600 solver seconds. Treatment receives up to 2,400 dive subprocess wall seconds, including cache load and preparation, then `floor(3600−actual_dive_wall)` solver seconds, with no minimum floor extension. Both final MIPs use eight threads, the paired seed and relative gap10⁻⁴; fleet-stage allowance is half their remaining solver allowance. MIP setup/physical replay is measured external overhead; the verified existing graph is a shared prerequisite, with original construction recorded separately. Nine runs exceed the charged limit by 0.68–4.70 s through solver termination, so this is not a hard end-to-end cap. Successful treatments take 14.14–35.41 minutes actual elapsed; the miss takes 61.57 minutes.

Physics is the historical 240 kWh battery/240 kW charger, zero reserve, flat tariff, start fee5, event model, 2.5 kWh SOC step/5-minute blocks, covering. All selected routes replay physically and every trip is covered at least once. Among successful treatments only C4/20260922 has no duplicate trips; the other six have 1–7 distinct duplicate trips without validated removal. Shared charger capacity is neither imposed nor validated. Claims of a dispatch-ready plan, global integer optimality, general success probability, or exact continuous-cost optimum are outside this evidence.

Code links: [master](/Users/nadan/Documents/projects/demandresponse/.codex-work/integer-columns-20260921/src/diving_pricing_pilot.py:146), [fixings](/Users/nadan/Documents/projects/demandresponse/.codex-work/integer-columns-20260921/src/diving_pricing_pilot.py:280), [pricer](/Users/nadan/Documents/projects/demandresponse/.codex-work/integer-columns-20260921/src/diving_pricing_pilot.py:392), [rankings](/Users/nadan/Documents/projects/demandresponse/.codex-work/integer-columns-20260921/src/diving_pricing_pilot.py:550), [node solve](/Users/nadan/Documents/projects/demandresponse/.codex-work/integer-columns-20260921/src/diving_pricing_pilot.py:602), [search](/Users/nadan/Documents/projects/demandresponse/.codex-work/integer-columns-20260921/src/diving_pricing_pilot.py:750), [incumbent export](/Users/nadan/Documents/projects/demandresponse/.codex-work/integer-columns-20260921/src/diving_pricing_pilot.py:906), [executed wrapper](/Users/nadan/Documents/projects/demandresponse/outputs/research_followup_20260921/integer_pricing_explanation/run_replication.executed.py), [paired proof links](/Users/nadan/Documents/projects/demandresponse/outputs/research_followup_20260921/integer_pricing_explanation/proof_links.md).
