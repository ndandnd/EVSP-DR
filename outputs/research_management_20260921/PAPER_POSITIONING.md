# What would make the results a paper contribution

21 September 2026. This is a focused check of primary-source abstracts and publication records, not a completed literature review or novelty claim.

## What is already established in the literature

- De Vos, van Lieshout and Dollevoet, *Electric Vehicle Scheduling in Public Transit with Capacitated Charging Stations*, Transportation Science 58(2), 279–294. Their model combines partial charging and limited station capacity, uses a path formulation and CG, and compares price-and-branch with diving. The abstract reports instances up to816trips and a diving solution within7hours and3%gap. These are their settings and claims, not a directly comparable benchmark for our data. [Publisher record and abstract](https://pubsonline.informs.org/doi/10.1287/trsc.2022.0253).
- Parmentier, Martinelli and Vidal, *Electric Vehicle Fleets: Scalable Route and Recharge Scheduling through Column Generation*. Their approach combines charging-arc reformulation, bidirectional pricing, graph sparsification and diving. Generic graph acceleration or diving is therefore not a novelty claim by itself. [Authors’ preprint](https://arxiv.org/abs/2104.03823).
- Maher and Rönnberg, *Integer programming column generation*. Their work directly addresses the mismatch between LP-oriented pricing and columns useful for integer solutions. Our observed k8 phenomenon is evidence about our formulation and pools, not discovery of this general principle. [Authors’ paper record](https://optimization-online.org/2022/03/8816/).

## Candidate results and what is still needed

| Candidate result | Evidence available | Next comparison needed |
|---|---|---|
| Integer-useful route generation | Four k8 pools proved9; witnessed8outside; original directed-pricing pilot3/4hits. | Balanced seeds and matched budgets, then larger inputs and a clearly specified standard diving/enrichment comparator. |
| A tractable representation for realistic charging | Fixed-dual packed benchmark and a larger strict endpoint. | Medium/large equivalence, full CG accounting, then shared-capacity pricing. Do not transfer no-capacity speedups automatically. |
| Value of changing duty assignments under tariffs | Saved-sequence recharging shows savings versus original charging. | Original repricing, flexible fixed-duty optimization and fresh joint CG with identical constraints, ending energy and validated dispatches. A fixed-path fee factorial answers only the fee question. |
| Practical scalability and reproducibility | Six sequential chains, fresh controls, saved journals, proofs and failed attempts. | Separate accumulated inheritance cost, graph cost and integer-stage work; distinguish end-to-end performance from cached-input performance. |

The strongest near-term computational narrative is: establish LP and integer behavior separately, demonstrate the missing-column mechanism, test a reproducible repair, then measure its cost and scope under more realistic physics. Whether this is sufficient for the intended journal still depends on the methodological difference from these existing approaches and on the economic result. The five draft figures make that evidence visible; they do not settle publication readiness.

Next literature task: read the complete methods and computational sections of these three works and construct a formulation/algorithm/constraint comparison with exact definitions before claiming novelty or choosing a state-of-the-art benchmark. Do not compare their reported runtimes to ours as if hardware, inputs and feasibility models matched.
