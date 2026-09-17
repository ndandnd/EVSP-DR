# Geography review: independent check, 17 September 2026

The reported partition minima reproduce. The strongest new result is a four-way partition constructed without GIRO duty labels whose time-only minima are 8+8+8+8=32. Its battery-feasible schedules remain untested. The tested geographic and spectral cuts are less promising; this is not an impossibility theorem for geographic decomposition.

`python3 verify.py` recomputes all 16 supplied partitions using the previously audited exact closure and an independent matching implementation. Equal-size matching/vertex-cover certificates and paths are saved in `certificates.json`; input hashes, comparison results and register counts are in `verification.json`. No solver or cluster jobs were launched. The supplied scripts and figure were inspected; the original reviewer files are unchanged.

| Claim | Check |
|---|---|
| Parent time-only optimum 32; ten duty partitions also 32 | Verified |
| Label-free time-only-cover partition has component minima 8/8/8/8 | Verified, including validity of its saved 32 paths |
| Vehicle group 32; nearest charger 34; Fiedler 40; AM/PM 64; spectral four-way 68 | Verified |
| All 39 component CG runs certified at 8 | Refuted by current register: 33 certified, six wall-limited. d04_g0 route weight is 8.00201922535692; the other 38 are numerically eight |
| Twelve component pools exclude eight buses | Verified as recorded finite-pool bounds; eleven proved nine, one incumbent ten with bound nine |
| Every balanced four-way cut severs about 70% of arcs | Not established: only selected partitions were tested. The four-way spectral partition is also unbalanced (320/219/174/37) |
| Shared charger reachability proves a global master is necessary | Not established: reachability is not simultaneous charging demand. A capacity model needs coordinated shared resources; a global master is one possible method |

## Which integer follow-ups can help?

| Saved component status | Count | Next useful test |
|---|---:|---|
| Eight buses, bound eight | 12 | Retain existing result; no fleet-search rerun needed |
| Nine or ten buses, bound eight | 15 | Longer MIP search can test whether eight exists in the saved pool |
| Nine or ten buses, bound nine | 12 | Eight is excluded from that pool: generate new columns if seeking eight |

One partition needs **all four** components at eight to construct a 32-bus union. “Most components reach eight” is insufficient. There are nine complete recorded partitions and one incomplete partition (d00 lacks g3).

Diving **with pricing** is new column generation and its cost must be counted. Reduced-cost fixing requires a valid bound and reduced costs for the same objective being optimized; the weighted CG objective cannot be substituted into a fleet-only fixing formula. Revalidate newly selected schedules and, when enabled, joint station capacities.

## Interpretation limits

The spectral figure shows +36 **additional** time-only buses, hence 68 total, not a 36-bus solution. It is a compatibility embedding, not a geographic map. Visual separation and the fraction of all arcs cut do not measure the loss of connections needed by an optimal schedule.

The increase between two time-only lower bounds is not generally a lower bound on the increase between two EV optima. For example, time-only bounds 32 and 34 can coexist with both EV optima equal to 35. Calling +2 a guaranteed EV penalty requires a matching unpartitioned EV upper bound of 32 under the same physics. The absolute partition bound of 34 remains valid.

LP weight eight alone does not establish an eight-bus integer EV schedule. Cross-component routes or overlapping decomposition may help integer feasibility even if every component LP equals eight. The present evidence supports prioritizing integer pool quality, while retaining the label-free cover partition as a separate experiment; it does not eliminate all value of changing the partition.

## Proposed order, not submitted

1. One longer MIP seed on the 15 open-gap pools; additional seeds only on misses.
2. Separately test integer-directed pricing on pools whose bound excludes eight.
3. Compare duty-based versus label-free cover components with the same physics and graph+CG+MIP accounting. Include a parent benchmark where feasible.

Source: `../../advisor_geography_review_20260917/README.md`, current `outputs/research_register/register.csv`, and `outputs/decomposition_union_audit_20260914/audit.json`. Register observations are saved-artifact checks, not a fresh cluster poll.
