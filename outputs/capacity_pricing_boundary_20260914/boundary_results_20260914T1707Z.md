# Capacity-pricing boundary results — 2026-09-14 17:07Z

Verified collection SHA-256: `3424cf0276cdebcab6cfa60164d0e330e30431dad330698c0f2a1fca4e87e565`.

At this snapshot, the duty-13408 matched pair completed; the other six allocations were still running. Both endpoints passed the strict worker/manifest/input/artifact hash gates.

| selector | job | CG stop | certified | iterations | pool columns | route weight | MIP fleet | MIP stages | capacity audit |
|---|---:|---|---:|---:|---:|---:|---:|---|---|
| prefix-memo | 189171 | exact_nonnegative_reduced_cost | True | 33 | 43 | 1.0 | 1 | OPTIMAL/OPTIMAL | True |
| reference | 189170 | exact_nonnegative_reduced_cost | True | 33 | 43 | 1.0 | 1 | OPTIMAL/OPTIMAL | True |

The reference and prefix-memo duty-13408 cells both reached an exact nonnegative-reduced-cost certificate under identical settings. They have the same exact objective (100036.536) and byte-identical saved pool SHA-256 (`f124c6dc77ffe16d5b262e594f74292d3e5787418d1ec96823044c575953247f`), so this matched case supports selector equivalence for the certified endpoint. Both finite-pool MIPs returned fleet 1 and charging cost 36.536, proved both stages optimal for that pool, and passed the driver station-capacity audit.

The remaining duties 13405–13407 have no admitted native endpoint in this snapshot. Running worker progress is not treated as a result.
