# Final LP support and integer pool recovery

This campaign asks whether a small set of columns used by a successful donor's final fractional LP repairs an old pool that was proved to require an extra bus. It compares 13 selected difficult recipient pools, not a random sample.

Each recipient receives two matched treatments: add all donor positive-LP-weight incidences absent from the recipient, or add exactly as many absent donor zero-LP-weight incidences. The latter are selected by SHA-256 rank of the canonical sorted trip-ID JSON, independently of the donor's selected integer routes. Zero weight does not mean zero reduced cost. The donor is always the matching completed compact CORE run; no best-arm selection is used. Nine additional controls solve each unique donor's positive support alone.

All 35 pools were built before production submission. Each paired augmentation adds 33–83 columns, without replacing original recipient columns. Support-only controls contain 80–122 columns. There were no zero-novel or insufficient-control exclusions. Every donor's exact frozen positive weights reconstruct trip coverage, route weight and weighted objective with zero artificials. A pool construction is not a CG run or a new pricing certificate.

The 13 recipients are C1 k8 integer/LP-weight and k10 integer; C2 k8 LP-weight; C3 k8 integer/LP-weight; C4 k8 and k10 integer/LP-weight; C5 k8 and k10 LP-weight; C6 k10 LP-weight. All original pools have a proved fleet above target. All nine matching donor pools have a physically validated target fleet. Original and donor computation costs remain in `source_index.json` and the manifest.

Every production MIP gets 12,600 seconds total, 10,800 seconds fleet-stage budget, 8 CPUs, 24 GiB requested RAM and a 4.5-hour allocation on default_partition, excluding scaglione-compute-01. All 35 are eligible independently with no CG or construction scheduler dependencies. Requeued attempts start a new MIP tree and use separate job/restart paths with locked registry appends. The copied base worker retains its generic cohort; campaign root and case IDs identify the population.

Remote root: `/home/nc437/ladder-lite/lp_support_pool_diagnostic_20260914`; case and pool storage: `/share/scaglione/nc437/evsp-dr/lp_support_pool_diagnostic_20260914`. Published results use `cases/<case>/mip_result.json` and `completion.json`. Synthetic sources are named `pool.json`, never `cg.json`.

`manifest.json` freezes sources, model, solver, resources, pools and tools. `source_index.json` records the original proof and donor source identities. `preflight.json` records exact source, input, proof and path gates. `validation.json` records native smoke checks on all three construction paths. `jobs.json` and `scheduler_verification.json` record actual production admission. Construction allocations are operational preparation, not extra research cases.

Shared-column audit: all shared donor-positive incidences have the same ordered trips. Five recipient pools retain recorded-cost differences up to 0.496 on some shared incidences. The augmentation preserves those recipient records; exact donor weighted-LP reconstruction is established for donor support alone, not asserted for every augmented pool. This is a controlled novel-incidence addition experiment.

Production launch: 35 jobs in the 207005–207040 range (one unrelated interleaved ID; exact map in jobs.json). At 2026-09-15T01:48:30.513942+00:00, seven were COMPLETED and 28 RUNNING. All 35 passed the native large-license check. Registry had 26 unique production attempts at this snapshot; remaining workers were completing hash preflight before appending. No CG dependencies or node-exclusion violations were observed.
