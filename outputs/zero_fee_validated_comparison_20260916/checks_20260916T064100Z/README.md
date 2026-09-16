# Zero-fee comparison: validated trip assignments

All costs below use continuously replayed charging, in tariff cost units. Each completed cleanup covers all 62 trips exactly once using five buses. Cleanup deletes repeated trips from the selected CG duties and reoptimizes their charging; it imports no GIRO duties and is a separate postprocessing step.

| Tariff peak | Fixed GIRO duties: charging optimized | Fresh CG + validated cleanup | Reduction | Same total ending energy? |
|---|---:|---:|---:|---|
| 08:00 | 128.29 | 124.69 | 2.81% | True |
| 12:00 | 164.23 | 160.61 | 2.21% | True |
| 18:00 | 95.29 | 88.30 | 7.34% | True |

These results use 240 kWh batteries, 350 kW charging, no reserve and no shared-station capacity. The common aggregate ending-energy minimum is 280.7833253 kWh; it is not a per-bus SOC requirement. Completed 08:00/12:00 selections and their fixed-duty comparators return with 281.1700005 kWh in total.

Full CG reported convergence on its weighted event-graph objective at 15.2, 16.8 and 22.2 minutes (graph construction excluded). Its original covering selections repeat trips. Requiring exact-once coverage using only those unchanged pools rules out five buses at 08:00 and 12:00; 18:00 has a five-bus exact-once selection. This is a pool limitation: deleting duplicated trips and creating new charging variants can produce feasible schedules outside that pool, as the completed cleanups show.

The 08:00 and 12:00 cleanup MIPs prove their grid charging objectives optimal within the cleanup pools. The 18:00 search ends at its one-hour limit: grid incumbent 90.403385, bound 90.280210, a 0.1363% gap. Its feasible improvement is verified; charging optimality remains unproved. All three fixed and cleaned schedules have matching total ending energy (281.1700005 kWh at 08:00/12:00, 282.9 kWh at 18:00).

A charging proof in the cleanup pool is not a full-model charging proof. Continuous replayed costs are not the grid objective covered by the solver proof. The results support an improvement over the tested fixed-duty optimizer under these declared assumptions, not universal superiority or full GIRO feasibility.

Sources: [comparison and hashes](comparison.json), [full-CG manifest](../zero_fee_full_cg_20260916/manifest.json), [cleanup manifest](../terminal_duplicate_cleanup_20260916/manifest.json), [exact-once pool check](../terminal_exact_once_20260916/manifest.json).
