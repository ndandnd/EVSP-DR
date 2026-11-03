# Partille k8 SOC-grid pilot

Collected 2026-11-03T12:51:11.789157+00:00. The 2.5 kWh rows are the registered October v15 controls. The 5 kWh rows are November array 917989. The three chain inputs, flat tariff, 60 kW PARX depot, 240 kW opportunity chargers, 240 kWh battery, per-start fee of 5, one-minute plug limits and fresh covering search match their respective control.

## Search and resources

| Chain | SOC kWh | Job state | Arcs, millions | Graph min | Root CG min | Root columns | Post-dive columns | Job min | Peak GiB |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| C2 | 2.5 | COMPLETED | 71.33 | 132.8 | 141.1 | 28,253 | 39,390 | 424.3 | 22.33 |
| C2 | 5.0 | COMPLETED | 18.03 | 33.2 | 79.6 | 25,445 | 39,807 | 263.5 | 7.02 |
| C3 | 2.5 | COMPLETED | 55.49 | 107.1 | 91.0 | 21,447 | 35,956 | 293.1 | 16.55 |
| C3 | 5.0 | COMPLETED | 14.03 | 22.1 | 43.3 | 19,621 | 42,630 | 169.7 | 5.07 |
| C6 | 2.5 | COMPLETED | 32.00 | 28.0 | 91.1 | 24,125 | 40,304 | 210.2 | 7.55 |
| C6 | 5.0 | COMPLETED | 8.06 | 6.4 | 45.4 | 21,518 | 30,937 | 116.3 | 2.77 |

Jobs had eight CPUs, 48 GiB RAM, and a 20-hour limit. Each job built a fresh graph. The root and simple-dive LP master used one solver thread. Finite-pool MIPs could use eight. The registered budgets were 180 min for root CG, 90 min for the simple dive and 60 min shared by the fleet and charging MIPs. Memory is peak resident memory for the batch step. Separate per-phase memory measurements are unavailable. The runs used different CPU models. The treatment jobs also include terminal replay. Wall times describe these attempts.

## Pricing certificates and integer schedules

| Chain | SOC kWh | Weighted grid LP lower bound | Route weight | Fleet / pool bound | Extra visits | Grid cost + fees | Continuous cost + fees | Finite-pool charging lower bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C2 | 2.5 | 800,236.677061 | 8.0000 | 9 / 8 | 44 | 324.736 | 310.100 | 215.176 |
| C2 | 5.0 | 800,270.066140 | 8.0000 | 8 / 8 | 19 | 317.504 | 282.611 | 294.194 |
| C3 | 2.5 | 800,212.819176 | 8.0000 | 8 / 8 | 5 | 225.912 | 213.593 | 225.912 |
| C3 | 5.0 | 800,236.886000 | 8.0000 | 8 / 8 | 7 | 259.472 | 234.630 | 259.472 |
| C6 | 2.5 | 800,295.169333 | 8.0000 | 8 / 8 | 3 | 310.432 | 301.661 | 310.432 |
| C6 | 5.0 | 800,322.256686 | 8.0000 | 8 / 8 | 0 | 325.024 | 299.812 | 325.024 |

The root lower bound combines the 100,000 bus coefficient with grid charging cost and start fees. It uses the minimum reduced cost from full conservative pricing and the registered fleet cap. The effective reduced-cost tolerance is shown per row in `paired.csv`. Each charging-stage bound applies to a fixed pool with its corresponding fleet cap and grid. Continuous costs come from replay of the selected schedules. A covering selection can visit a passenger trip multiple times. Every terminal selection passed independent route and shared-plug replay.

## Paired comparisons

- **C2**. The coarse graph had 74.7% fewer arcs, graph construction took 75.0% less time and root CG took 43.6% less time. Batch peak memory fell 68.5%. The fleet changed from 9 to 8, with 44 and 19 extra passenger-trip visits, respectively.
- **C3**. The coarse graph had 74.7% fewer arcs, graph construction took 79.4% less time and root CG took 52.5% less time. Batch peak memory fell 69.3%. The fleet changed from 8 to 8, with 5 and 7 extra passenger-trip visits, respectively. At that fleet the grid cost with fees changed by +33.560, and the continuous replay cost changed by +21.037.
- **C6**. The coarse graph had 74.8% fewer arcs, graph construction took 77.0% less time and root CG took 50.2% less time. Batch peak memory fell 63.4%. The fleet changed from 8 to 8, with 3 and 0 extra passenger-trip visits, respectively. At that fleet the grid cost with fees changed by +14.592, and the continuous replay cost changed by -1.849. The coarse selection covered all 128 trips exactly once with zero plug overloads. Its fleet attains the independently counted 8-trip overlap lower bound.

`aggregate.json` records hashes and solver status for each arm, including the original and replayed selection, root journal, post-dive pool, immutable source revision and verifier. The collector authenticates all six arms against the prelaunch manifest and requires byte-identical replay receipts for every terminal case.
