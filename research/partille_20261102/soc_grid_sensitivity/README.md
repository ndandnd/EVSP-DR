# Partille k8 SOC-grid sensitivity

The registered comparison changes the battery state grid from 2.5 to 5.0 kWh for chains C2, C3 and C6. The 2.5 kWh controls are the archived October v15 PARX60 runs. All three controls ran on the default partition with eight CPUs. Each treatment uses the same frozen source commit and task arguments, adding one explicit `--soc-step 5.0`. A unique output directory captures each restart. Graphs and route pools are reconstructed from scratch.

`registration/manifest.json` identifies the cases, inputs, control result hashes, solver, procedure, resource requests and comparisons before submission. It binds the seven original worker, smoke and replay files by hash. `submission.json` records the array ID and verified resource settings. The November pilot uses normal scheduler priority, whereas the original controls used nice 10000. The treatments and controls run on different dates and may run on different CPU models. Time-limited fleet and charging comparisons describe these attempts.

| Replayed 2.5 kWh control | Fleet | Extra trip visits | Grid charging + fees | Continuous charging + fees |
|---|---:|---:|---:|---:|
| C2 | 9 | 44 | 324.736 | 310.100 |
| C3 | 8 | 5 | 225.912 | 213.593 |
| C6 | 8 | 3 | 310.432 | 301.661 |

Each selected control covers every trip and passes independent route replay and the shared one-minute plug rows. Each has duplicated trip service. The registered fine and coarse arms use a covering master, the recorded flat tariff, a per-start fee of 5, a 240 kWh battery, 240 kW opportunity chargers, PARX 60 kW with unlimited plugs, and zero reserve and terminal requirement. The full comparison will separate graph time and size, weighted grid-LP certificates, finite-pool fleet and cost, and independently validated selected schedules. No optimistic bound was launched.

The four-trip end-to-end CLI smoke certified both grid roots and produced MIP selections. The graph had 432 arcs at 2.5 kWh and 218 at 5 kWh. The separate grid-aware validator reproduced total terminal energy of 536.769 and 534.269 kWh across the respective selections. The production replay also checks input hashes, final journal membership, charge windows, billed costs and minute-by-minute finite plug limits.
