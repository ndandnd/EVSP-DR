# Completed k33–40 extension

Figure: `full40_fleet_heatmap.png` (also PDF and editable-text SVG). Primary editable data: `full40_fleet_results.csv`. Rebuild with `build_full40.py`; input and output hashes are in `provenance.json`.

Rebuild command (bundled Python; existing system matplotlib because the bundled environment lacks it):

```sh
MPLCONFIGDIR=/private/tmp/evsp-dr-matplotlib PYTHONPATH=/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages /Users/nadan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 outputs/research_management_20260925/figures/build_full40.py
```

**Editable caption:** Final saved-pool MIP fleet across six chains and targets k=33–40. Cell numbers show actual buses; shading shows buses above that cell's target. All 48 endpoints miss their target and retain open finite-pool fleet gaps. All 48 CGs lack pricing certificates, so their restricted-master objectives are not certified full-model lower bounds. The source field is the final published `buses`, including the charging stage's selected incumbent; fleet-stage values are retained separately in the CSV. Native individual-route replay passes under baseline physics; duplicate removal and shared charger capacity remain unvalidated. This is a completed baseline extension, not a matched large-case fresh/sequential comparison.

**Physics and scope:** 240 kWh/240 kW, zero reserve, flat tariff, start fee 5, covering, 2.5 kWh/5-minute grid, no terminal-energy floor or shared capacity. Four-hour CG and one-hour two-stage MIP allowances; graph construction is separate. Full40 has three input classes: C1/C4 (948 trips), C2/C3/C6 (947), C5 (946). Final k40 fleets are 43/44/44/47/44/45; the approximately 39 finite-pool bound is a k40 statement, not an all-target bound. Exact per-case bounds are in the CSV. Source collection: 25 September 2026, 17:30 UTC.
