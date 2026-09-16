# F9 — register and chronology repair

**VERIFIED and corrected:** chain fee/tariff fields were missing even though pinned code and manifests specified them. The register builder now reads the audited chain campaign identity instead of assigning global defaults.308 case/telemetry rows receive fee5, flat tariff/hash, uniform PARX/non-PARX240kW, reserve0, no shared capacity and no terminal floor. The16 workflow records are not solver experiments and are not filled with inferred settings.

The rebuilt register retains all3447 rows and their stable identifiers. Existing objective, fleet, pool-bound, proof, input-hash and source-path fields are unchanged.102 CG rows gain separate **numerical event-model lower bound** columns. Original CG-certificate and finite-pool proof flags are not overwritten. Replay counts distinguish unchanged original charging schedules from reoptimized fixed-trip schedules.

The current cumulative-budget README's “largest25” headline was scoped to its through-k25 subset, despite sharing a later collection timestamp. It now states that scope and links the complete102-row k16–32 table. The underlying historical results are preserved.

Artifacts: `validation.json` checks IDs, row count and unchanged scientific-result fields; `build_register_before.py` preserves the prior builder; `published_files.json` lists regenerated outputs. `../audited_chain_results.csv` combines F3 bounds and F4 replay counts without overwriting the frozen source table.

Capacity-pricing timeout pairs remain time-censored and non-comparable, with no pricing proof: no speedup or equivalence claim is added.
