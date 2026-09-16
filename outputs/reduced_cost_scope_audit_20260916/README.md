# Reduced-cost correction: bounded impact audit

The correction in c210187b restricts subtraction of the 100,000 bus cost to the charging-only objective. Previously, lazy pricing also subtracted it under the combined objective when the route/fleet dual was nonzero.

**The active chain driver at a0e0bb7 does not enter this faulty path.** Its ordinary CG calls (exact_pricer_expanded.py:2697 and 2711) pass trip duals and column-selection options, leaving `objective="combined-cost"` and `route_dual=0.0`. Its optional diversification calls do likewise. The zero-dual combined branch uses the original arc costs directly. The separate fleet-only certificate call uses `objective="fleet-only"`, another unaffected branch.

This is a static call-path audit of the frozen current chain revision, not a complete historical audit and not a general correctness proof. Historical custom adapters and other execution commits still require checking. Do not attribute historical performance differences to this bug without showing that an affected call occurred. The new terminal-aware full-CG campaign is pinned to the corrected revision and has enumeration-based tests.

Sources: frozen revision and SHA-256 values in [source_hashes.json](source_hashes.json); current chain execution is recorded in [manifest](../chain_extension_20260915/manifest.json).
