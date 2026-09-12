# Independent implementation audit

Reviewed accounting-only commit `550bc795b18f801dca3a07b8becc1dc4c5527abb`. Event network passes station-specific power through realization mapping into cost reconstruction. Override-before-base-alias-before-global precedence matches graph power semantics. Block splitting, realized energy allocation and final power validation now use the station rate. Independently reran `python3 -m unittest discover -s tests -p test_capacity_accounting_regression.py`: two test methods passed, including eight whole-route reduced-cost subcases and fractional/multiple-station tariff-block checks. Whole-route RC is independently reconstructed from record cost, trip duals and occupancy union, with explicit expected tariff costs at 60kW. No blocker found in the event-capacity CG path.

Generic `run_exact_pool_mip.py` and ordinary seed readers still validate using their global power configuration; this pre-existing scope does not establish heterogeneous-power generic MIP validation. The fix enables the dedicated capacity event accounting path and must not be described more broadly.

Reviewed master omission commit `9a957e9a1a317988660396c5d06e4ccfc5dbd679`: ordinary, diversification and final LP resolve use the helper, SciPy retains matrix creation, Gurobi route synchronization validates all routes. No numerical blocker identified by static review.

Reviewed preliminary indexed-replay changes: successor bounds select exactly the contiguous target-trip range or sink from sorted rows; order is preserved, pricing traversal unchanged, caches do not persist runtime index mode, and legacy opt-in validation checks row ordering. Index setup is included in graph build/total wall time but excluded from the existing inherited-import audit timer. Report total setup-inclusive time. Strict source-hash cache identity remains enforced; deployer must build valid caches rather than silently bypassing provenance. Local pytest invocation was unavailable (`No module named pytest`); owner reports direct source-only test execution.

Reviewed preliminary capacity prefix selector: fresh pricing-call scope prevents stale dual reuse; memo charge key includes station, feasible interval, energy, power and grid. Candidate ordering matches source; conservative floating error envelope invokes exact row sums near ties and always for the returned winner. This is a preliminary static review pending owner's final commit and regression evidence.

Final replay target is `0d3f1129`; owner reports16 focused tests passing, including binding max_columns2 and production-shaped512/900s/8-worker importer controls. Static review remains clear.

Independently executed selector regression suite from owner checkout: `python3 -m unittest discover -s tests -p test_capacity_window_selector.py`, four tests passed in1.350s. Suite covers4,320 exact arc comparisons plus memo repeats, eight whole-route equal records and independent RC, changed dual/configuration, deadline and storage caps. No blocker identified for paired opt-in validation.

## Final integration verification

Supersedes preliminary test-availability statements above. Final capacity selector commit is `309d98d266ebaf6b7e99543a67f8f2be5736874a`, including the final infeasible-duration guard. Final focused evidence is recorded in `../local_validation.json`; the integrated master/replay pytest suite passed 34 tests using an isolated pytest installation. The independent eight-trip full-CG comparison and eight whole-route station-power accounting controls are saved under `../local_smoke/`. These checks establish the stated tested equivalences, not a full-scale cluster speedup.
