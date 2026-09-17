# Reviewer prediction check

Predictions remain unchanged. Current endpoints are compared separately from replay progress.

| Arm | Replay shards / 125 | CG weight | CG prediction | MIP fleet | MIP prediction |
|---|---:|---:|---|---:|---|
| baseline | 3 | None | PENDING | None | PENDING |
| parx60_only | 6 | None | PENDING | None | PENDING |
| reserve15_only | 6 | None | PENDING | None | PENDING |
| battery236p44_only | 4 | None | PENDING | None | PENDING |
| battery239p01_only | 3 | None | PENDING | None | PENDING |
| segregation_only | 5 | None | PENDING | None | PENDING |

Numeric agreement is not causal proof. Certified route weight belongs to one weighted-objective LP optimum and need not be unique. Finite-pool integer proof does not establish the full-model integer minimum. Replay progress from completed shards excludes partial work.

See comparison.json for exact stop reasons, certificates, group splits, physical-validation scope and source hashes.
