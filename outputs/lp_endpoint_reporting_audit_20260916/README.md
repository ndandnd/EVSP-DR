# LP endpoint labels corrected

The chain-table builder labeled every `final_lp` record as a final-pool re-solve. That field can instead retain the last successfully solved iteration when the final re-solve runs out of time. Ten of 96 collected CG endpoints at the 09:42 UTC snapshot use this fallback. All ten already have pricing certificate = false.

The corrected table reads the explicit `final_lp.source` field. The register and workbook now expose that source too. No objective values, fleet counts, solver execution or certificate flags change. The earlier report wording claiming a final-pool re-solve for every row is superseded by this correction.

For example, chain2 target31 records the last solved iteration187, with217,870 columns, while CG subsequently saved217,900 columns. Its weighted objective3,101,231.4802128877 and fractional weight31 describe that solved iteration; they are not evidence that the larger final pool was solved or that pricing converged.

The audit asserts all ten fallback cases are uncertified and all certified chain endpoints use a final-pool re-solve. Source hashes and affected cases are in checks.json. Neither endpoint source alone proves full-model optimality.
