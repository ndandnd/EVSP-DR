# Independent parent service-overlap bound

**At least29buses are required. This bound is below the32bus target.** The first maximum occurs during the half-open service-clock interval **[08:13,08:14)**. Six maximum-overlap intervals are retained in the JSON audit.

All750rows of the frozen parent were included, with no additional exclusions. Exact input SHA256: `4367335166098c6c50fb283b1cd3307a72720ea0b70fff4567b085af9a37e66e`. The audit uses actual `Start1`/`End1` values, without rounding or deadhead/charging padding; trips ending at a timestamp leave before trips starting then enter. All intervals have positive duration and all stable IDs are unique. An independent direct count at every service start agrees with the event sweep.

The29simultaneously active `Ordered_Trip_ID` values for the first witness are:

28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 206, 217, 244, 246, 249, 251, 252, 379, 380, 381, 382, 499, 500, 588, 590, 787, 788, 899, 900.

A future physically validated32bus schedule covering the full frozen input establishes **29 ≤ minimum fleet ≤32**. It does **not** prove global fleet optimality from this overlap bound. A32bus finite-pool optimum would still be a restricted-pool proof. This audit supplies a structural fleet lower bound, not a CG/pricing certificate or weighted-cost bound.

[Full intervals and trip timestamps](parent_overlap_bound.json) · [Reproducible read-only audit](audit_parent_overlap.py)
