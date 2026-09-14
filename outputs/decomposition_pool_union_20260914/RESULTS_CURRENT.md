# Current decomposition pool-union results

Reported at 2026-09-14T18:28:39.689550+00:00 against manifest 69cf4c1f7c3795ce17ce62adb4bef2dd7d34194a5b95a86684e4920ec27af4c0.

All nine data-only constructions passed their source hashes, real trip attribute mappings, mandatory-cover checks, and pool caps. The native whole-parent fixture admitted all 2,048 source columns with zero rejections or repairs and returned the expected 34-bus covering solution.

Of 46 production MIPs, 45 have published verified results; m_all09 is still running. The completed result counts are {"34": 9, "35": 30, "36": 5, "37": 1}. The best result remains 34 buses. No completed pair has improved on the best single-partition source upper bound.

These runs compare pool treatments on one 750-trip parent; they are not independent instance replications. Their MIP budgets are matched across single-partition controls and pairs, while the all-nine case has the separately declared four-hour budget. Pool construction performs no CG and supplies no pricing certificate or full-model LP bound. A value above 32 cannot establish that a 32-bus schedule is impossible.
