# k02_s2 local cross-resolution pool union

This is a local, no-cluster decision run over certified RAW exact-CG journals
generated from the same instance and physics:

- instance SHA-256:
  `6ca7e2db690120699d59fc81428b0f1af00c5cc3889770e8f2860af040244932`
- physics: `g_kwh=300`, `charge_kw=300`, `min_soc_frac=0`
- solver: `scipy.optimize.milp` / HiGHS, constant-zero target-feasibility
  objective

## Source pools

| Grid | Certified | Final columns | Route weight | Journal SHA-256 |
|---|---:|---:|---:|---|
| 15 kWh / 10 min | yes | 548 | 2.1875 | `33ae967934595f2a6eb103af3dfa2d5bd928b505aa59e3d00bf13a7d597ef393` |
| 1 kWh / 5 min | yes | 930 | 2.0000 | `30cb8a10565e79bb5c40491e5ad7232545f5667fd2ff8548b263429b384b415d` |

The physical gate admitted 566 and 937 append-only journal records,
respectively. Complete-route hashing produced a 1,503-record union with journal
SHA-256
`6616d34ab3801eb354e10dccb2191878782761da72e9820aa7c63fef1df413b8`.
The union's source-superset check passed. Identity and physics mismatches remain
fail-closed.

## Outcomes and controls

| Pool | Target | Outcome | Witness |
|---|---:|---|---:|
| primary 15/10 only | 4 | FEASIBLE | 4 |
| primary 15/10 only | 3 | INFEASIBLE | — |
| primary 15/10 only | 2 | INFEASIBLE | — |
| fine 1/5 only | 2 | FEASIBLE | 2 |
| cross-resolution union | 3 | FEASIBLE | 3 |
| cross-resolution union | 2 | FEASIBLE | 2 |

The requested target was reached, and target 2 was also reached. Strict
continuous replay validated both union witnesses.

However, this does **not** show that pool enrichment repairs the primary-grid
gap:

- all three routes in the target-3 witness came from the 1/5 journal;
- the 1/5 pool reaches target 2 without the union;
- the target-2 union witness mixes one 15/10 route and one 1/5 route;
- the primary finite pool itself remains infeasible at target 3.

Thus the result changes the admitted route space rather than recovering the
missing 15/10 columns. It cannot substitute for the branch-and-price proof that
the primary-grid optimum is 3. A decisive pool-composition test must union
independent **15/10** journals (or primary-grid columns exposed by the search
tree), not mix discretizations.

Machine-readable identities, source counts, outcomes, and witness attribution
are in `summary.json`. Full generated journals and witnesses remain under the
Git-ignored local directory `results/k02_s2_union_local_20260820/`.
