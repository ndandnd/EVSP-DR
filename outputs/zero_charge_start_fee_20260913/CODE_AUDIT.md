# Zero charge-start fee code audit

## Source pins

This branch starts from CG commit `e091a4dba549510238507ef5e5367abea958bd30`. Before fee changes, `src/run_exact_pool_mip.py` and its test file were copied byte-for-byte from MIP commit `871d057e1067411f09581e37d78f7c1ca43f68bb`; the staged Git blob IDs matched that commit (`5c7df2ede9c617fa68564e4f6e4784d8e06ecd76` and `a218a996bfec78ecc4de2e13bfb0f1c1c662c45e`). The resulting commit is one unified CG/MIP execution pin.

## Finding and implementation

The exact expanded-grid CG had no fee CLI. Event and uniform graph construction, event-route replay, inherited-sequence replay, journal costs, strict MIP pool replay, injected-route repricing, and final selected-route validation ultimately used the module constant `CHARGE_START_COST=5.0`. No `value or 5` expression was found; zero was unavailable because the constant was imported directly.

`exact_pricer_expanded.py` now accepts a finite nonnegative `--charge-start-cost` and records it in status, resume identity, provenance arguments, network audit, and every new journal column. Event and uniform arc construction use the argument. Inherited columns retain only ordered trip sequences and are solved again on the child graph, so same-k coverage patterns are preserved while charging timing, activity count, energy, and cost are recomputed for the destination fee. Source cost and source/destination fees remain separate provenance.

`run_exact_pool_mip.py` accepts the same flag and requires it to match the source CG status. A legacy status or journal record with no fee field means the historical fee 5. A fee-0 status therefore cannot admit a missing-fee or fee-5 journal record. Strict pool replay, deterministic repair, injected routes, initial partitions, and final selected-route validation all use the bound fee. Stage 2 retains the pinned 871 constraint `sum(a) <= stage1_buses`, including the validated-start fallback behavior.

Route and result metadata now separates charging activities, charge-start fee subtotal, expanded and continuous energy, and expanded and continuous electricity cost. A charge activity is one charging stop, not one tariff block.

## Cache safety

Event cache geometry is independent of the fixed fee, but packed and explicit cache arc costs contain the fee. On load, the in-memory network changes only charge-entry arc costs by the fee delta, including terminal charge arcs. Lazy caches identify charge arcs from the existing nonzero action recipe; explicit caches identify `action.kind == "charge"`. The cache pickle and manifest are never modified. A focused test shows fee-5-to-fee-0 repricing equals a freshly constructed fee-0 graph for both cache modes and leaves the pickle hash unchanged.

Strict cache identity remains the default. `--event-network-cache-source-commit COMMIT` permits an audited older cache only when COMMIT equals the manifest's source commit and `git_commit` is the sole identity difference. Pickle hash, all input/physics identities, object type, and network metrics remain mandatory. The runtime compatibility decision and fee-repricing audit are recorded. A manifest already rewritten to the execution commit loads normally even when the explicit source-commit flag is present.

A real large-cache Linux smoke is still required before the campaign: use `--event-network-cache-mode require --event-network-cache-only --event-network-cache-source-commit <manifest identity git_commit> --charge-start-cost 0`. The returned network audit must show a cache hit, current fee 0, and a positive repriced arc count; the source pickle and manifest hashes must remain unchanged.

## Scope and validation

The production scope implemented here is exact expanded-grid CG, fixed-duty expanded seed validation, event inherited-pool replay, and pinned two-stage exact pool MIP. `fixed_duty_continuous_optimizer.py` already has an explicit nonnegative `charge_start_cost`; historical GIRO/audit frontends that import `CHARGE_START_COST` directly were not changed and must either pass zero into that optimizer from a controlled wrapper or remain fee-5-only.

Validation on macOS Python 3.12:

- `python3 -m py_compile src/*.py`
- `git diff --check`
- 60 focused event cache, inheritance, realization, and pinned MIP tests passed, with 8 subtests.
- 5 production fixed-duty transition/trace tests passed; 2 tracked generated-artifact determinism tests were deliberately excluded because their certificates bind the optimizer source hash and require artifact regeneration in a separate evidence update.
