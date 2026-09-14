# Fixed-dual capacity pricing prefix retry 1

This immutable retry contains only the two prefix-memo arms from the parent campaign. The original attempts `202361` and `202363` recorded the correct actual dual arguments, then failed before selector construction because the wrapper removed the frozen checkout's `src` directory before the driver's lazy import. This wrapper eagerly imports the existing `capacity_window_selector.py` from the same detached commit and verifies its exact path; no solver or pricing algorithm is changed.

The parent reference calls `202362` and `202364` continue unchanged. Pair comparisons combine each validated retry endpoint with its parent reference endpoint. A result remains one fixed-master pricing call, never a complete-CG or full-model certificate.

Native gate `202418` completed in 2:00 on `snavely-cpu-12`. It entered the real prefix selector with one nonzero capacity dual for ten seconds, recorded raw dual hash `814db9b…09f84`, and loaded selector SHA `9743c2c5…f0131` from the detached execution checkout. The expected censored gate result has no reduced-cost or certificate claim.

Collect with `python collect.py --root /home/nc437/ladder-lite/capacity_fixed_dual_retry_20260914`. This retry root contains prefix endpoints only; combine each by `pair_id` with the corresponding reference endpoint in parent manifest `fe3eea1b…c62e9`.
