# Fixed-dual capacity pricing diagnostics

This campaign compares the frozen reference and prefix-memo capacity selectors on identical saved RMP states. Each production case copies a hash-bound atomic pool, reconstructs and hashes the complete starting dual vector, and invokes exactly one new pricing call through driver commit `309d98d266ebaf6b7e99543a67f8f2be5736874a`.

The two states are duty 13407 immediately before iteration 16 and strict k2 240r0 capacity immediately before iteration 5. Existing runs show ordinary calls taking seconds while the first nonzero-capacity-dual call consumes hours. These four cases test whether that increase is selector-specific without repeating completed CG trajectories.

A returned negative route is one exact priced column at one fixed dual state. A `pricing_deadline` endpoint is censored and has no terminal reduced cost. Neither outcome is a complete-CG or full-model certificate unless the frozen driver itself returns `exact_nonnegative_reduced_cost`.

Production uses four independent default-partition jobs, one CPU and 24 GB each, a 14,400-second driver deadline, a 120-second outer grace, and a 4:15 allocation. Every job excludes `scaglione-compute-01`; requeue is disabled. No MIP is run.


The native cluster preflight is `202202`: it completed on `unicorn-cpu-74` with the shared license, reconstructed both production RMPs, verified the next iterations (16 and 5), and observed one reference and one prefix-memo smoke call using identical actual raw dual arguments and candidate payloads. The smoke state is validation-only and makes no certificate claim.

Collection is read-only:

```bash
python collect.py --root /home/nc437/ladder-lite/capacity_fixed_dual_20260914
```

The collector emits `evsp-dr-fixed-capacity-collector-v1`. Its records are pricing-call diagnostics (`fixed_master_single_exact_pricing_call`), not CG or MIP records. Pair equality is reported only after both arms publish validated endpoints; attempt progress also exposes failures or interrupted attempts that lack an endpoint.
