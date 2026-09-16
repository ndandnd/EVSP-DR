# Review monitoring — 16 September, 18:30 EDT

SSH and all six focused collectors succeeded; collection took 1.99 seconds. Registered jobs: **89 running, 79 pending**. No failed public worker attempts or unsatisfiable dependencies were observed. No jobs submitted, cancelled or requeued in this check. Held historical jobs and other projects were untouched.

The main audit and Frölunda k1/k2 findings were just reported to the user. This check records additional **native endpoints awaiting independent validation**, without sending another notification or changing the current Doc's conclusions:

| Finding | Newly collected evidence | Interpretation |
|---|---|---|
| F4 | First strict component, C5 k1 / 18E1: native CG certificate, one-bus pool proof; PARX 60 kW, battery 236.44 kWh, reserve 35.466 kWh. CG driver runtime 811.91 s, pricing 782.17 s, iteration LP 0.039 s. | One small component, not the k31 union. Native route feasibility is by construction; independent physical replay remains to be checked. |
| F5 | Random stages 4/5 report 7/8 buses, saved-pool fleet proofs and individual-route replay. Stage 5 CG certified. | Stage numbers are not GIRO fleet targets. Duplicate-service dispatch validation remains separate. The final matched-set comparison is unfinished. |
| F8 | Frölunda k3: 67 trips, 3 buses, native pricing certificate in 115.39 s, pool proof and individual-route replay. | Duplicate-removal flag is false; do not extend the independently verified k1/k2 dispatch claims until service-assignment replay is checked. |

P1 longer-search hypotheses, the larger demand-response experiment, full Partille and the larger strict/Frölunda ladders remain unresolved. Source paths, hashes, native flags and differences are retained in `snapshot.json` and `delta_raw.json`. SE3 scientific outputs remain remote.

Next hourly check: independently validate these new endpoints, then consolidate a useful batch into the current tables. Current Doc still accurately reports the last independently verified k1/k2 cases. No figure, slide or workbook regeneration is warranted for these provisional flags alone.

Preemption study refreshed: 1,044 allocation-attempt rows; no newly confirmed preemptions. The historical registry and review-only scheduler snapshot cover different populations.
