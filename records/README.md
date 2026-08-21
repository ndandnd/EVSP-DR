# records/ — durable project record

The three authoritative CSVs are curator-owned and append-only. Branch agents
must not allocate `B####`/`D####` IDs or append result rows: parallel branches
cannot coordinate the global ID namespace.

| File | One row per | Written by |
|---|---|---|
| `DECISION_LOG.csv` | irreversible or scientifically material decision | records curator |
| `BUG_LOG.csv` | defect, with symptom, root cause, fix commit, status | records curator |
| `RESULTS_LOG.csv` | completed ladder cell | curator-controlled ingestion |
| `runs/<RUN_ID>/` | full normalized CSV set for one campaign run | `scripts/ladder_lite/record_results.sh` |
| `inbox/<branch>.md` | provisional branch-local findings | branch agent |

Branch inbox entries use `LOCAL-1`, `LOCAL-2`, ... and include the claim,
evidence path, and producing commit SHA. These labels are never authoritative;
the curator assigns final IDs after cross-branch deduplication and validation.

## Conventions

- `date_utc` is `YYYY-MM-DD`; timestamps inside a run use ISO-8601 UTC.
- `id` is `D####` / `B####`, monotonic, never reused.
- `status` for bugs: `open`, `open-mitigated`, `fixed-in-<commit>`,
  `wont-fix-abandoned`, `superseded-by-<thing>`.
- Per `D0019` (which supersedes `D0005`), a pricing-certified cell's
  `route_weight_meaning` must read
  `fleet LP lower bound (certified discretized model; grid stated; D0019)`.
  An uncertified cell must instead read
  `upper bound on LP optimum only; no fleet LP lower bound`. Never use one
  label for both states.
- `label` carries provenance honestly: `ladder_lite_direct_array`,
  `legacy_scheduler_unverified`, `diagnostic_only`, `budget_overridden`.
- RAW and KNOWN cells never share a row. `arm` distinguishes them; a
  `KNOWN-PARTITION` row is a plumbing positive control, not algorithmic
  recovery.
- Missing or censored cells get a row with `status=missing` or
  `status=censored` and a populated `censor_reason`. Absence is data; silent
  omission is not.
