# records/ — durable project record

Append-only CSVs. **Rows are never edited in place.** A correction is a
new row whose `supersedes` (or `notes`) names the row it replaces. This keeps the
record auditable when a result or a diagnosis later turns out to be wrong.

| File | One row per | Written by |
|---|---|---|
| `DECISION_LOG.csv` | irreversible or scientifically material decision | human, by hand |
| `BUG_LOG.csv` | defect, with symptom, root cause, fix commit, status | human, by hand |
| `RESULTS_LOG.csv` | completed ladder cell | `scripts/ladder_lite/record_results.sh` |
| `runs/<RUN_ID>/` | full normalized CSV set for one campaign run | `scripts/ladder_lite/record_results.sh` |
| `ID_REGISTRY.csv` | canonical B/D ID allocation | records curator only |
| `ID_MAPPING.csv` | colliding feature-branch `(branch, old_id)` → canonical ID | records curator only |

## Freeze (2026-08-21)

`DECISION_LOG.csv` and `BUG_LOG.csv` are **frozen** (`B0001`–`B0031`,
`D0001`–`D0033`) and unfrozen only through the records curator. Feature-branch
agents write findings to `records/inbox/<branch>.md` with provisional
`LOCAL-N` labels; authoritative IDs come only from `ID_REGISTRY.csv`. All
future allocations use **`B0100+` / `D0100+`**; `B0032`–`B0099` and
`D0034`–`D0099` are retired and never assigned. See
`CURATOR_NOTES_20260821.md`.

## Conventions

- `date_utc` is `YYYY-MM-DD`; timestamps inside a run use ISO-8601 UTC.
- `id` is `D####` / `B####`, monotonic, never reused.
- `status` for bugs: `open`, `open-mitigated`, `fixed-in-<commit>`,
  `wont-fix-abandoned`, `superseded-by-<thing>`.
- `route_weight_meaning` in `RESULTS_LOG.csv` must read
  `combined-cost-master route weight` until the three-phase lexicographic master
  exists. It is **not** a fleet LP lower bound. See decision `D0005`.
- `label` carries provenance honestly: `ladder_lite_direct_array`,
  `legacy_scheduler_unverified`, `diagnostic_only`, `budget_overridden`.
- RAW and KNOWN cells never share a row. `arm` distinguishes them; a
  `KNOWN-PARTITION` row is a plumbing positive control, not algorithmic
  recovery.
- Missing or censored cells get a row with `status=missing` or
  `status=censored` and a populated `censor_reason`. Absence is data; silent
  omission is not.
