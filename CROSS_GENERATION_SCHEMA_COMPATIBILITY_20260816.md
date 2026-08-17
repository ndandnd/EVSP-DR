# Cross-generation schema compatibility

The pipeline preserves semantic differences instead of coercing every artifact
into falsely equivalent columns.

| Source schema | Objective field | Timing | Missing-by-design fields |
|---|---|---|---|
| Historical heuristic DP | `legacy_master_objective` from `Master_Obj` | Explicit per-iteration master/pricing and cumulative split | LP route weight, artificials, queue/tier/dominance instrumentation |
| Current heuristic DP | `master_objective_before_add` from `Master_Obj_Before_Add` | Explicit master/pricing split plus tier timings | Exact expanded-network phase timings unless separately instrumented |
| Exact expanded-network CG | `lp_objective` from `lp_obj` | Cumulative `elapsed_s`; no master/pricing split | Master/pricing shares remain null unless phase telemetry was enabled |
| MIP convergence | Integer incumbent/objective and finite-pool bounds | Callback observation time and nominal checkpoint mark | No global route-space proof |
| Endpoint JSON/manifests | Endpoint/provenance only | No trajectory implied | Time-to-event statistics remain censored/unavailable |

## Scientific guardrails

- LP route weight is not an integer fleet.
- Zero artificial mass does not certify pricing.
- Pricing certification does not prove a finite-pool integer master.
- Finite-pool MIP optimality is not global route-space optimality.
- An exact incidence partition is not a physically validated charging schedule.
- RAW, MATCHING, and GIRO-augmented MIP pools are separate treatments.
- Legacy `Master_Obj` and current `Master_Obj_Before_Add` are retained in
  separate normalized columns.
- Missing timing or provenance remains null with an availability reason.
- Master/pricing shares are calculated only from explicitly measured fields.

## Tail handling

CSV and JSONL sources are read without modification. One malformed final
append-only record may be ignored and recorded as
`interrupted_final_*_record`; malformed interior data fails validation.

## Migration policy

Historical artifacts are never rewritten. Compatibility is provided through
read-only adapters in `src/cross_generation_schema.py`. New schema support must
add an adapter and data-dictionary entry; it must not reinterpret an existing
normalized field.
