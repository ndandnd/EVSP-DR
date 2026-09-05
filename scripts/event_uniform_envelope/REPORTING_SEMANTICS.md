# Event-CG Reporting Semantics

This reporting contract is intentionally separate from the solver and from
the immutable campaign artifacts.

| Field | Meaning |
| --- | --- |
| `cg_route_weight_endpoint` | Raw `final.route_weight` at the observed combined-cost CG endpoint. Descriptive only. |
| `lp_route_weight_matches_target` | Descriptive flag: the endpoint ceiling equals the row target. It is not a proof. |
| `certified_rc_optimal` / `certified` | Reduced-cost certification of the combined-cost CG run. These fields retain their existing meaning. |
| `L_model` | Populated only from a hash-authenticated, certified fleet-only LP certificate for the same CG status, journal, cell, and representation, with no residual artificials. |
| `I_model` | Integer bus count of a matching, physically valid witness that closes `ceil(L_model)`. |
| `I_model_proven` | True only when both the `L_model` certificate and same-representation integer witness are authenticated and the witness count equals `ceil(L_model)`. |
| `fleet_target_proved` | Deprecated compatibility alias for the formal `I_model_proven` count. It is no longer derived from the combined-cost endpoint. |

The old `L_model` column in historical CSVs was ambiguous: it contained the
combined-cost endpoint. `summarize_cg_frontier.py` reads that old value into
`cg_route_weight_endpoint` for backward readability and marks
`legacy_l_model_endpoint_fallback=True`; it never treats it as a model proof.

An authenticated fleet certificate must carry the existing phase-2 certificate
schema and digest, exact source status and journal hashes, and enough source
identity to bind its cell and representation. An integer witness must carry
the same source hashes, `physical_witness_valid=true`, and an integer bus
count. A continuous physical upper-bound flag is never promoted to an event
representation witness. Missing or conflicting identity fails closed.

Historical `medium_event_summary.csv`, `cg_acceleration_rows.csv`,
`cg_frontier_rows.csv`, and `cg_frontier_by_scale.csv` files should be
re-audited before being used for proof claims. Existing certification counts
remain usable as certification counts.
