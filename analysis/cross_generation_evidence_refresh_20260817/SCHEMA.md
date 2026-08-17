# Cross-generation normalized schema

- `legacy_master_objective` preserves historical `Master_Obj`.
- `master_objective_before_add` preserves current
  `Master_Obj_Before_Add`; the two are not silently equated.
- LP route weight is not an integer fleet.
- Zero artificials, pricing certification, physical schedule validation,
  finite-pool fleet proof and global route-space optimality are distinct.
- Master/pricing shares are populated only for schemas that measured both.
- Missing instrumentation remains null with an availability reason.
- RAW, MATCHING, and GIRO-augmented pools are separate treatments.
- Time-to-event columns include explicit right-censoring fields.
