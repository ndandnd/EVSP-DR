# Six single-factor timing pilots — completed

**F4: verified endpoint hashes and native physical-validation flags; full-pool and fleet conclusions remain unresolved.** The same20 deterministic trip sequences were tested in every arm.

| Arm | Feasible sequences | Other outcomes | Total seconds |
|---|---:|---|---:|
|baseline|20/20|None|118.2|
|parx60_only|20/20|None|49.6|
|reserve15_only|19/20|1 infeasible_in_fixed_sequence_event_model|44.3|
|battery236p44_only|20/20|None|51.6|
|battery239p01_only|20/20|None|51.6|
|segregation_only|17/20|3 structurally_excluded_mixed_groups|42.9|

No timeouts or unknown outcomes occurred. The reserve-only no-path sequence contains55 trips; this is infeasibility of that fixed sequence in this event representation, not infeasibility of the instance. Group separation excludes three mixed-group sequences by definition.

Control time includes70.2seconds of one-time pool extraction; later arms reuse it. Charging replay itself took about39–47seconds per20-sequence arm. This is a timing sample, not a statistically representative estimate of survival across254,068sequences. No full replay, CG or MIP was launched.

Every result and outcome hash was checked against native files; feasible outcomes carry physical replay and fixed-sequence event-model charging-optimality flags. [Endpoint audit](endpoint_audit.json). This check did not independently re-solve120pricing problems.
