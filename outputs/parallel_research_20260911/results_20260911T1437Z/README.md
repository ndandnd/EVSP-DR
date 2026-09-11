# 10:37 EDT results — 11 September 2026

Warm P2 k=8 matches the target with eight buses, proved optimal within its 42,732-column inherited pool. Stage 1 took 6.159 seconds. Stage 2 reached the total one-hour solver budget: charging-related objective 264.352, bound 248.152483, gap 6.128%. This is a fleet-only pool proof; charging optimality is not proved.

The fresh P2 k=8 run found nine buses with pool bound eight at its time limit. Therefore this comparison demonstrates an improved incumbent with inherited columns, not proof that the fresh pool cannot contain an eight-bus solution. Inputs and physics are retained in evidence.json.

CG used 296 iterations and 285.35 minutes, terminating with its conservative expanded-grid pricing certificate. Weighted LP objective is 800235.914468357; fractional route weight is approximately eight. These are distinct quantities. Selected integer routes passed individual physical replay and cover 173 trips at least once; 11 trips are overcovered. Duplicate removal and shared-station capacity remain unchecked. No GIRO routes were injected.

All five capacity retries remain running and have nonempty atomic pool checkpoints. The saved JSONL line counts are file records, not asserted route counts; checkpoints can include metadata. Four pools were last updated near startup while their next pricing call remains in progress. Duty13406 capacity has a later checkpoint. This validates preservation of accepted columns, not resumption of an in-flight pricing traversal or CG convergence.

Default MIP accounting remains 75 completed / zero recorded preemptions. The warm Scaglione reference has 28 completed and eight pending cases. No new execution failure was observed. The remaining work has real predecessor dependencies; held historical jobs were untouched.

Sources: dated full snapshot20260911T143701Z.json, evidence.json, checkpoint_check.json. Register rebuilt to1513 artifact/stage records across31source groups; Google Doc status and register timestamp updated in place. Slides untouched.
