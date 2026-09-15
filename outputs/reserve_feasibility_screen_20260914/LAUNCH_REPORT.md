# Reserve-feasibility screen launch

Frozen manifest: `8bf3b35fe4d3175a399456fadecfde08bc38de9b41c0ea15590f7522f7c4052f`. Native validation `207691` passed; validation `207021` is preserved as a superseded reporting-schema failure after its smoke commands completed.

All ten cases were admitted concurrently on the default partition. They requested one CPU, 24 GB, 4:15, no dependencies, no requeue, and excluded `scaglione-compute-01`. Every driver is restricted to one thread.

| Case | Job | Launch state |
|---|---:|---|
| `k1_13405_flat_236p44r15_baseline` | `207726` | RUNNING |
| `k1_13405_flat_236p44r15_parx60` | `207727` | RUNNING |
| `k1_13406_flat_236p44r15_baseline` | `207728` | RUNNING |
| `k1_13406_flat_236p44r15_parx60` | `207729` | RUNNING |
| `k1_13407_flat_236p44r15_baseline` | `207730` | RUNNING |
| `k1_13407_flat_236p44r15_parx60` | `207731` | RUNNING |
| `k1_13408_flat_236p44r15_baseline` | `207732` | RUNNING |
| `k1_13408_flat_236p44r15_parx60` | `207733` | RUNNING |
| `k1_13408_flat_236p44r15_capacity` | `207734` | RUNNING |
| `k1_13408_flat_236p44r15_combined` | `207735` | RUNNING |

The joint 236.44 kWh battery and 35.466 kWh reserve setting changes two parameters relative to 240/0. Results cannot identify an isolated reserve effect. Capacity timeouts are algorithmic endpoints and do not prove infeasibility. The 600-second MIP proves only its saved finite pool. The model remains constant-rate event physics and does not implement the nonlinear 18E1 curve or a 65% terminal target.

Collector contract: `evsp-dr-strict-capacity-parallel-collection-v1` via `collect.py --manifest <manifest> --campaign-root <root>`. `native_fixture_collection.json` contains validation-only CG/MIP endpoints marked `is_validation=true`.
