# Terminal diagnostic review

Job **982348** ended with `diagnostic_error`: child return code **−11 (SIGSEGV)** after **366.18 s**, not the 900-second watchdog. Slurm recorded FAILED/1:0 for the wrapper after 6m10s. Stderr was empty. All eight artifacts supplied by the manager were authenticated against their embedded SHA-256 values.

## What the profile establishes

The graph was advancing. At313.47s it had packed3,011,382arcs and finalized141trip/SOC states plus the depot, out of29,396trip/SOC states. It retained178,765window-cache entries and reported283,088KiB maximum RSS (about276MiB). Source density varies, so this is not a defensible linear completion-time forecast. The observed run is neither an initialization hang nor evidence of memory exhaustion.

Both complete120-second stack samples place the main thread in JSON encoding from `_add`, called by `_charge_arcs`. The third sample, around360seconds, is truncated inside JSON encoding; the process died shortly afterward. Two samples identify a candidate hotspot, not a measured percentage of total CPU time.

Static source inspection strengthens that lead: `_add` eagerly computes `json.dumps(action, sort_keys=True)` for every candidate before checking whether its `(target, dual)` entry exists or whether a strict cost comparison already decides which arc survives. `_finalize_source` also constructs JSON sort keys. This is graph construction work, before any CG pricing iteration or master LP. The tariff/window-prefix changes do not target this path.

## Instrumentation failure must stay separate

The original run timed out; this diagnostic segfaulted. They are different failures. The timing and truncated traceback make the asynchronous stack sampler suspect, but do not prove it caused the cluster SIGSEGV. No core dump or native debugger trace was collected.

A standalone local probe containing only JSON encoding and the optional faulthandler timer reproduced a related instrumentation failure: the control completed1,247,831encodings in3seconds; enabling a2ms periodic timer caused the same3-second loop to remain unfinished at the10-second external timeout, with a truncated stack log. This is a solver-free timer-associated hang on macOS Python3.12.2. It is **not** a reproduction of the cluster SIGSEGV on Python3.12.13, nor a representative performance measurement. The earlier isolated5-second timer probe also timed out; the retained reproducible probe and result are sampler_probe.py and sampler_isolation_local.json.

## Bounded next recommendation

1. Do not reuse the asynchronous timer sampler. Preserve the failed diagnostic as instrumentation/error evidence.
2. Prepare a synchronous alternative: ordinary progress counters and bounded cProfile collection/checkpointing at completed source boundaries, without background stack sampling. Verify that instrumented and reference packed graphs, action recipes, realized route costs and pricing results agree on small real inputs. Any profiling timings are diagnostic, not solver speedup measurements.
3. Only after those checks, consider another bounded profile with a new instrumentation pin and output directory. No automatic retry was submitted by this review.
4. The first code-efficiency candidate is to defer JSON tie-key creation until equal-cost alternatives actually need comparison. Preserve the existing exact cost/JSON lexicographic tie order and check action immutability. Test adversarial ties, cheaper/dearer alternatives, arc/action identity and whole-route reduced costs before any production adoption. Do not yet claim a speedup from this idea.

Keep the original14400-second construction-inclusive budget and censored result. Any extended cache+CG run must be a separately labeled budget extension. The blocked historical MIP remains untouched; no long retry or MIP was launched.
