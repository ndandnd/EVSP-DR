# w2_k14 import timeout investigation

**Finding: the900-second inherited replay limit is not a hard bound on preparation, and forked signal handlers can prevent timeout cleanup from returning.** No cluster retry or solver edit was performed.

Job949703 loaded its279,321,120-arc cache in46.37seconds, published an initial status and fsynced the iteration CSV header. It then reached a4h47 scheduler timeout with no imported columns, no importer audit and no CG iterations. The recorded96GiB allocation did not report OOM. Direct singleton initialization occurs after inherited import in this source; it is therefore not the leading missing phase.

## Exact timer scope

In sourcea299, inherited_event_pool_records hashes and reads the entire parent journal, validates/deduplicates its pool, maps CSV identifiers, creates candidate sequences and sorts/selects the bounded512 before starting the900-second monotonic deadline. The deadline includes elapsed worker-pool creation in arithmetic but does not interrupt that constructor. It bounds waits for completed replay results. `terminate`, `close` and `join` have no hard cleanup deadline. Completed accepted records reach the caller/journal only after this cleanup returns. The global14400-second CG limit also cannot interrupt this synchronous preparation path.

The parent w2_k13 status hash matches the child declaration. It is certified with29,008columns and a103,937,837-byte journal. This size does not itself explain hours of delay, but no stage-by-stage preparation timings were captured, so I/O or other startup problems cannot be completely excluded. The supplement includes truncated cg.json text: its displayed substring cannot be authenticated against the full-file hash. Other complete-text checks and that limitation are explicit in analysis.json.

## Reproduced shutdown mechanism

`run_cg` installs SIGTERM/SIGINT/SIGUSR1 handlers that set a termination flag. A fork-based Pool inherits them; its replay function neither resets SIGTERM nor checks the flag. When the result deadline expires, `pool.terminate()` sends SIGTERM to those workers and performs termination/join work. The inherited handler can leave workers alive, so cleanup can block beyond the advertised limit.

A minimal multiprocessing probe reproduced this. More importantly, production_import_probe.py invokes the unchanged actual inherited_event_pool_records on a small valid synthetic parent journal and a deliberately slow network callback:

| Worker signal behavior | Replay limit | Outcome |
|---|---:|---|
| Default SIGTERM |0.2s| Returns in0.208s with import_deadline_reached=true |
| Inherited solver-style handler |0.2s| Does not return by4s; isolated process group killed by test watchdog |
| Worker initializer resets signals |0.2s| Returns in0.207s with import_deadline_reached=true |

All test child processes were terminated by their containing process-group watchdog when necessary. These tests establish the code defect, not the exact program counter of the historical timed-out job. Without a saved stack, unbounded worker cleanup remains a strong explanation rather than a proven historical trace.

## Bounded next work

Before a retry, add a worker initializer that restores effective SIGTERM termination, plus a finite graceful-shutdown period and SIGKILL escalation/reaping for workers that still fail to exit. Keep the parent checkpoint handler intact. Regression tests must run the actual importer under the same parent signal handlers as run_cg, including a stuck replay, zero completed records and some already validated completed records. Direct importer tests under default signals miss this defect.

Add synchronous phase records before/after journal hashing, parsing, selection, pool startup, replay and shutdown. Preserve the current512-route selection and900-second replay treatment for comparability; if imposing a new full-import budget, label that policy separately rather than silently changing which columns get replayed. A separate startup watchdog should preserve already validated completed columns and explicit failure state.

No change to physics, tariff, fleet objective or claimed bound follows from this diagnosis. The original timeout and source artifacts remain preserved. Do not blindly requeue the same worker setup.
