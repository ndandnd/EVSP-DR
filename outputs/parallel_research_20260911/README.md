# Parallel research expansion

## Submitted

| Campaign | Cases | CG execution | Purpose |
|---|---:|---|---|
| Fresh set covering | 75 | 810454, array 0–74, concurrency 50 | Complete six chains k=2–15; reuse existing nine results |
| Inherited columns P1/P2/P4/P6 | 36 | Four independent k=2–10 pipelines | Complete warm-start evidence beyond P3/P5 |

The fresh campaign holds the original CG implementation, inputs, physics, tariff, pricing selection and eight-hour budget fixed; changes partitioning to covering. Prioritized k=5,8,10 on P2/P4/P6 first. Source input hashes and cached-network identities are checked before each case. See [submission and exact plan](cover75/submission.json).

Fresh jobs request 1 CPU and 32 GiB each. The prior largest observed k=15 CG peak was approximately 11 GiB. All Scaglione nodes are excluded from this fresh CG array to reserve their RAM for MIPs; compute-01 remains excluded from all CPU-only jobs. Default capacity observed before launch exceeded 7,000 idle CPUs. This is shared capacity, not an entitlement to fill every node.

Scaglione CPU nodes had approximately 13–30 GiB unallocated RAM despite many idle CPUs. Measured MIP peak RSS: P5 k5 0.46 GiB, P5 k6 1.58 GiB, P3 k10 4.11 GiB. Requesting 16 GiB for comparable small MIPs provides substantial headroom and enables admission; historical held jobs remain untouched.

## Reporting

New campaign roots are in the remote evidence collector. Preserve all raw statuses, logs, inputs and output hashes. Scheduler completion, CG pricing certificate, finite-pool MIP proof, physical replay and target attainment remain separate. A queued job has no numerical result. Downstream MIPs depend on their own case, not all CG cases finishing.

## Decisions from the new evidence

1. Paired covering versus partitioning across all six chains, with exact RMP objectives and stop reasons.
2. Fresh versus inherited columns on the same chain and k, including import time.
3. Fleet and charging results after validated two-stage MIP; separate time-limited gaps.

See [gap matrix](GAP_MATRIX.md) for existing controls and figure opportunities.
