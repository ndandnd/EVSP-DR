# Authoritative graph-timeout gates — v2

**18 v2 gates are waiting afterany their original graphs.** The18unused pending v1 gates were cancelled only after each affected CG's dependency replacement was verified. Original graphs and CGs were not cancelled or changed. These gates are operational safeguards, not new research cases or pricing certificates.

Authoritative remote root: `/home/nc437/ladder-lite/graph_timeout_gates_v2_20260914`. Private graph attempts live in `/share/scaglione/nc437/evsp-dr/graph_timeout_gates_v2_20260914/cases`. The immutable v1 root remains historical and superseded.

V2 tolerates transient accounting lag for up to120seconds, including query time. It checks the exact task, retries missing or stale nonterminal records, rejects ambiguous records immediately, and fails if terminal evidence has not arrived by the deadline. It never treats an active graph as timed out. Native build, no-op and publication code remain byte-identical to the validated v1 code. The original10.27GiBcache no-op and tiny same-source graph fixture are hash bound; five new lag tests and the four existing gate tests pass. A native v2 accounting lookup returned the correct completed task/raw-ID pair.

[Manifest](manifest.json) SHA256: `4c6d3578612d238156fad1cebe535bce87f1ee486da13fb5fcbda6441f480585`. Resources remain2CPU64GiB,24hour watchdog,24h30mSlurm allowance, default partition andexcludedscaglione-compute-01. Gates only rebuild for authenticated schedulerTIMEOUT or internal timeout124; valid caches cause a no-op. Original failed attempts remain intact, and replacements receive explicit timeout-recovery provenance.

[Deployment validation](deployment_validation.json) accounts for all52original solver dependency edges. There were51remaining before and after v2 replacement: the W6k25parent134079 had already completed successfully, satisfying its edge before the v2 deployment. Only18v1-gate parents were replaced; all remaining previous-k and own-CG MIP edges stayed intact. The completed parent's accounting and CG hash evidence are retained.

See the [exact job map](case_jobs.json), [append-only mutation ledger](deployment_ledger.jsonl), [v1 amendment link](v1_amendment.json), [before dependencies](before_dependencies.json), [after dependencies](after_dependencies.json), and [validation](validation.json). V2jobIDs189915–189936are noncontiguous. All18effective allocations/exclusions and18v1cancellations were verified.

Use [collect_adapter.py](collect_adapter.py) for operational metadata only; it never reads graph pickles. Register this v2root as authoritative and retain v1records as superseded history. The original extension collector continues to observe canonical graph markers and subsequent solver results. Source artifacts, original manifests and v1deployed files were preserved.
