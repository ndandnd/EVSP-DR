# Queue audit

The single fresh queue sample was taken **14 September 2026 at17:46:29UTC**. The directory name refers to the parent publication snapshot; it is not this audit's sampling time.

**61EVSP jobs are running,67 are waiting on real dependencies.** Running work comprises21seed-content CGs,10of their MIPs,6capacity-boundary jobs,18graph builds,4extension CGs,1extension MIP and1earlier diagnostic MIP. Running allocations request340CPUs and3984GiB memory; this is allocated/requested memory, not measured usage. The historical held array537227 is excluded. No other project jobs were present.

All83visible afterok edges point to parents still present in the queue. There are no absent unfulfilled parents and no DependencyNeverSatisfied reasons. The seed campaign has already published15CG and5MIP endpoints; its reduction from36running CGs is progress, not lost admission. No failed seed attempt was found in this filesystem read. Do not duplicate the80newly launched jobs.

All18large graphs are advancing, with progress files younger than56seconds. They have completed27.0–42.7% of source nodes after about5.1hours. Naive linear extrapolations give11.9–18.8hours total, compared with a12hour watchdog. These extrapolations are warnings, not runtime predictions: source-node workloads vary. The graph deadline deserves attention; cancelling or rebuilding now would discard active work. This audit made no changes.

The next useful independent addition would be **three matched fresh4hour CG controls: C1k15,C2k15,C4k15**, using the same frozen inputs, existing caches, code and settings as the current paired seed experiment, followed by the same12600/10800second MIP treatment. Their existing fresh certificates arrived at17443,15582,14790seconds; those endpoints cannot serve as four-hour controls. The other15matched inputs already have certified fresh endpoints before14400seconds, so rerunning all18would mostly duplicate known endpoints. First check whether immutable240minute pools already exist; reuse them if they do. This is a targeted budget-control question, not justification for another broad fill-the-queue batch. No jobs were submitted or cancelled.

Evidence: [queue response](squeue.txt), [queue metadata](queue.json), [parsed allocations/dependencies/graph progress](status.json), [graph filesystem evidence](filesystem.json), [published seed endpoint inventory](seed_endpoints.json).
