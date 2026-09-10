# EVSP–DR working instructions

## Cluster resources

Before cluster submissions, read `/home/nc437/ladder-lite/SCAGLIONE_RESOURCE_POLICY.md` on Unicorn (local reference: `outputs/meeting_20260910/SCAGLIONE_RESOURCE_POLICY.md`). Independent default-partition CG arrays use a default concurrency of **50**, or all cases if fewer exist. Do not impose smaller arbitrary throttles. Record a concrete cluster-policy or measured resource reason for any lower limit. Preserve true previous-k data dependencies.

Exclude `scaglione-compute-01` from every CPU-only job, regardless of partition. Other Scaglione CPU nodes remain usable for MIPs. Leave held historical jobs untouched unless explicitly authorized. Notify the user promptly if Unicorn access is lost.

## Experiment records

Use `outputs/research_register/README.md` as the experiment entry point. Record source input hashes, execution commit, physics, objective, master sense, initialization, resource requests, job dependencies, output paths and hashes. Keep scheduler status, CG certificate, finite-pool MIP proof, physical validation and GIRO target attainment separate. Do not call an RMP objective a full-model lower bound without the corresponding pricing certificate. Distinguish fractional route weight from the weighted objective and from a fleet-only lower bound.

Update the experiment register and current Google Doc status when new verified results arrive; preserve source artifacts and superseded-result flags. Google Slides are not to be edited under the current standing instruction.
