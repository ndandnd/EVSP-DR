# Separate longer searches for the C1 and C5 target-27 pools

Prepared 15 September 2026. This tests whether each existing pool already contains a target-sized fleet. Original C1 k27 found28 with bound27; C5 k27 found29 with bound26. Both gaps remain open. No new column generation or GIRO routes are added.

The existing native-validated worker, submission script and native871d057 solver are unchanged. Only case selection and the immutable campaign directory differ. The duplicate audit searches existing longer-MIP manifests by input and source-journal hash before allowing a case. Inputs, physics, source hashes, ordered original pools, original Gurobi parameters, native execution and physical replay are checked in the manifest.

Each new search uses12600seconds total and10800seconds maximum fleet stage, then charging cost with fleet at most the validated incumbent. Baseline covering;240kWh/240kW,100000bus coefficient,electricity and5per charging start; no reserve,shared capacity or ending-SOC floor. Native greedy policy and default Seed0 remain fixed. Hardware and elapsed search paths can differ; this is not an isolated causal effect of more time.

Default partition;8CPUs,24GB;exclude scaglione-compute-01. Both are independent because their source pools are finished. Requeues use unique attempt directories; every attempt is registered in the existing MIP preemption study. Original results are retained. These searches create no new CG or full-model optimality certificate. No historical held jobs or EVSPV2Gwork are changed.

Status: submitted 15 September at 08:18 EDT. Jobs **227897 (C1 k27)** and **227898 (C5 k27)** were both RUNNING at 08:24 EDT, on snavely-cpu-11 and snavely-cpu-12. Both native full-size license checks passed; two unique attempts are registered. See jobs.json, scheduler_verification.json and launch_status.json.

The manifest SHA256 is `7f091c0cb5779242b448ad3e517884752fffd4948a417902aa31f5e867cd50f1`. Preparation validation passed both original pools. A separate launch_collection.json verifies two registered cases and no completed MIPs; the shared collector and normalizer recognize the campaign. The main full collection began before this new root was added, so these launch records retain their separate timestamps.
