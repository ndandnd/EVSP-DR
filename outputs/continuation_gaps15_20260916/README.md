# First longer MIP for chain 1 at target 32

The original one-hour saved-pool search found 35 buses with a fleet bound of 32. This independent follow-up asks whether the unchanged 279,371-column pool can match 32 with a longer fleet search. All non-time settings, ordered column hashes, native execution commit 871d057e1067411f09581e37d78f7c1ca43f68bb and greedy initialization policy are verified by the immutable manifest and preflight. Hardware/search timing may differ. No additional columns or GIRO seeds are supplied.

Budget: 12,600 total solver seconds, 10,800 fleet-stage seconds; second stage uses a fleet upper bound. Resources: 8 CPUs, 24 GB, 4.5-hour allocation, default partition, private requeue attempts, exclude scaglione-compute-01. No dependency because the canonical parent result is complete and hash-verified. Duplicate-submission preflight passed.

Job 336492 was accepted despite sbatch reporting a socket timeout. We recovered its receipt from squeue/scontrol, checking campaign output path, command, name and resources; no second submission was made. Original uncertain intent and recovery evidence are preserved. SSH remained available. This campaign was submitted after collection 20260916T194843Z started; it will appear in the next collection.

This tests finite-pool MIP search only. It does not establish full-model optimality, duplicate-free feasibility or shared charger capacity.
