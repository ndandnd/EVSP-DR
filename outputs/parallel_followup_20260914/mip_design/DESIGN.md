# Independent saved-pool follow-up —14 prepared, not submitted

Frozen selection: `20260914T122906Z`. Campaign root on Unicorn:
`/home/nc437/ladder-lite/parallel_pool_followup_20260914`; storage root:
`/share/scaglione/nc437/evsp-dr/parallel_pool_followup_20260914`.

The **14 existing-pool MIPs are prepared and validated**, with no dependencies.
The six union constructions are deferred to compute allocations; they are not
included in the immutable14-case manifest. The first completed construction
is retained under `pools/c1_k08` after the preparation process exited255 without
a traceback; access was immediately rechecked successfully. The cause is not
established. No repeated heavy union preparation was run on the login node.

Each MIP uses the same reviewed
`871d057` MIP runner, eight CPUs, 24 GB, default Gurobi seed, covering physics,
10,800 seconds of fleet search and 12,600 seconds total. The default partition
must exclude `scaglione-compute-01`. Existing chain dependencies are unchanged.
No submissions are part of this preparation. The20-case table below is the
design, of which the first14 existing-pool cases are ready.

| Cases | Number | Question |
|---|---:|---|
| C2k25, C4k22, C6k23 original extension pools | 3 | Do the newly observed one-bus gaps resolve with additional search on the identical pool? |
| C4k19 continued/certified pool | 1 | Does this pool recover the19-bus target under the same longer fleet budget? The earlier one-hour run is unresolved. |
| c200/complementary pools for C1k8, C2k8, C4k8, C5k8, C5k10, C6k10, excluding C1k8/C5k8 complementary | 10 | Do existing treatment columns already support the target? Excluded pools already prove target absence. |
| Original+c200+complementary unions for those six inputs | 6 | Does combining already generated columns close gaps that remain in individual pools? |

The original fresh three-hour searches, completed inherited-pool searches and
27 operational repeatability allocations are not repeated. Each union uses
the same fleet-search allowance as the new individual-treatment controls.
A changed incumbent alone does not prove target absence; an optimal fleet
above target proves absence only from that finite pool.

Union construction preserves whole source route records without generating
routes or reconstructing costs. It verifies identical ordered trip IDs, input,
battery, power, reserve, SOC/time grid, tariff, master sense, and reference,
deadhead and producer hashes. Deterministic source order is original, c200,
complementary. Deduplication follows the baseline covering loader: trip-set
identity, retain strictly lower recorded cost by more than1e-9, retain first
source on ties. This identity is not valid for shared-capacity models.

All twelve constituent treatment MIPs already report zero rejected and zero
repaired columns. The unchanged native MIP gate still replays the resulting
pool before every solve and validates the selected solution afterward. The
preparer does not claim to have executed that new union gate.

Synthetic union input files are **pool constructions, not CG endpoints**:
`optimization_run=false`, `certified_rc_optimal=false`, no retained final LP.
The compatibility `final.iter` field belongs to the original constituent,
as explicitly documented in the artifact, and must not become a union CG
iteration count. Record each construction and its ordered source hashes
separately. Its source certificates remain source certificates; no new LP
bound or pricing certificate follows from writing a union file.

The exact frozen worker and native large-license gate are reused. Its existing
registry cohort string is retained; case IDs and output paths distinguish this
campaign. Unit checks cover strict identity rejection, invalid columns,
minimum-cost selection, stable ties and retention of the whole source record.
