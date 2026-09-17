# Hourly check at 17 September 01:45 UTC

SSH and all seven campaign collectors succeeded. Four-result gate remains **35/37 endpoints**, two pending, no integrity errors; the requested briefing is not ready.

**Newly classified failure:** job341179, C4k15 fixed-duty noon tariff, ended OUT_OF_MEMORY after2h59m22s with32G requested and33,404,312KiB batch MaxRSS. Native stderr confirms the OOM kill. This was already present in the previous scheduler snapshot but had not been reported; it is not described as having just failed. C5's equivalent job341187 was already reported. Both are resource failures, not infeasibility; neither was retried.

Six action3 arms have no finalCG/MIP endpoints. Completed replay shards per125: control7, PARX60 9, reserve15 9, battery236.44 7, battery239.01 6, segregation7. All14,336 completed control sequences are feasible; no anomaly/error was collected. Predictions remain pending. Partial replay is not a fleet conclusion.

Full40CG343119 remains queued on its existing graph341404_0 prerequisite. Its12h/120G Scaglione settings are unchanged; MIP341406 stays held. SupersededCG341405 cancellation is expected, not a failure.

One additional intermediate random-trip endpoint arrived: r1_s12 MIP17 buses, native finite-pool bound14, unproved; individual-route replay true, duplicate cleanup false. Stage12 is not a12-bus target. This remains source-reported partial experiment evidence, not a promoted headline.

No submissions, retries, holds/releases or changes to scientific settings. Frozen Doc counts and source tabs unchanged. New notification receipt: monitor/notified_failures.json.
