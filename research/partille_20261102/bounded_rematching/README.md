# Rematching the three unsuccessful eight-bus cases

C1, C4 and C5 each had an eight-path timetable cover that included a duty that could not finish even with exclusive charger access. The follow-up asks whether another choice of connections can fix those same three misses. This is exploratory work on known failures. Success would justify a separate test on new trip sets.

The algorithm selects eight paths using the connections observed in the original root columns. When a complete duty fails the fixed-sequence charging search even with all chargers free, the matching model rules out that exact duty. Other paths may still use its connections and an extension of the rejected sequence is allowed. If every duty can finish alone, the algorithm tries to place their charging sessions together. A conflict at this step only rules out retrying that candidate in this search. Every reported success requires independent replay of the complete dispatch.

The frozen budget is 180 seconds and at most 32 candidate covers per chain, including reading the archived root and testing the charging schedules. Historical graph and CG computation is recorded separately. All three archived roots lacked a pricing certificate. No GIRO duty or known dispatch is used as input.

The timestamped [freeze](FREEZE.json), [protocol](protocol.json) and [source fingerprints](source_index.json) bind the implementation and the three private input manifests. Authorized reproduction requires the original restricted records and manifests under the project campaign. The unit check exhaustively tests the matching exclusion on 225 small cases and the solver smoke test verifies that it still permits extensions.
