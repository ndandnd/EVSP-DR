# Eight-bus schedules on three previously missed chains

Changing the connections between passenger trips and recalculating charging recovered eight buses on C1, C4 and C5. The original finite pools had already been proved unable to serve these trips exactly once in eight whole duties. Eight simultaneous passenger trips require at least eight buses on each chain. The replayed schedules therefore attain the minimum fleet under the tested event model.

| Chain | Passenger trips, each served once | Candidate sets of eight duties tried | Exact duties excluded after failing alone | Candidate sets skipped after one shared charging attempt | Time on the saved root | Earlier graph + CG time on the selected attempt |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 194 | 7 | 6 | 0 | 31.73 s | 5.05 h |
| C4 | 188 | 2 | 1 | 0 | 20.25 s | 4.86 h |
| C5 | 167 | 5 | 0 | 4 | 45.61 s | 3.99 h |

Every one of the 525 consecutive trip connections in the three successful schedules already occurred in the original columns (186 on C1, 180 on C4 and 159 on C5). None of the 24 complete winning passenger sequences occurred in those same columns. The repair could assemble trips into duties the whole-duty MIP could never select. Link availability alone does not guarantee feasible charging. [Membership counts](membership.json) · [Reproducible checker](check_membership.py)

The counts describe two different failures. On C1 and C4, an entire sequence of trips could not finish on the fixed-sequence charging graph even with all chargers free. The matching model ruled out that specific complete duty while leaving its individual connections available. On C5, every tested duty could finish alone. The algorithm moved on from four combinations after a single attempt to place their charging sessions together. Another charging order could still make those combinations feasible.

The cold graph and column-generation time above is upstream of the rematch. C4 also had an earlier interrupted run with 3.00 hours of recorded CG. None of these three root runs certified its weighted LP optimum. The eight-bus minimum instead follows from passenger overlap and independent replay of an eight-bus schedule. Charging costs were evaluated for feasibility and reporting. The study did not optimize cost at eight buses.

These three chains were selected because all three missed on the original single attempt. The prospective single-attempt result stays 2/5. This exploratory follow-up recovered the remaining three cases with a different matching objective and multiple attempts, without access to GIRO assignments. The comparison cannot isolate which individual change is responsible. The next test is on previously unused eight-bus cases with the matching rule and time budget frozen in advance. The distinct fresh-pricing campaign keeps its own registered gates.

Each bus started full at 240 kWh, with zero reserve and no positive terminal floor. PARX supplied 60 kW with unlimited depot plugs. Opportunity chargers supplied 240 kW with minute-level plug limits. Costs use the historical flat prices and a five-unit fee per charging start.

[Independent physical replay, input hashes and timings](aggregate.json) · [Replay and registration audit](audit_winners.py) · [Original five-chain comparison](../root_link_holdout/RESULTS.md) · [Frozen follow-up rule](protocol.json)
