# GIRO known-column audit

This audit answers a narrow Goal-1 question before more long runs: at the
dual solution of a saved restricted master, are the historical GIRO duties
improving columns, and can the current DP resources actually express them?

The audit uses only tracked derived CSVs. It does not read the original GIRO
workbooks and it does not add historical duties to column generation.

## Run it

SciPy is sufficient; Gurobi is not used.

The saved pools are GitHub release assets, not tracked repository files. From a
clean clone, download and verify release `results-goal1-5h-20260802` first:

```bash
RELEASE_DIR=src/results/releases/results-goal1-5h-20260802
mkdir -p "$RELEASE_DIR/extracted"
gh release download results-goal1-5h-20260802 \
  --repo ndandnd/EVSP-DR \
  --pattern 'evsp_goal1_5h_20260802.tar.gz*' \
  --dir "$RELEASE_DIR"
(cd "$RELEASE_DIR" && shasum -a 256 -c evsp_goal1_5h_20260802.tar.gz.sha256)
tar -xzf "$RELEASE_DIR/evsp_goal1_5h_20260802.tar.gz" \
  -C "$RELEASE_DIR/extracted"
```

The expected archive SHA-256 is
`0129ce6c1c4f4efd94f7af04e8c67a28d2c39cc02f379bac681b8b3c94e380df`.
The public release page is
<https://github.com/ndandnd/EVSP-DR/releases/tag/results-goal1-5h-20260802>.

```bash
POOL=$(find src/results/releases/results-goal1-5h-20260802/extracted/evsp_goal1_5h \
  -path '*Practice_10bus_GREEDY*' -name 'routes_colgen_final_*.json' -print -quit)

python src/audit_giro_known_columns.py \
  --pool "$POOL" \
  --output /tmp/evsp_giro_10bus_audit.json
```

Replace `Practice_10bus_GREEDY` with `Practice_15bus_GREEDY` for the 15-bus
pool. The script verifies the saved instance and price SHA-256 values before
solving the restricted master with SciPy/HiGHS.

## Released 5-hour pool findings

| Instance | Saved LP route weight | Release-baseline duties feasible | Duties feasible under repaired runner | Sum of historical-import reduced costs |
|---|---:|---:|---:|---:|
| Practice 10 | 13 | 7/10 | 10/10 | -299,978.948 |
| Practice 15 | 21 | 10/15 | 15/15 | -600,073.237 |

The release-baseline column is the old fixed 30-kWh SOC grid plus 220-minute
station-to-trip limit. The repaired-runner column uses the production action
set: full-horizon station waiting, successor-boundary SOC targets capped at 64,
and the same immediate-start charging rule as pricing. Thus the known duties are
no longer excluded by the current SOC/wait representation.

The release did not save the exact Gurobi dual vectors used during pricing.
The audit therefore re-solves each saved master and obtains one HiGHS dual
solution. Individual duty reduced costs are dual-degeneracy dependent. For the
10-bus pool, the current-model realization of duty `13301` is feasible, costs
100,051.811, and has reduced cost -800,601.661 at the reconstructed dual. The
saved DP pool contains no route with the same 36-trip set; its best DP overlap
is only 7 of those 36 trips. This is direct evidence of a pricing-search
failure, not a feasibility excuse for that duty.

Adding the seven release-baseline-feasible known duties to the saved 10-bus pool does
not by itself lower the LP: none can replace an entire combination of the 13
GREEDY columns. This is why a negative reduced cost against one degenerate dual
can enter at a zero step while another optimal dual takes its place. Track LP
improvement and positive post-reoptimization weight, not merely the number of
negative columns returned.

## Exact release-baseline feasibility blockers

`13303` fails at local trips `134 -> 135`. They are consecutive at 23:27, but
the best current-grid label leaves only 5.868 kWh after trip 134 while trip 135
requires 10.279 kWh: a 4.411 kWh shortfall. The 30-kWh target-SOC grid caused
the loss. A successor-aware counterfactual repairs the complete duty by also
trying the maximum SOC reachable at each next-trip departure deadline. One
repaired realization includes non-grid targets 236.1329999 kWh between trips
`89 -> 90` and 219.2330012 kWh between `93 -> 94`. No single non-grid target
was sufficient in the fixed-order enumeration; at least two were needed.

There is an important implementation caveat: `_generate_charge_options`
currently receives the 26-hour horizon as its departure deadline. Adding
`max_reachable_soc_at_departure_deadline` with that unchanged argument merely
returns 300 kWh, which is already in the grid, and does **not** fix `13303`.
The deadline must be successor-specific (next trip start minus station travel),
or the station expansion must otherwise include successor-boundary targets.

Four split-shift duties failed for a different reason. Waiting at a station
fixes all four; a finer SOC target does not.

| Duty | Trip gap | PARX arrival | Latest immediate full-charge end | Earliest departure allowed by 220-minute limit |
|---|---:|---:|---:|---:|
| 13304 | 396 min | 09:10 | 09:54.54 | 11:55 |
| 13307 | 393 min | 08:51 | 09:21.84 | 11:27 |
| 13311 | 374 min | 08:54 | 09:24.19 | 11:17 |
| 13314 | 277 min | 08:37 | 09:07.02 | 09:17 |

The release DP charged immediately, reached its selected SOC too early, could
not wait, and then rejected the next trip because its station-to-trip wait
exceeded 220 minutes. The maintained Goal-1 runner now permits the full-horizon
station-to-trip wait and represents all 10/10 and 15/15 known duties. It still
cannot delay the *start* of charging, so this repair is sufficient for flat-price
rediscovery but not for the later temporal demand-response experiment.

## Search-order smoke results

At the saved 10-bus HiGHS dual, with 5,000 labels per node and a one-second
budget:

| Queue order | Negative routes returned | Best reduced cost | New route sets |
|---|---:|---:|---:|
| `(time, reduced_cost)` | 0 | none | 0 |
| `(reduced_cost, time)` | 150 | -600,504.220 | 150 |

The best route's reduced cost was independently recomputed as
`100,000 - 700,504.2197727703 = -600,504.2197727703`, exactly matching the DP.
Reoptimizing after adding all 150 routes still left route weight 13 and gave
them zero weight because of the dual-degeneracy behavior described above.
Reduced-cost-first is therefore a necessary search correction on that saved
GREEDY dual, but it also needs depth/diversity controls so short fixed-cost
subsets do not crowd out complete replacement duties.

In a matched 20-second test, a depth tie-break increased the longest returned
route from 18 to 21 trips and the best overlap with duty 13301 from 7/36 to
10/36. It still did not find the 36-trip duty, did not reach its known reduced
cost of -800,601.661, and did not lower the reoptimized master objective.

The maintained runner also repeated all three supported heaps for one second on
the same repaired 10-route MATCHING master (5,000 labels per node):

| Queue order | Accepted columns | Best reduced cost | Longest route | Mean trips/route |
|---|---:|---:|---:|---:|
| `time` | 88 | -100,112.036 | 4 | 2.47 |
| `reduced_cost` | 150 | -100,110.639 | 9 | 5.24 |
| `reduced_cost_bound` | 150 | -100,112.036 | 6 | 4.81 |

None changed the already-certified 10-route LP in that one-second window. The
heap effect is therefore drastic but dual-dependent: bound priority was not
uniformly deeper than raw reduced-cost priority on this master. A useful next
pricing experiment is a diversified batch that returns columns from both
priorities, not another single-priority multi-hour run.
