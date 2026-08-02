# EVSP-DR Unicorn checkpoint and runbook

This is the entry point for a fresh GPT/Codex session on the Unicorn computer.
It describes the maintained, reproducible path for measuring the current DP
before making the charge-timing model more ambitious.

## Start here

Use the `issue20-local-pricing-audit` branch for this checkpoint. It is based on
the maintained `issue20` line. Do not merge `main` into it: `main` diverged
before the maintained DP work and contains obsolete generated files and
conflicting model/data changes.

```bash
git clone https://github.com/ndandnd/EVSP-DR.git
cd EVSP-DR
git switch issue20-local-pricing-audit
git pull --ff-only origin issue20-local-pricing-audit
git status --short --branch
```

The checkout must be clean before submitting a benchmark. Generated results,
checkpoints, Gurobi files, and logs are ignored by Git and stay under
`src/results/` and `src/logs/`.

Do not copy or commit `data/original_giro_data/`. Those local provenance files
are C1/Internal and are intentionally excluded from this public repository.

## What this checkpoint is testing

The current checkpoint includes these DP/master fixes:

- unrestricted LP column upper bounds, removing a dual-degeneracy stall mode;
- identical hour-split charging costs in the DP, RMP, resume path, and final MIP;
- station-specific prices resolved from physical names rather than copy names;
- one pricing node per physical station, while keeping the PARX charger distinct
  from the source/sink depot node;
- one-time DAG construction and reusable per-call label limits;
- O(1) stale-label rejection;
- dominance that preserves both remaining recharge capacity and the minimum
  completed-trip requirement, without pruning later station labels that retain
  a different 220-minute successor window;
- current-master incidence/cost dominance: for an already represented trip set,
  only a strictly cheaper charging/path realization is admitted;
- existing incidence patterns are filtered before the K-best cutoff, avoiding a
  false no-new-column stop when lower-ranked unseen patterns remain;
- negative depot completions are condensed online to the cheapest realization
  per trip set instead of all being retained until search ends;
- optional diversified K-column output and incidence-diverse dominance retain
  more complementary histories;
- an optional `start_fair_bound` scheduler rotates across first-trip groups
  while using the future-dual bound within each group;
- zero-charge station pass-through and SOC-safe restricted-wait dominance fix
  two demonstrated label-completeness errors;
- checkpoints that store the complete trip set, input hashes, mode, arguments,
  and Git revision;
- final-MIP audits for missing-column trips, artificial `q_i`, and overcoverage;
- one-trip columns allowed by default, so the pricer does not discard otherwise
  feasible routes merely as a speed heuristic.

This is still the **Goal-1 restricted model**. Charging begins immediately on
station arrival, and pricing retains the 57-minute trip-to-trip and 61-minute
trip-to-station restrictions. Station-to-trip waiting defaults to the full
1560-minute horizon for rediscovery. That is acceptable for a flat-price
benchmark, but immediate charging still cannot support the later headline
temporal demand-response claim.

Benchmark interpretation:

- `Practice_10bus.csv` and `Practice_15bus.csv` are the clean regression cases.
- `Practice_20bus.csv` and `Practice_43bus.csv` mix weekday variants. They may be
  used only as synthetic scaling tests, not as one-day GIRO parity evidence.
- `Practice_43bus.csv` maps to 42 literal historical `VehicleTask` values; its
  filename is not evidence of a verified 43-bus day.
- The intended full-day comparison target remains 43 buses, but the current
  tracked 987-trip file is not a verified single-day parity instance.
  Reconstructing that input remains a separate task.

## Unicorn environment

The submitted jobs default to the environment previously used on Unicorn:

```bash
source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate /home/nc437/evsp_env
```

Override `EVSP_CONDA_SH` or `EVSP_CONDA_ENV` if the cluster environment has
moved. Goal-1 column generation defaults to the free SciPy/HiGHS master and does
not require a Gurobi license. Python 3.10+, `scipy`, `pandas`, and `numpy` are
required; `gurobipy` and `GRB_LICENSE_FILE` are needed only when explicitly
using the Gurobi backend or solving the later final integer master. See
`requirements-unicorn.txt`.

From the repository root, run the preflight once on the login node:

```bash
python -u src/unicorn_preflight.py \
  --csv Practice_10bus.csv \
  --prices_csv hourly_prices_flat.csv \
  --mode MATCHING \
  --skip_gurobi
```

`hourly_prices_flat.csv` is a two-column temporal curve. The maintained loader
now replicates it across all six physical stations, giving the neutral Goal-1
price input that the older scripts lacked.

## Recommended first experiment

First submit one short environment/plumbing smoke:

```bash
bash src/submit_goal1_matrix.sh smoke
```

Inspect its log and confirm that it writes a run-local checkpoint and summary.
The hard 10/15 MATCHING seeds already equal their peak-concurrency lower bounds,
so a multi-hour MATCHING run cannot improve their fleet counts. To test the DP
itself, first generate the deterministic synthetic suite:

```bash
python -u src/generate_random_goal1_instances.py
```

For the next reportable collection, submit the three complementary policies on
all five synthetic 10-bus and all five synthetic 15-bus samples. The helper
creates 30 independent GREEDY jobs (10 instances times 3 policies), each with a
distinct reproducible run tag:

```bash
bash src/submit_goal1_portfolio_matrix.sh 5m goal1_portfolio_5m_round1
```

This is the recommended use of Unicorn now. It is a breadth gate totaling 2.5
aggregate Python active-hours; requested allocation time and CPU-hours are
higher because every job includes setup/master overhead and requests four
CPUs. It is not a claim that the production solver already runs an integrated
portfolio. After every job finishes, merge the three pool JSONs for
each instance separately with `src/audit_goal1_column_pools.py`; the audit
rejects pools from different instances. Run
`src/audit_matching_cover_pricing.py` on at least one GREEDY pool per size.

If the merged five-minute pools repeatedly improve their GREEDY seed route
weights, collect the corresponding 30-minute matrix:

```bash
bash src/submit_goal1_portfolio_matrix.sh 30m goal1_portfolio_30m_round1
```

Do not jump directly to the helper's `3h` profile. First verify that the
30-minute unions continue to move and that memory/label-cap statistics remain
acceptable. Saved pools are the experiment output; preserve the entire result
directories before pruning anything.

The matched local comparison of heap orders on identical GREEDY masters is now
complete. Random 15-r04 has peak
concurrency 14, but a verified 15-trip reachability antichain gives the stronger
fractional lower bound 15. MATCHING supplies 15 feasible routes and GREEDY
supplies 17, so the valid pricing target is 17 to 15; below 15 would indicate a
correctness problem.

No individual five-minute policy lowered route weight from 17. A union of all
eight local pools did reach 16.857142857, and the smallest improving union used
three complementary policies. Therefore do not submit a multi-hour
single-queue job. First use Unicorn only to reproduce a short smoke or to
generate distinct short pools for the union audit. The hard first-ten case is
the regression control. Nested instance names below are safe relative paths
under `data/`:

```bash
R04='random_goal1_instances/seed_20260802/Practice_SyntheticRandom_15bus_s20260802_r04.csv'

EVSP_INSTANCES="Practice_10bus.csv,${R04}" \
EVSP_MODES=GREEDY \
EVSP_QUEUE_ORDER=time \
  bash src/submit_goal1_matrix.sh smoke heap_time

EVSP_INSTANCES="Practice_10bus.csv,${R04}" \
EVSP_MODES=GREEDY \
EVSP_QUEUE_ORDER=reduced_cost \
  bash src/submit_goal1_matrix.sh smoke heap_rc

EVSP_INSTANCES="Practice_10bus.csv,${R04}" \
EVSP_MODES=GREEDY \
EVSP_QUEUE_ORDER=reduced_cost_bound \
  bash src/submit_goal1_matrix.sh smoke heap_bound

EVSP_INSTANCES="$R04" \
EVSP_MODES=GREEDY \
EVSP_QUEUE_ORDER=start_fair_bound \
EVSP_PRICING_OUTPUT_SELECTION=diversified \
EVSP_DOMINANCE_MODE=resource \
  bash src/submit_goal1_matrix.sh smoke fair_resource

EVSP_INSTANCES="$R04" \
EVSP_MODES=GREEDY \
EVSP_QUEUE_ORDER=start_fair_bound \
EVSP_PRICING_OUTPUT_SELECTION=diversified \
EVSP_DOMINANCE_MODE=incidence_diverse \
  bash src/submit_goal1_matrix.sh smoke fair_incidence
```

Compare reoptimized master outcomes, not only best reduced cost. Preserve each
whole result directory, then run `src/audit_goal1_column_pools.py` over the
saved final-pool JSON files. Run `src/audit_matching_cover_pricing.py` on one
GREEDY pool to verify that a current-model 15-route cover enters in
complementary negative waves without using historical duties. Exact local
results and interpretation are in `GOAL1_LOCAL_RESULTS_20260802.md`.

The following single-queue extension is shown only as a command template. It
is **not currently authorized by the five-minute gate**:

```bash
EVSP_INSTANCES="$R04" \
EVSP_MODES=GREEDY \
EVSP_QUEUE_ORDER=reduced_cost_bound \
  bash src/submit_goal1_matrix.sh 3h r04_bound_3h
```

The separate-job five-minute union gate above may authorize the matching
separate-job 30-minute collection. It does not remove the need to implement an
integrated bounded portfolio before calling the production CG pricer repaired
or before spending three hours on one queue.

The older broad matrix remains available as a regression/performance study,
but the current evidence does not justify launching it for pricing progress:

```bash
bash src/submit_goal1_matrix.sh 6h
```

Its default four jobs are:

- 10-bus MATCHING;
- 10-bus GREEDY;
- 15-bus MATCHING;
- 15-bus GREEDY.

MATCHING supplies a model-derived route cover constructed without GIRO's
`VehicleTask` assignments. It is the primary Goal-1 initializer because its
resource-feasible routes give the restricted master a real cover immediately.
It first retries deterministic alternate maximum matchings. If none realizes as
an exact minimum path cover, it cuts relaxed paths into the fewest contiguous
resource-feasible routes and records both counts plus
`resource_repair_mode: contiguous_split` in `Seed_Matching_Provenance`; do not
describe that repaired seed as an exact minimum cover. GREEDY is the control
initializer: it also avoids GIRO's assignment, but its sequential construction
can start with more buses than the matching cover.

NO_CHEAT remains available as a pricing-only ablation. It starts with artificial
trip variables and no real routes, so it is expected to be much harder and is
not the recommended way to judge whether the current model can match the target
fleet count:

```bash
EVSP_MODES=NO_CHEAT bash src/submit_goal1_matrix.sh smoke
```

`CHEAT` remains available only as a translation diagnostic:

```bash
EVSP_MODES=CHEAT bash src/submit_goal1_matrix.sh smoke
```

Those historical columns are reconstructed from GIRO output but are **not**
validated against the current DP's time/SOC/restricted-graph rules. A CHEAT
fleet count or zero-artificial result proves coverage mapping only; do not call
it a model-feasible benchmark or use it as parity evidence.

Each broad-matrix job runs for at most six active compute hours and, if it reaches both
thresholds, saves 3h and 6h column-pool snapshots. The Slurm request is eight
wall-clock hours because the active budget counts master plus pricing time and a
partially completed iteration needs padding. Pricing tiers are clipped at milestone boundaries, so
these are the first completed-iteration checkpoints at approximately 3h and 6h.

To run all four initialization modes for diagnosis:

```bash
EVSP_MODES=MATCHING,GREEDY,NO_CHEAT,CHEAT \
  bash src/submit_goal1_matrix.sh 6h
```

To run only a three-hour wave:

```bash
bash src/submit_goal1_matrix.sh 3h
```

The launcher prints a unique run tag. Save it. Reusing the same tag with the
same critical pricing configuration automatically resumes the matching
`ckpt_latest_*.json`; completed column pools are never deleted. Active limit and
milestone targets may be extended, but changing the pricing tiers, label cap,
minimum trips, or commit requires a fresh tag (or an explicitly unsafe resume).

## Unlimited Python-side run

An “unlimited” run disables the Python active-time stop, but Slurm still needs a
walltime. Check the current partition policy rather than guessing:

```bash
/usr/local/slurm/current/bin/scontrol show partition
```

Then supply an allowed walltime explicitly, for example:

```bash
bash src/submit_goal1_matrix.sh unlimited 2-00:00:00
```

The job will continue until restricted pricing terminates, the iteration guard
is reached, or Slurm ends the allocation. If Slurm ends it, resubmit with the
same printed run tag to resume from the last completed iteration. Milestones at
3h, 6h, 12h, and 24h are saved by default.

## Useful overrides

The launchers accept environment overrides without editing tracked files:

```bash
# One synthetic scaling job only; do not call this GIRO parity.
EVSP_INSTANCES=Practice_43bus.csv \
EVSP_MODES=MATCHING \
  bash src/submit_goal1_matrix.sh 3h provisional987

# Change the pricing escalation schedule.
EVSP_MAX_LABELS=200000 \
EVSP_PRICING_TIERS=100000:500,200000:3000 \
  bash src/submit_goal1_matrix.sh 6h

# A price scenario must regenerate columns under that price file.
EVSP_PRICE_CSV=spatiotemporal_single_peak_08.csv \
EVSP_PRICE_TAG=peak08 \
  bash src/submit_goal1_matrix.sh 6h

# Select a verified partition or memory request.
EVSP_PARTITION=PARTITION_NAME EVSP_MEMORY=64G \
  bash src/submit_goal1_matrix.sh 6h
```

Important controls:

- `EVSP_AUTO_RESUME=0`: force a fresh result directory;
- `EVSP_RESUME_CKPT=/path/to/file.json`: select a checkpoint explicitly;
- `EVSP_KBEST`: columns accepted per iteration, default 150;
- `EVSP_MAX_LABELS`: default label cap, default 100000 in cluster jobs;
- `EVSP_MIN_TRIPS_PER_ROUTE`: default 1; larger values deliberately restrict the
  pricing graph and must be reported;
- `EVSP_MASTER_TIME_LIMIT`: exact master-LP limit per iteration, default 120
  seconds. A non-optimal master stops rather than pricing invalid duals; resume
  the same tag with a larger value if this occurs;
- `EVSP_MASTER_BACKEND`: defaults to `scipy` for these column-generation jobs;
  `gurobi` remains an explicit diagnostic override;
- `EVSP_QUEUE_ORDER`: `time`, `reduced_cost`, `reduced_cost_bound`, or
  `start_fair_bound`; bound is the default, and the fair mode is experimental;
- `EVSP_PRICING_OUTPUT_SELECTION`: `reduced_cost` (default) or `diversified`;
- `EVSP_DOMINANCE_MODE`: `resource` (default) or the experimental
  `incidence_diverse`;
- `EVSP_MAX_CHARGE2TRIP`: station-to-trip wait cap in minutes, default 1560;
- `EVSP_MAX_SUCCESSOR_TARGETS`: cap on successor-boundary SOC targets, default
  64;
- `EVSP_PRICING_TIERS`: `labels:seconds` escalation list;
- `EVSP_PRICING_WALL_PER_ITER`: total pricing wall cap per iteration;
- `EVSP_STAGNATION_WINDOW` and `EVSP_IMPROVEMENT_BOUND`: the benchmark launcher
  effectively disables early stagnation by default;
- `EVSP_PRICE_TAG`: output label; by default it is derived from the price CSV;
- `EVSP_ALLOW_UNSAFE_RESUME=1`: explicitly allow mixing a checkpoint with a
  different commit or critical pricing configuration. Do not use this for a
  reported benchmark.

Auto-resume refuses a checkpoint made by another Git commit or critical pricing
configuration. The job lock also rejects two simultaneous writers for the same
instance/mode/run-tag instead of risking checkpoint corruption.

Old list-only pools can be preserved, but they lack instance, price, and code
provenance. To migrate one, set the exact original instance/mode/price, provide
`EVSP_RESUME_CKPT`, and set `EVSP_ALLOW_UNSAFE_RESUME=1`. Treat the resulting run
as a legacy diagnostic until its provenance and route feasibility are audited.

## Monitor and inspect resource use

```bash
squeue --me
sacct -S today --name='G1_*' \
  --format=JobID,JobName,State,Elapsed,Timelimit,MaxRSS,ExitCode
```

Do not infer that a run “solved” merely because the process exited successfully.
Read `termination_reason`, the pricing timeouts, and the artificial-trip counts.

## Run final MIPs from the saved pools

Column generation deliberately skips the final MIP. Submit it separately for
each desired 3h and 6h snapshot:

```bash
find src/results -type f -name 'routes_3h_snapshot_*.json' -print

sbatch --time=01:30:00 --mem=32G --cpus-per-task=8 \
  src/submit_final_mip.sub \
  /absolute/path/to/routes_3h_snapshot_NAME.json 3600
```

Add a verified `--partition=NAME` if Unicorn requires one. Repeat with
`routes_6h_snapshot_*.json`. The final-MIP script writes its `.sol`,
log, and `final_mip_summary_*.json` beside the source snapshot. It verifies the
generating Git commit and price hash by default.

Passing a different price file only to the final MIP is rejected: reselecting
old-price columns can miss newly attractive routes and is not a valid savings
experiment. `EVSP_ALLOW_RESTRICTED_POOL_REPRICE=1` exists only for an explicitly
labeled diagnostic; reported price scenarios must rerun column generation.

If restricted pricing converges before a milestone, use
`routes_colgen_final_*.json`. If Slurm interrupts a run after at least one
completed iteration, use its `ckpt_latest_*.json`. Both contain the full trip
set and provenance required by the standalone MIP. An interruption before the
first completed iteration may leave no usable checkpoint and must be restarted.

A fleet result is usable only when all of these are reported:

```text
artificial_trips_used == 0
missing_column_trips == 0
dummy_routes_used == 0
buses_used is not null
unsafe_checkpoint_override == false
restricted_pool_reprice == false
```

For a model-feasible claim, `mode` must be `MATCHING`, `NO_CHEAT`, or `GREEDY`;
CHEAT is excluded by the historical-route validation caveat above.

Also report `overcovered_trips`, LP objective, MIP objective/bound/gap, columns in
the pool, Git commit, mode, active time, and termination reason.

The per-iteration pricing CSV records the master state before newly priced
columns are added in `Artificial_Trips_Before_Add`,
`Artificial_Total_Before_Add`, and `LP_Route_Weight_Before_Add`. Use the first
row with `Artificial_Trips_Before_Add == 0` as the time-to-first-real-cover
metric. The final summary records the re-solved pool in
`Final_LP_Artificial_Trips`, `Final_LP_Artificial_Total`, and
`Final_LP_Route_Weight`. Compare MATCHING and GREEDY only at the same instance,
commit, flat price file, hardware request, pricing schedule, and active-time
budget. Report NO_CHEAT as a pricing-only ablation and CHEAT separately as an
unvalidated historical-translation diagnostic.

## Preserve and collect long-running columns

A normally completed result directory contains `ckpt_latest_*.json`, a final
route pool, and any milestones it actually reached. An interrupted run may have
only the last completed checkpoint. These JSON files contain the generated
columns. They are ignored by Git intentionally; back up complete directories.

From another computer:

```bash
RUN_TAG='paste_the_printed_run_tag_here'
LOCAL_ROOT="$HOME/Downloads/evsp_goal1_${RUN_TAG}"
REMOTE_REPO='/home/nc437/EVSP-DR'  # replace with the actual clone path
mkdir -p "$LOCAL_ROOT"

rsync -av --partial --prune-empty-dirs \
  --include='*/' \
  --include="*${RUN_TAG}*/***" \
  --exclude='*' \
  "nc437@unicorn-login-01.coecis.cornell.edu:${REMOTE_REPO}/src/results/" \
  "$LOCAL_ROOT/results/"

rsync -av --partial \
  "nc437@unicorn-login-01.coecis.cornell.edu:${REMOTE_REPO}/src/logs/G1_*" \
  "$LOCAL_ROOT/logs/"
```

Never use `git clean` as a result-collection step. Git does not contain these
long-running columns.

## Decision after the first batch

1. If MATCHING does not produce a zero-artificial feasible pool, repair the
   current-model initialization/data translation before interpreting DP speed.
2. Compare MATCHING with GREEDY to separate pricing performance from the quality
   of the real-route warm start. Use NO_CHEAT only to diagnose pricing from an
   artificial-only pool.
3. Treat CHEAT only as a coverage-mapping check unless a separate validator is
   added for every imported route's current time/SOC/restricted-graph feasibility.
4. If 10/15-bus parity is healthy, reconstruct the valid single-day full input
   and verify its historical blocks before attempting full parity.
5. Only after Goal 1 is trustworthy should delayed-start charging and relaxed
   waiting restrictions enter Goal 2.
6. Before any temporal savings claim, charging time must be a first-class
   decision and charger/solar-capacity assumptions must be stated explicitly.

`HANDOFF_ISSUE18.md` is historical context. Its old random-instance arrays and
local paths are not the clean-clone run path; use this file and the new generic
launchers instead.
