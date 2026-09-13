# Research register data dictionary

The register is generated from a compact `collect_remote.py` snapshot. The raw snapshot is the source of truth for this build. A field is `null` in JSON and `unknown` in CSV/Markdown when the snapshot does not retain it. Values are never reconstructed from filenames when a direct field is required for a scientific claim; filename parsing is limited to indexing fields such as target `k` and chain number.

Rebuild with:

```bash
python3 build_register.py --snapshot /absolute/path/to/collector.json \
  --out-dir /absolute/path/to/outputs/research_register \
  --reports-root /absolute/path/to/outputs
```

Use repeatable `--supplement /absolute/path/to/audit.json` arguments to preserve and index standalone debugging audits supplied after the collector snapshot.

## Files

| File | Purpose |
|---|---|
| `raw/<timestamp>.json` | Byte-identical collector snapshot. |
| `register.json` | Canonical normalized register. Each row includes normalized fields, `unknown_fields`, and the complete compact source `details` for that artifact or stage. |
| `register.csv` | Flat form of the same rows. Lists and dictionaries are canonical JSON strings. |
| `history.csv` | Workbook-friendly normalized rows without the large `details` payload. |
| `campaign_index.csv` | One row per campaign, with remote root, family, row count, workflow job IDs, and matching local report paths. |
| `campaigns.csv`, `campaigns.json` | Equivalent campaign-index forms for workbook and programmatic consumers. |
| `REGISTER.md` | Human-readable campaign index and coverage counts. |
| `RESULTS.md` | Human-readable table of every normalized artifact/stage row. |
| `validation.json` | Row counts and reproducibility/schema checks. |

## Stable identity and provenance

| Field | Meaning |
|---|---|
| `row_id` | SHA-256 of campaign, case, family, stage, substage, arm, source path, and source hash. Distinct substages and arms inside one result artifact remain distinct rows. |
| `campaign_id` / `campaign_root` | Stable source-group/campaign name and recorded remote root. Groups include scheduler and log-only collections and must not be counted as independent experiments. |
| `case_id` | Artifact case or cell name. It is not assumed to be globally unique without `campaign_id`. |
| `result_family` / `stage` / `substage` / `arm` | Schema family and the actual artifact, solver stage, or comparison arm represented by the row. Small-reference result files are expanded into separate CG-arm, LP, and MIP rows; tariff comparisons are expanded into original-GIRO, fixed-duty, and joint rows. |
| `artifact_status` | Result, telemetry, workflow record, diagnostic log, rejected physical output, failed pre-optimization attempt, or queue state. |
| `authority_role` | Whether the row is current, historical, authoritative recovery, superseded duplicate, rejected observation, or retained failed attempt. |
| `source_path` | Exact path recorded by the collector, or a clearly labeled snapshot locator for scheduler rows. |
| `source_sha256` | SHA-256 of the original artifact only when the collector retained it. It stays unknown for CG status, telemetry, logs, and workflow files whose source hash was not collected. |
| `snapshot_payload_sha256` | Canonical hash of the compact payload stored in the source snapshot. This is never presented as the source-file hash. |
| `snapshot_time_utc` | Collector timestamp shared by all rows in a build. |
| `completion_marker_matches` | Whether the recorded completion marker hashes the corresponding artifact. |

## Inputs, configuration, and code

| Field group | Meaning |
|---|---|
| `input_path`, `input_sha256`, `trip_count`, `target_k`, `chain`, `replication` | Instance identity and indexed case dimensions. |
| `tariff_path`, `tariff_sha256`, `prices_sha256` | Tariff path and stored tariff/price hashes. Separate fields preserve differences among schemas. |
| `reference_sha256`, `deadhead_sha256`, `master_sha256` | Reference, deadhead, and source-master identities when present. |
| `code_commit`, `code_branch`, `code_dirty` | Recorded execution code identity. Unknown means the compact artifact did not retain it. |
| `solver_backend`, `solver_version` | Recorded optimization backend/version. |
| `battery_kwh`, `initial_soc_kwh`, `reserve_kwh`, `charge_kw`, `parx_kw`, `non_parx_kw`, `soc_step_kwh`, `block_minutes` | Vehicle, charging, and event-grid physics. |
| `capacity_enforced`, `charger_counts_json` | Whether shared station capacity is in the represented model and the documented inventory used. |
| `terminal_energy_policy` | Recorded end-energy constraint or policy. No terminal percentage is inferred. |
| `master_sense` | Covering (`>=1`) or partitioning (`=1`) when retained. |
| `initialization`, `column_pool_treatment` | RAW, GIRO-seeded/augmented, greedy, singleton, saved-pool, or other source labels. Unknown values are not guessed. |

## Column generation and LP fields

| Field | Meaning |
|---|---|
| `pool_size`, `cg_iterations`, `stop_reason`, `runtime_s` | Stored pool/iteration/termination/timing fields. Runtime does not automatically include input import or network construction unless the source schema says it does. |
| `phase_runtime_json` | Aggregated phase-telemetry durations. It is a sum of recorded phase calls and can exceed wall time because some phases are nested. |
| `recorded_lp_objective`, `lp_objective_kind` | Exact source objective and its supported interpretation: `bus_plus_charging`, `fleet_only`, or unknown. |
| `weighted_lp_objective` | Compatibility field populated only for a recorded bus-plus-charging objective. It stays unknown for fleet-only small-reference LPs. It is never treated as a fleet count. |
| `fractional_fleet` | Stored sum of LP route weights. The builder never computes this by dividing the weighted objective by 100,000. |
| `artificial_total`, `min_reduced_cost` | Stored artificial mass and terminal minimum reduced cost. Numerical values near zero retain their sign. |
| `full_model_lp_certified` | True only when the source explicitly records a valid reduced-cost certificate within that source graph and approximation. It does not upgrade an event grid, omitted capacity physics, or other source approximation into a continuous/full-physical-model certificate. Workbook labels should say “Pricing certified (source scope).” |
| `lp_bound_scope` | Exact scope reported by the source. Capacity-dual-omitting or shared finite-pool comparisons remain restricted-pool results. |
| `fleet_lower_bound`, `fleet_lower_bound_scope` | Separately recorded fleet bound and its scope. A simultaneous half-open trip-overlap lower bound, restricted-pool LP bound, and restricted-pool MIP bound are different claims. |

## MIP and two-stage fields

| Field | Meaning |
|---|---|
| `mip_incumbent_fleet`, `mip_bound_fleet`, `mip_status`, `mip_gap` | Stored integer incumbent and solver bound/status/gap. A bound from a saved column pool is labeled restricted-pool unless an independent full-model argument exists. |
| `fleet_proven`, `optimal_scope` | Source proof flag and stated optimization scope. |
| `stage1_incumbent_fleet`, `stage1_bound`, `stage1_gap`, `stage1_proven` | Fleet-minimization stage. |
| `stage2_executed`, `stage2_status`, `stage2_charging_cost`, `stage2_charging_bound`, `stage2_gap` | Charging-cost stage under the recorded fleet cap. These fields remain unknown if the source is a one-stage or failed run. |
| `charging_cost_grid`, `charging_cost_continuous` | Conservative event-grid objective and continuous replay outcome, kept separate. |
| `charging_cost_exact`, `charging_cost_lower`, `charging_cost_upper` | Exact or interval-valued cost fields, primarily for the original repriced GIRO arm where within-window charging power may be unobserved. |
| `comparator_eligible`, `beats_original_robustly` | Source eligibility and robust-comparison flags. The robust test uses the original interval lower endpoint where defined. |
| `terminal_energy_grid_kwh`, `terminal_energy_continuous_kwh` | Aggregate terminal energy under grid and continuous replay semantics. |

## Physical and proof fields

| Field | Meaning |
|---|---|
| `physical_pool_validated` | Whether all accepted pool columns passed the source physical gate. |
| `physical_selected_validated` | Whether the selected incumbent or witness was physically replayed/validated. A failed pre-optimization run has no validated selected solution. |
| `physical_validation_scope` | Exact validation scope or failure reason. |
| `overcovered_trips` | Passenger trips covered more than once under a covering master. Zero does not imply shared charger capacity was checked. |
| `cross_route_capacity_validated` | Whether simultaneous charging across selected routes passed the source station-capacity audit. |
| `proof_scope`, `limitations`, `notes` | Source-scoped interpretation, known limits, and provenance warnings. |

## Special authority rules

- The 57-second artifact from canceled duplicate job `772031` is retained with `authority_role=superseded_duplicate_772031`. It is not the authoritative result for job `772009`.
- The uniquely named recovered `772009` result is marked `authoritative_772009_recovery`; its scheduler publication failure remains part of the provenance.
- Files named as legacy, historical controls, or `old70` remain separate rows with `legacy_historical` authority. They are not merged with newer configurations.
- Rejected physical-replay outputs remain `rejected_observational` and cannot be used as schedules.
- In the 22:37 UTC snapshot, the three terminal-energy fixed frontiers completed, while all three dependent joint MIPs failed before optimization because older saved-pool records lacked `expanded_grid_terminal_soc_kwh`. Frontier results and failed MIP attempts are separate rows.

## Charging-start fee comparisons

`charge_start_cost` is the explicitly recorded objective coefficient per charging activity; zero is retained as zero. Missing values remain unknown, not a default of5. `electricity_cost_grid` / `electricity_cost_continuous` exclude the start penalty. `charging_starts_grid` / `charging_starts_continuous` count activities, not tariff blocks. `charged_energy_grid_kwh` / `charged_energy_continuous_kwh` report charged energy. `charging_metrics_reconcile` checks electricity plus fee×starts against the saved charging total; it does not establish route or station-capacity feasibility.
