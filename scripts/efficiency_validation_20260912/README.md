# Paired efficiency validation

Nine independent paired allocations, no automatic requeue or prior-job changes. Preparation and worker writes are confined to a new campaign root. Only `launch --submit` submits jobs. A pre-existing `jobs.json` blocks resubmission, including partial launches; inspect the saved IDs manually before deciding any recovery.

Deploy clean detached baseline and capacity checkouts first. Baseline pin is the full commit containing these scripts and master/replay flags; capacity pin must include accounting fix and final selector, currently `309d98d2` (resolve its full hash on deployment). Copy the audited `source_evidence.json` and `capacity_evidence.json` to the campaign `audit/` directory. Keep the cluster policy at the campaign parent's `SCAGLIONE_RESOURCE_POLICY.md`.

```sh
python scripts/efficiency_validation_20260912/campaign.py prepare \
  --root /home/nc437/ladder-lite/efficiency_validation_20260912 \
  --audit /home/nc437/ladder-lite/efficiency_validation_20260912/audit \
  --baseline-code /home/nc437/ladder-lite/efficiency_validation_20260912/code-baseline \
  --capacity-code /home/nc437/ladder-lite/efficiency_validation_20260912/code-capacity \
  --baseline-commit FULL_BASELINE_COMMIT --capacity-commit FULL_CAPACITY_COMMIT
python scripts/efficiency_validation_20260912/campaign.py launch \
  --root /home/nc437/ladder-lite/efficiency_validation_20260912
```

Inspect the dry-run commands and manifest, then repeat launch with `--submit`. Every job has the default partition and physical-node exclusion. The launcher saves each submitted ID before verifying `scontrol`; any ambiguous failure stops further submissions and preserves the ledger. It does not cancel jobs.

| Cases | Resources | Per arm | Slurm allocation |
|---|---|---|---|
| d00_g0, d00_g1 | 2 CPU,32G | 7200s | 4h30 |
| w1_k08,w4_k11,w6_k12,w1_k08_repeat | 8 CPU,96G | 7200s | 8h |
| duty13406 capacity,k2 capacity,duty13406 combined peak12 | 1 CPU,24G | 10800s | 6h30 |

Alternating order makes the repeated w1_k08 comparison reverse its first order. Warm preparation builds a new source-authenticated cache once per allocation, outside paired timing, with a10800s watchdog. Both arms then require that same cache. Old cache identities are never rebound. Warm initialization remains512 selected routes/900s/8 workers, with completion order left unchanged. Each mode's total process timing includes index setup, pool parsing, replay and CG. Report import audit timing alongside setup/phase time, not as a complete importer speedup by itself.

Each process group receives SIGTERM at its external wall cap and SIGKILL after90s if necessary. CG arms get60s external allowance beyond their internal solver limit to permit checkpoint/final-write completion. A failed first arm remains preserved; the second starts if the allocation has enough time for its whole cap. Do not infer certification from process exit status. Status, pool, phase, native solver logs and command/resource/hash records remain in unique job/restart directories.

`collect --root ROOT` emits compact read-only JSON from allocation records, including preparation and first-arm starts before pair_status exists, pending cases/arms, allocation metadata, planned/observed flags, input/cache/phase/status/pool hashes and both solver certificate schemas. Source histories and route arrays are omitted. Cache hashes come from the completed preparation record; collection reads only current cache size/mtime, avoiding repeated large cache reads during paired timing. Stdout/stderr are also reported by path/stat, while status/pool/phase hashes are current. Started without a completion record is not proof of present process liveness; the collector does not infer liveness from files. Redirect its stdout to a dated collector file outside code if desired. This collector does not query the scheduler or infer MIP/physical validation/GIRO attainment.

Capacity comparisons both use corrected station-power accounting. The peak12 combined arm changes tariff and PARX power relative to flat capacity cases, so compare selector modes within that case. Generic heterogeneous-power MIP validation is outside this campaign. Historical capacity k2 binding pricing took8598s;10800s may still censor full convergence. Equal-call speed comparisons require both matching pricing calls to complete.
