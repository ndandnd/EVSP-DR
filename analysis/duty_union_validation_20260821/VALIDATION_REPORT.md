# Independent validation: 18 duty-union instances (ladder commit `72c7bf4`)

Date 2026-08-21. Validator: records curator (`cursor/records-curator-2969-8502`).
Producer: Agent A, `cursor/ladder-lite-20260819-2969@72c7bf4` (canonical
`D0112`, inbox LOCAL-6). **The producer's report was not used as evidence**;
every claim below is re-derived by
`scripts/curator/validate_duty_union_extension.py` (stdlib only, committed)
from the committed ground truth. No cluster jobs were run.

## Ground truth

Duty membership and trip counts are validated against the GIRO industrial
master `data/Par_VehicleDetails_Updated.csv` (duty = `VehicleTask` over
`Identifier == "Regular"` rows with a non-null `Ordered_Trip_ID`). Its SHA-256,

    6b46acce8b0870aff967c73aac372b90873ed32a6e424e55b851e4b8676ab57f

equals the `source_master_sha256` recorded in the frozen-instances manifest
and the `source_sha256` recorded in the SyntheticRandom manifest, so validator
and producers used the same source identity. `target_fleet = k` is ground
truth for duty unions because the industrial schedule ran one bus per duty.

## Verdict

**All 29 checks pass (exit 0).** Per instance (12 checks each): file SHA-256
against the extension record, the 6-selection manifest row and the producer MD
table; `duty_count = scale = target_fleet`; weekday-variant policy
`one_literal_per_numeric_base_no_siblings` (numeric bases pairwise distinct);
every claimed duty exists in the master; `duty_set_sha256` recomputed from the
documented canonical-JSON convention; instance trip count equals the sum of
the claimed duties' master trip counts; the instance's trip multiset equals
the exact union of the claimed duties' master trips (full physical identity:
endpoints, times, distance, energy, `Ordered_Trip_ID`); all rows `Regular`;
rows sorted by start-minute then trip id with `count_trip_id = 0..n-1`; and
the manifest row's trip hashes (`ordered_trip_id_set`, `ordered_trip_sequence`,
`solver_local_trip_index`) recomputed. Structurally: the 6-selection manifest
is byte-prefixed by the legacy 22-row manifest (SHA-256 `a7ef8b77…`), its own
SHA-256 is the stated `8bf292bf…`, and the 40-cell membership preflight hashes
to the stated `ba7074a7…`.

This executes the correction demanded by authoritative `B0031` (regenerate
selections 4–6 from the seed-20260803 duty-union families and commit the
CSVs) and validates it independently. The ledgers are frozen, so `B0031`'s
status row is not edited; the operator may close it citing this report.

## Duty-union family (target_fleet = k is ground truth)

| instance | duties (per-duty trip counts) | trips | 12 checks |
|---|---|---:|---|
| k02_r4 | `13319=27;13414=12` | 39 | all pass |
| k02_r5 | `13321=22;13323=8` | 30 | all pass |
| k02_r6 | `13312=46;13317=37` | 83 | all pass |
| k03_r4 | `13312=46;13317=37;13321=22` | 105 | all pass |
| k03_r5 | `13305=30;13310=24;13313=37` | 91 | all pass |
| k03_r6 | `13323=8;13409=17;13410=15` | 40 | all pass |
| k05_r4 | `13303=54;13324t=18;13326=21;13404=17;13407=17` | 127 | all pass |
| k05_r5 | `13304=37;13311=14;13406=14;13409=17;13410=15` | 97 | all pass |
| k05_r6 | `13305=30;13309=22;13313=37;13316uwt=21;13404=17` | 127 | all pass |
| k08_r4 | `13307=15;13312=46;13313=37;13315=31;13406=14;13407=17;13408=11;13410=15` | 186 | all pass |
| k08_r5 | `13306=30;13316uwt=21;13321=22;13323=8;13402=17;13405=13;13408=11;13411=14` | 136 | all pass |
| k08_r6 | `13311=14;13313=37;13316m=23;13323=8;13402=17;13405=13;13409=17;13413=15` | 144 | all pass |
| k13_r4 | `13303=54;13305=30;13307=15;13308=31;13312=46;13315=31;13320=16;13405=13;13407=17;13408=11;13410=15;13411=14;13412=15` | 308 | all pass |
| k13_r5 | `13302=50;13303=54;13304=37;13306=30;13307=15;13308=31;13310=24;13315=31;13319=27;13325=42;13411=14;13413=15;13414=12` | 382 | all pass |
| k13_r6 | `13301=36;13303=54;13305=30;13315=31;13317=37;13319=27;13325=42;13326=21;13404=17;13406=14;13409=17;13410=15;13413=15` | 356 | all pass |
| k20_r4 | `13301=36;13303=54;13304=37;13310=24;13311=14;13312=46;13316m=23;13318=33;13320=16;13321=22;13322=26;13323=8;13324t=18;13325=42;13401=12;13403=14;13404=17;13407=17;13409=17;13414=12` | 488 | all pass |
| k20_r5 | `13304=37;13305=30;13309=22;13310=24;13312=46;13314=16;13315=31;13317=37;13318=33;13319=27;13321=22;13323=8;13324t=18;13401=12;13402=17;13403=14;13409=17;13412=15;13413=15;13414=12` | 453 | all pass |
| k20_r6 | `13301=36;13302=50;13304=37;13305=30;13308=31;13310=24;13311=14;13315=31;13316m=23;13323=8;13324t=18;13325=42;13404=17;13405=13;13406=14;13407=17;13409=17;13410=15;13412=15;13413=15` | 467 | all pass |
Machine-readable: `duty_union_validation_results.csv`.

## SyntheticRandom family (separate; no comparator)

Kept in its **own manifest** (`data/scale_ladder/instances/
random_goal1_seed_20260821/manifest.csv`, absent from the duty-union manifest
— verified) and its **own results table**
(`analysis/synthetic_random_goal1_seed_20260821/results.csv`). These
instances have **no ground-truth fleet comparator** and their own manifest
records `single_day_verified=False`; only file identity is validated here.
The two families must never share a table.

| instance | trips | sha256 check | single_day_verified |
|---|---:|---|---|
| Practice_SyntheticRandom_2bus_s20260821_r04.csv | 39 | pass | False |
| Practice_SyntheticRandom_2bus_s20260821_r05.csv | 44 | pass | False |
| Practice_SyntheticRandom_2bus_s20260821_r06.csv | 59 | pass | False |
| Practice_SyntheticRandom_3bus_s20260821_r04.csv | 60 | pass | False |
| Practice_SyntheticRandom_3bus_s20260821_r05.csv | 90 | pass | False |
| Practice_SyntheticRandom_3bus_s20260821_r06.csv | 66 | pass | False |
| Practice_SyntheticRandom_5bus_s20260821_r04.csv | 114 | pass | False |
| Practice_SyntheticRandom_5bus_s20260821_r05.csv | 82 | pass | False |
| Practice_SyntheticRandom_5bus_s20260821_r06.csv | 108 | pass | False |
| Practice_SyntheticRandom_8bus_s20260821_r04.csv | 211 | pass | False |
| Practice_SyntheticRandom_8bus_s20260821_r05.csv | 154 | pass | False |
| Practice_SyntheticRandom_8bus_s20260821_r06.csv | 179 | pass | False |
| Practice_SyntheticRandom_13bus_s20260821_r04.csv | 266 | pass | False |
| Practice_SyntheticRandom_13bus_s20260821_r05.csv | 310 | pass | False |
| Practice_SyntheticRandom_13bus_s20260821_r06.csv | 332 | pass | False |
| Practice_SyntheticRandom_20bus_s20260821_r04.csv | 512 | pass | False |
| Practice_SyntheticRandom_20bus_s20260821_r05.csv | 434 | pass | False |
| Practice_SyntheticRandom_20bus_s20260821_r06.csv | 458 | pass | False |
Machine-readable: `synthetic_random_family_check.csv`.

## Left unverified, deliberately

- Preflight JSON contents beyond hash and 40-cell count (not in scope).
- SyntheticRandom weekday compatibility (its own manifest marks it
  unverified; the family is a robustness set, not a Goal-1 comparator).
- Any solver behaviour on these instances — this is input validation only.

## Executed output (verbatim)

    [PASS] extension record lists 18 instances — found 18
    [PASS] GIRO master parsed — 42 duties in master
    [PASS] legacy manifest sha256 matches stated — a7ef8b77351440a8d7873b949891663ca7b28f135d366d4c6b003d09ca84839a
    [PASS] 6sel manifest sha256 matches stated — 8bf292bf71229d29feffa7dca4bfaa2f5d6b5943863559468c594a731bd904d3
    [PASS] 6sel manifest = legacy bytes + appended rows — legacy 14723B prefix of 25665B
    [PASS] 6sel manifest has 22 legacy + 18 new rows — 40 rows
    [PASS] campaign manifest binds the stated 6sel manifest sha — 8bf292bf71229d29feffa7dca4bfaa2f5d6b5943863559468c594a731bd904d3
    [PASS] preflight json sha256 matches stated — ba7074a7ed5b342cb64d350fd95945099170b68bc3847619ef94d6f728fbe656
    [PASS] preflight contains 40 cells — 40 cells
    [PASS] k02_r4 all checks — 12/12
    [PASS] k02_r5 all checks — 12/12
    [PASS] k02_r6 all checks — 12/12
    [PASS] k03_r4 all checks — 12/12
    [PASS] k03_r5 all checks — 12/12
    [PASS] k03_r6 all checks — 12/12
    [PASS] k05_r4 all checks — 12/12
    [PASS] k05_r5 all checks — 12/12
    [PASS] k05_r6 all checks — 12/12
    [PASS] k08_r4 all checks — 12/12
    [PASS] k08_r5 all checks — 12/12
    [PASS] k08_r6 all checks — 12/12
    [PASS] k13_r4 all checks — 12/12
    [PASS] k13_r5 all checks — 12/12
    [PASS] k13_r6 all checks — 12/12
    [PASS] k20_r4 all checks — 12/12
    [PASS] k20_r5 all checks — 12/12
    [PASS] k20_r6 all checks — 12/12
    [PASS] no SyntheticRandom row in the duty-union manifest
    [PASS] SyntheticRandom family hash-verified in its own manifest — 18 rows
    
    29 checks, 0 failures
