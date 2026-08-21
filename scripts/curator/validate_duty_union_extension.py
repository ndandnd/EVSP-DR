#!/usr/bin/env python3
"""Independent validation of the 18 duty-union instances from ladder commit 72c7bf4.

Validator role (CURATOR_WORKORDER_20260821 deliverable 6): Agent A is the
producer; its report is not taken as evidence. Every check below re-derives
the claim from the committed ground truth:

- ground truth for duty membership and trip counts is the GIRO industrial
  master `data/Par_VehicleDetails_Updated.csv` (duty = `VehicleTask` over
  `Identifier == "Regular"` rows with a non-null `Ordered_Trip_ID`);
- stated hashes are recomputed from bytes on this branch;
- the canonical JSON hash convention is re-implemented here from its
  documented definition (sorted keys, compact separators), not imported from
  the producer.

Everything is stdlib; no cluster access, no pandas.
"""
import csv
import hashlib
import json
import pathlib
import re
import sys
from collections import Counter

REPO = pathlib.Path(__file__).resolve().parents[2]
INST = REPO / "data/scale_ladder/instances"
MASTER = REPO / "data/Par_VehicleDetails_Updated.csv"
EXT_RECORD = INST / "duty_union_extension_seed20260803.json"
MANIFEST_6SEL = INST / "scale_ladder_instance_manifest_6sel_seed20260803.csv"
MANIFEST_LEGACY = INST / "scale_ladder_instance_manifest.csv"
PREFLIGHT = REPO / "data/scale_ladder/known_membership_preflight_6sel_seed20260803.json"
CAMPAIGN_6SEL = INST / "campaign_input_manifest_6sel_seed20260803.json"
SYNTH_DIR = INST / "random_goal1_seed_20260821"
OUT = REPO / "analysis/duty_union_validation_20260821"

# Producer-stated identities from SIX_SELECTION_DUTY_UNION_EXTENSION_20260821.md
STATED_6SEL_SHA = "8bf292bf71229d29feffa7dca4bfaa2f5d6b5943863559468c594a731bd904d3"
STATED_LEGACY_SHA = "a7ef8b77351440a8d7873b949891663ca7b28f135d366d4c6b003d09ca84839a"
STATED_PREFLIGHT_SHA = "ba7074a7ed5b342cb64d350fd95945099170b68bc3847619ef94d6f728fbe656"

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    return bool(ok)


def sha_file(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def canonical_sha(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def minutes(hhmm):
    hh, mm = str(hhmm).split(":")
    return int(hh) * 60 + int(mm)


def base_task(duty):
    m = re.match(r"(\d+)", str(duty))
    return m.group(1) if m else str(duty)


def trip_key(row):
    # Full physical trip identity; floats parsed so formatting differences
    # between the master and pandas-written instances cannot mask equality.
    return (row["From1"], row["Start1"], row["End1"], row["To1"],
            float(row["Distance1"]), float(row["Usage kWh"]),
            int(row["Ordered_Trip_ID"]))


def load_master_duties():
    duties = {}
    with MASTER.open(newline="") as h:
        for row in csv.DictReader(h):
            if row["Identifier"] != "Regular" or not row["Ordered_Trip_ID"]:
                continue
            duties.setdefault(str(row["VehicleTask"]), []).append(row)
    return duties


def main():
    ext = json.loads(EXT_RECORD.read_text())
    instances = ext["instances"]
    check("extension record lists 18 instances", len(instances) == 18,
          f"found {len(instances)}")

    master = load_master_duties()
    check("GIRO master parsed", len(master) > 0, f"{len(master)} duties in master")

    # --- manifest structure ---
    legacy_raw = MANIFEST_LEGACY.read_bytes()
    full_raw = MANIFEST_6SEL.read_bytes()
    check("legacy manifest sha256 matches stated",
          sha_file(MANIFEST_LEGACY) == STATED_LEGACY_SHA, sha_file(MANIFEST_LEGACY))
    check("6sel manifest sha256 matches stated",
          sha_file(MANIFEST_6SEL) == STATED_6SEL_SHA, sha_file(MANIFEST_6SEL))
    check("6sel manifest = legacy bytes + appended rows",
          full_raw.startswith(legacy_raw),
          f"legacy {len(legacy_raw)}B prefix of {len(full_raw)}B")
    with MANIFEST_6SEL.open(newline="") as h:
        man_rows = list(csv.DictReader(h))
    check("6sel manifest has 22 legacy + 18 new rows", len(man_rows) == 40,
          f"{len(man_rows)} rows")
    man_by_key = {(int(r["scale"]), int(r["selection_replicate"])): r for r in man_rows}

    campaign = json.loads(CAMPAIGN_6SEL.read_text())
    check("campaign manifest binds the stated 6sel manifest sha",
          campaign["instance_manifest_sha256"] == STATED_6SEL_SHA,
          campaign["instance_manifest_sha256"])

    check("preflight json sha256 matches stated",
          sha_file(PREFLIGHT) == STATED_PREFLIGHT_SHA, sha_file(PREFLIGHT))
    preflight = json.loads(PREFLIGHT.read_text())
    n_cells = len(preflight.get("cells", preflight if isinstance(preflight, list) else []))
    check("preflight contains 40 cells", n_cells == 40, f"{n_cells} cells")

    # --- per-instance validation against the GIRO master ---
    results = []
    for inst in sorted(instances, key=lambda r: (r["scale"], r["selection_replicate"])):
        scale, rep = inst["scale"], inst["selection_replicate"]
        name = f"k{scale:02d}_r{rep}"
        path = REPO / inst["relative_path"]
        duties = json.loads(inst["duties_json"])
        row_ok = {}

        row_ok["file_sha256"] = sha_file(path) == inst["instance_file_sha256"]
        row_ok["duty_count=scale=target_fleet"] = (
            len(duties) == inst["duty_count"] == scale == inst["target_fleet"])
        bases = [base_task(d) for d in duties]
        row_ok["weekday_policy_no_siblings"] = len(set(bases)) == len(bases)
        row_ok["duties_exist_in_master"] = all(d in master for d in duties)
        row_ok["duty_set_sha256"] = canonical_sha(sorted(duties)) == inst["duty_set_sha256"]

        with path.open(newline="") as h:
            inst_rows = list(csv.DictReader(h))
        expected = [r for d in duties for r in master.get(d, [])]
        per_duty_counts = {d: len(master.get(d, [])) for d in duties}
        row_ok["trip_count_equals_duty_union_total"] = len(inst_rows) == len(expected)
        row_ok["trip_multiset_equals_duty_union"] = (
            Counter(map(trip_key, inst_rows)) == Counter(map(trip_key, expected)))
        row_ok["all_rows_regular"] = all(r["Identifier"] == "Regular" for r in inst_rows)
        sort_keys = [(minutes(r["Start1"]), int(r["Ordered_Trip_ID"])) for r in inst_rows]
        row_ok["sorted_by_start_then_trip"] = sort_keys == sorted(sort_keys)
        row_ok["count_trip_id_sequence"] = (
            [int(r["count_trip_id"]) for r in inst_rows] == list(range(len(inst_rows))))

        man = man_by_key.get((scale, rep))
        ids = [int(r["Ordered_Trip_ID"]) for r in inst_rows]
        row_ok["manifest_row_consistent"] = man is not None and (
            man["instance_file_sha256"] == inst["instance_file_sha256"]
            and man["duties_json"] == inst["duties_json"]
            and int(man["trip_count"]) == len(inst_rows)
            and man["ordered_trip_id_set_sha256"] == canonical_sha(sorted(ids))
            and man["ordered_trip_sequence_sha256"] == canonical_sha(ids)
            and man["solver_local_trip_index_sha256"] == canonical_sha(list(range(len(ids))))
            and man["weekday_variant_policy"] == "one_literal_per_numeric_base_no_siblings"
            and man["generator_family"] == inst["generator_family"])

        results.append({
            "instance": name, "scale": scale, "selection_replicate": rep,
            "duty_count": len(duties), "trip_count": len(inst_rows),
            "per_duty_trip_counts": ";".join(f"{d}={per_duty_counts[d]}" for d in duties),
            "instance_file_sha256": inst["instance_file_sha256"],
            **{k: ("pass" if v else "FAIL") for k, v in row_ok.items()},
        })
        check(f"{name} all checks", all(row_ok.values()),
              ", ".join(k for k, v in row_ok.items() if not v) or "12/12")

    # --- family separation ---
    manifest_text = full_raw.decode()
    check("no SyntheticRandom row in the duty-union manifest",
          "SyntheticRandom" not in manifest_text and "random_goal1" not in manifest_text)

    synth_manifest = SYNTH_DIR / "manifest.csv"
    with synth_manifest.open(newline="") as h:
        synth_rows = list(csv.DictReader(h))
    synth_results = []
    all_synth_ok = True
    for r in synth_rows:
        p = SYNTH_DIR / r["output_csv"]
        ok = p.exists() and sha_file(p) == r["output_sha256"]
        all_synth_ok &= ok
        synth_results.append({**r, "hash_check": "pass" if ok else "FAIL"})
    check("SyntheticRandom family hash-verified in its own manifest",
          all_synth_ok and len(synth_rows) == 18, f"{len(synth_rows)} rows")

    # --- write outputs ---
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "duty_union_validation_results.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(results[0].keys()), lineterminator="\n")
        w.writeheader(); w.writerows(results)
    if synth_results:
        with (OUT / "synthetic_random_family_check.csv").open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(synth_results[0].keys()), lineterminator="\n")
            w.writeheader(); w.writerows(synth_results)

    n_fail = sum(1 for _, ok, _ in CHECKS if not ok)
    for name, ok, detail in CHECKS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    print(f"\n{len(CHECKS)} checks, {n_fail} failures")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
