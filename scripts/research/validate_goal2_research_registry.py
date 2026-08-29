#!/usr/bin/env python3
"""Validate the Goal 1 proof audit and Goal 2 research registry."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "analysis" / "goal2_scale_study_20260830"
WORKBOOK = (
    ROOT
    / "outputs"
    / "goal2-research-registry-20260830"
    / "EVSP_DR_goal1_goal2_research_registry.xlsx"
)


def rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truth(value: str) -> bool:
    return value.strip().lower() == "true"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    proof = rows("goal1_model_proof_registry.csv")
    historical = rows("goal1_historical_primary_proofs.csv")
    summary = rows("goal1_proof_summary.csv")
    instances = rows("goal2_instance_registry.csv")
    progress = rows("goal2_event_ladder_progress.csv")
    sampling = rows("goal2_sampling_plan.csv")
    schema = rows("goal2_run_result_schema.csv")

    require(len(proof) == 54, "current proof registry must contain 54 rows")
    require(
        len({(r["cell_id"], r["representation_id"]) for r in proof}) == 54,
        "current proof keys must be unique",
    )
    require(sum(truth(r["L_model_certified"]) for r in proof) == 54, "expected 54 certified LP rows")
    require(sum(truth(r["I_model_proven"]) for r in proof) == 39, "expected 39 full-model integer proofs")
    require(sum(truth(r["I_pool_proven"]) for r in proof) == 42, "expected 42 finite-pool integer proofs")

    expected_model_open = {
        *(('uniform_10_10', cell) for cell in ('k02_s3', 'k03_s1', 'k03_s2', 'k05_s1', 'k05_s2', 'k05_s3')),
        *(('uniform_4_5', cell) for cell in ('k03_s1', 'k03_s2', 'k05_s2', 'k05_s3')),
        *(('uniform_2_5', cell) for cell in ('k02_s1', 'k03_s1', 'k05_s1', 'k05_s2', 'k05_s3')),
    }
    actual_model_open = {
        (r["representation_id"], r["cell_id"])
        for r in proof
        if not truth(r["I_model_proven"])
    }
    require(actual_model_open == expected_model_open, "unexpected current I_model open-row set")

    event = [r for r in proof if r["representation_id"] == "event_2p5_event5"]
    require(len(event) == 9, "event proof slice must contain nine rows")
    require(sum(truth(r["I_model_proven"]) for r in event) == 9, "event I_model must be 9/9")
    require(sum(truth(r["I_pool_proven"]) for r in event) == 7, "event I_pool must be 7/9")
    require(
        {r["cell_id"] for r in event if not truth(r["I_pool_proven"])} == {"k05_s2", "k05_s3"},
        "event pool-open cells must be k05_s2 and k05_s3",
    )

    require(len(historical) == 9, "historical proof table must contain nine rows")
    require(sum(truth(r["I_model_proven"]) for r in historical) == 7, "historical I_model must be 7/9")
    require(
        {r["cell_id"] for r in historical if not truth(r["I_model_proven"])} == {"k05_s1", "k05_s3"},
        "historical open cells must be k05_s1 and k05_s3",
    )

    require(len(summary) == 3, "proof summary must contain three scoped rows")
    require(len(instances) == 40, "existing instance registry must contain 40 rows")
    require(len({r["cell_id"] for r in instances}) == 40, "instance cell_id values must be unique")
    require(
        Counter(int(r["scale"]) for r in instances)
        == Counter({2: 6, 3: 6, 5: 6, 8: 6, 13: 6, 20: 6, 30: 3, 40: 1}),
        "unexpected existing scale distribution",
    )

    require(len(progress) == 40, "event progress table must contain 40 rows")
    require({r["cell_id"] for r in progress} == {r["cell_id"] for r in instances}, "progress and instance keys must match")
    require(
        Counter(r["run_status"] for r in progress)
        == Counter({"committed_result_available": 9, "validated_input_not_run": 31}),
        "unexpected event-ladder progress states",
    )

    require(len(sampling) == 9, "sampling plan must contain nine target rows")
    medium = [r for r in sampling if int(r["target_fleet"]) in {8, 10, 13, 20}]
    require(sum(int(r["planned_total_n"]) for r in medium) == 66, "medium study must plan 66 rows")
    require(sum(int(r["new_probability_sample_n"]) + int(r["new_feature_space_n"]) for r in sampling) == 48, "expected 48 new selections")

    required_schema = {r["column_name"] for r in schema if truth(r["required"])}
    for field in {
        "run_id", "cell_id", "sample_family", "solver_commit", "instance_sha256",
        "L_model", "I_model_lower", "I_model_upper", "I_pool_lower", "I_pool_upper",
        "I_timed", "t_first_target_s", "t_certificate_s", "pricing_wall_s",
        "master_lp_wall_s", "mip_wall_s", "peak_rss_mb", "physical_witness_valid",
    }:
        require(field in required_schema, f"required run-schema field missing: {field}")

    require(WORKBOOK.is_file() and WORKBOOK.stat().st_size > 0, "companion workbook missing or empty")
    print(
        "validated Goal 1/2 registry: "
        "54 current proof rows, 9 historical rows, 40 existing instances, "
        "66 planned medium-study rows"
    )


if __name__ == "__main__":
    main()
