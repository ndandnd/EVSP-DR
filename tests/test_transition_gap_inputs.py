import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "scripts" / "event_uniform_envelope"
INPUTS = (
    REPO / "data" / "scale_ladder" / "instances"
    / "transition_gap_20260902"
)


def test_frozen_transition_gap_inputs_validate():
    completed = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "validate_transition_gap_inputs.py"),
            "--repo",
            str(REPO),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "rows=36" in completed.stdout
    assert "42/42" in completed.stdout


def test_transition_gap_selection_contract():
    with (INPUTS / "selection_manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert Counter(
        (int(row["scale"]), row["sample_family"]) for row in rows
    ) == Counter({
        **{(scale, "probability"): 6 for scale in (3, 4, 6, 7)},
        **{(scale, "stress"): 3 for scale in (3, 4, 6, 7)},
    })
    assert {row["selection_role"] for row in rows if row["sample_family"] == "stress"} == {
        "trip_heavy", "energy_heavy", "tight_gap"
    }
    assert all(
        row["known_partition_continuous_physical_upper_bound"] == "True"
        for row in rows
    )
    plan = json.loads((INPUTS / "input_plan.json").read_text())
    assert plan["known_partition_caveat"].startswith(
        "physical k-route upper bound only"
    )


def test_transition_gap_launcher_keeps_model_identity_explicit():
    launcher = (TOOLS / "submit_transition_gap_event.sh").read_text()
    worker = (TOOLS / "medium_event_cg.sub").read_text()
    assert "EVSP_TIME_MODEL=event" in launcher
    assert "EVSP_EVENT_ARC_MODE=lazy" in launcher
    assert "--time-model \"$EVSP_TIME_MODEL\"" in worker
    assert "--event-arc-mode \"$EVSP_EVENT_ARC_MODE\"" in worker
    assert "--initial-pool singletons" in worker
