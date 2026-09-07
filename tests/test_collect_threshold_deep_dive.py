import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/event_uniform_envelope/collect_threshold_deep_dive.py"
)
SPEC = importlib.util.spec_from_file_location("collect_threshold_deep_dive", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_telemetry_sums_stages_without_double_counting_shortest_path(tmp_path):
    source = tmp_path / "source.jsonl"
    resume = tmp_path / "resume.jsonl"
    source.write_text("\n".join([
        json.dumps({
            "record_type": "phase", "phase": "master_attempt",
            "duration_s": 2.0, "outcome": "ok",
        }),
        json.dumps({
            "record_type": "phase", "phase": "pricing_extra_columns",
            "duration_s": 5.0, "outcome": "ok",
        }),
    ]) + "\n")
    resume.write_text("\n".join([
        json.dumps({
            "record_type": "phase", "phase": "master_attempt",
            "duration_s": 3.0, "outcome": "error",
            "details": {"method": "highs-ds"},
        }),
        json.dumps({
            "record_type": "phase", "phase": "pricing_shortest_path",
            "duration_s": 1.0, "outcome": "ok",
        }),
    ]) + "\n")

    totals, rows, malformed, errors = MODULE.telemetry([source, resume])

    assert totals["master_attempt"] == 5.0
    assert totals["pricing_extra_columns"] == 5.0
    assert totals["pricing_shortest_path"] == 1.0
    assert rows == 4
    assert malformed == 0
    assert len(errors) == 1


def test_outcome_labels_preserve_baseline_and_continuation_semantics():
    assert MODULE.classify(
        {"certified_rc_optimal": True}, 172800.0, True
    ) == "certified_le_12h"
    assert MODULE.classify(
        {"certified_rc_optimal": True}, 172800.0, False
    ) == "certified_12_to_48h"
    assert MODULE.classify(
        {"stop_reason": "wall_limit", "wall_s": 172790.0}, 172800.0, False
    ) == "wall_cap_48h"
    assert MODULE.classify(
        {"stop_reason": "master_failed", "wall_s": 100.0}, 172800.0, False
    ) == "master_failed"


def test_iteration_sampling_keeps_endpoints_and_tail_improvement():
    rows = [
        {"iteration": float(i), "lp_obj": 100.0 - i}
        for i in range(1000)
    ]
    sampled = MODULE.sample_iterations(rows, maximum=20)

    assert len(sampled) == 20
    assert sampled[0]["iteration"] == 0.0
    assert sampled[-1]["iteration"] == 999.0
    assert MODULE.tail_improvement(rows, 100) == 100.0
