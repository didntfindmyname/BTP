from __future__ import annotations

import json

import pytest

from agentic_selection.agent.controller import AgentController
from agentic_selection.agent.llm_backends import MockBackend

ATTRS = ["response_time", "availability", "throughput"]


def make_pool():
    import pandas as pd
    from agentic_selection.data.preprocessing import CandidatePool

    df = pd.DataFrame(
        {
            "response_time": [0.9, 0.3, 0.6],
            "availability": [0.2, 0.9, 0.5],
            "throughput": [0.5, 0.5, 0.9],
        },
        index=["svcA", "svcB", "svcC"],
    )
    return CandidatePool(df, ATTRS)


def test_controller_well_formed_response_no_fallback(tmp_path):
    good = json.dumps(
        {"weights": {"response_time": 0.2, "availability": 0.5, "throughput": 0.3}, "strategy": "topsis", "justification": "j"}
    )
    ctrl = AgentController(MockBackend(fixed_response=good), ATTRS, tmp_path / "mem.jsonl")
    decision = ctrl.decide("financial task", make_pool())
    assert not decision.fallback_triggered
    assert decision.strategy == "topsis"
    assert decision.top_service_id() in make_pool().index


def test_controller_garbage_response_triggers_fallback(tmp_path):
    ctrl = AgentController(MockBackend(fixed_response="not json at all, sorry"), ATTRS, tmp_path / "mem.jsonl")
    decision = ctrl.decide("some task", make_pool())
    assert decision.fallback_triggered


def test_controller_strategy_override_forces_topsis_even_if_llm_says_otherwise(tmp_path):
    resp = json.dumps(
        {"weights": {"response_time": 0.3, "availability": 0.3, "throughput": 0.4}, "strategy": "weighted_sum", "justification": "j"}
    )
    ctrl = AgentController(MockBackend(fixed_response=resp), ATTRS, tmp_path / "mem.jsonl")
    decision = ctrl.decide("task", make_pool(), strategy_override="topsis")
    assert decision.strategy == "topsis"  # override wins for execution
    # but the memory log should still reflect what the LLM actually chose
    logged = ctrl.memory.load_all()
    assert logged[-1].strategy == "weighted_sum"


def test_controller_logs_to_memory_and_uses_memory_across_calls(tmp_path):
    resp = json.dumps({"weights": {"response_time": 1.0, "availability": 1.0, "throughput": 1.0}, "strategy": "topsis", "justification": "j"})
    mem_path = tmp_path / "mem.jsonl"
    ctrl = AgentController(MockBackend(fixed_response=resp), ATTRS, mem_path)
    ctrl.decide("task one", make_pool())
    ctrl.decide("task two", make_pool())
    assert len(ctrl.memory.load_all()) == 2


def test_controller_use_memory_false_does_not_log(tmp_path):
    resp = json.dumps({"weights": {"response_time": 1.0, "availability": 1.0, "throughput": 1.0}, "strategy": "topsis", "justification": "j"})
    ctrl = AgentController(MockBackend(fixed_response=resp), ATTRS, tmp_path / "mem.jsonl")
    ctrl.decide("task", make_pool(), use_memory=False)
    assert len(ctrl.memory.load_all()) == 0


def test_controller_sequenced_responses_can_script_a_specific_scenario(tmp_path):
    responses = [
        json.dumps({"weights": {"response_time": 1.0, "availability": 1.0, "throughput": 1.0}, "strategy": "topsis", "justification": "ok"}),
        "malformed garbage",
    ]
    ctrl = AgentController(MockBackend(responses=responses), ATTRS, tmp_path / "mem.jsonl")
    d1 = ctrl.decide("first", make_pool())
    d2 = ctrl.decide("second", make_pool())
    assert not d1.fallback_triggered
    assert d2.fallback_triggered


def test_controller_unknown_strategy_override_raises(tmp_path):
    resp = json.dumps({"weights": {"response_time": 1.0, "availability": 1.0, "throughput": 1.0}, "strategy": "topsis", "justification": "j"})
    ctrl = AgentController(MockBackend(fixed_response=resp), ATTRS, tmp_path / "mem.jsonl")
    with pytest.raises(ValueError):
        ctrl.decide("task", make_pool(), strategy_override="not_a_real_strategy")
