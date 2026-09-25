from __future__ import annotations

import pytest
from agentic_selection.data.preprocessing import CandidatePool

from agentic_selection.agent.perception import summarize_pool
from agentic_selection.agent.reasoning import (
    LLMOutputParseError,
    build_prompt,
    parse_llm_json,
)
import pandas as pd


def test_parse_llm_json_plain():
    text = '{"weights": {"a": 0.5, "b": 0.5}, "strategy": "topsis", "justification": "x"}'
    parsed = parse_llm_json(text)
    assert parsed["strategy"] == "topsis"


def test_parse_llm_json_markdown_fenced():
    text = '```json\n{"weights": {"a": 1.0}, "strategy": "weighted_sum", "justification": "x"}\n```'
    parsed = parse_llm_json(text)
    assert parsed["strategy"] == "weighted_sum"


def test_parse_llm_json_with_preamble_text():
    text = 'Sure, here is my answer:\n{"weights": {"a": 1.0}, "strategy": "topsis", "justification": "y"}\nHope that helps!'
    parsed = parse_llm_json(text)
    assert parsed["strategy"] == "topsis"


def test_parse_llm_json_totally_invalid_raises_with_raw_text_attached():
    text = "I refuse to produce JSON today."
    with pytest.raises(LLMOutputParseError) as exc_info:
        parse_llm_json(text)
    assert exc_info.value.raw_text == text


def test_parse_llm_json_non_object_json_raises():
    with pytest.raises(LLMOutputParseError):
        parse_llm_json("[1, 2, 3]")


def test_build_prompt_contains_task_and_attributes():
    df = pd.DataFrame({"a": [0.1, 0.9], "b": [0.9, 0.1]})
    perception = summarize_pool(CandidatePool(df, ["a", "b"]))
    system, user = build_prompt("A test task description", perception, ["a", "b"])
    assert "a" in system and "b" in system
    assert "A test task description" in user
    assert "JSON" in system


def test_build_prompt_includes_memory_digest_when_present():
    df = pd.DataFrame({"a": [0.1, 0.9], "b": [0.9, 0.1]})
    perception = summarize_pool(CandidatePool(df, ["a", "b"]))
    _, user_with = build_prompt("task", perception, ["a", "b"], memory_digest="past decision info")
    _, user_without = build_prompt("task", perception, ["a", "b"], memory_digest="")
    assert "past decision info" in user_with
    assert "past decision info" not in user_without
