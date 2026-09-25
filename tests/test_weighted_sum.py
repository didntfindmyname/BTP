from __future__ import annotations

import pandas as pd
import pytest
from agentic_selection.data.preprocessing import CandidatePool

from agentic_selection.baselines.weighted_sum import weighted_sum, rank_weighted_sum


def test_weighted_sum_hand_computed(small_2attr_pool):
    w = {"attr1": 0.5, "attr2": 0.5}
    scores = weighted_sum(CandidatePool(small_2attr_pool, ["attr1", "attr2"]), w)
    assert scores["A"] == pytest.approx(0.5)
    assert scores["B"] == pytest.approx(0.65)
    assert scores["C"] == pytest.approx(0.55)
    assert scores["D"] == pytest.approx(0.3)


def test_weighted_sum_renormalizes_weights_not_summing_to_one(small_2attr_pool):
    w_scaled = {"attr1": 1.0, "attr2": 1.0}  # sums to 2, should renormalize to 0.5/0.5
    scores = weighted_sum(CandidatePool(small_2attr_pool, ["attr1", "attr2"]), w_scaled)
    assert scores["A"] == pytest.approx(0.5)


def test_weighted_sum_missing_attribute_column_raises(small_2attr_pool):
    with pytest.raises(KeyError):
        weighted_sum(CandidatePool(small_2attr_pool, ["nonexistent"]), {"attr1": 1.0})


def test_weighted_sum_missing_weight_for_attribute_raises(small_2attr_pool):
    with pytest.raises(ValueError):
        weighted_sum(CandidatePool(small_2attr_pool, ["attr1", "attr2"]), {"attr1": 1.0})


def test_weighted_sum_rejects_negative_weights(small_2attr_pool):
    with pytest.raises(ValueError):
        weighted_sum(CandidatePool(small_2attr_pool, ["attr1", "attr2"]), {"attr1": -0.5, "attr2": 1.5})


def test_weighted_sum_rejects_nan_values():
    df = pd.DataFrame({"a": [0.5, float("nan")], "b": [0.5, 0.5]})
    with pytest.raises(ValueError):
        weighted_sum(CandidatePool(df, ["a", "b"]), {"a": 0.5, "b": 0.5})


def test_rank_weighted_sum_sorted_descending(small_2attr_pool):
    w = {"attr1": 0.5, "attr2": 0.5}
    ranked = rank_weighted_sum(CandidatePool(small_2attr_pool, ["attr1", "attr2"]), w)
    assert list(ranked.index) == ["B", "C", "A", "D"]
    assert ranked["score"].is_monotonic_decreasing
