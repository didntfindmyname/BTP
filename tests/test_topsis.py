from __future__ import annotations

import math

import pandas as pd
import pytest
from agentic_selection.data.preprocessing import CandidatePool

from agentic_selection.baselines.topsis import topsis, rank_topsis


def test_topsis_hand_computed_ABC_only():
    """Independently re-derived closed form (see project dev notes):
    with X = [[0.8,0.2],[0.4,0.9],[0.6,0.5]], w=[0.5,0.5]:
        CC_A = 4/11, CC_B = 7/11, CC_C = sqrt(13) / (2*sqrt(5) + sqrt(13))
    """
    df = pd.DataFrame({"attr1": [0.8, 0.4, 0.6], "attr2": [0.2, 0.9, 0.5]}, index=["A", "B", "C"])
    w = {"attr1": 0.5, "attr2": 0.5}
    scores = topsis(CandidatePool(df, ["attr1", "attr2"]), w)

    expected_c = math.sqrt(13) / (2 * math.sqrt(5) + math.sqrt(13))
    assert scores["A"] == pytest.approx(4 / 11, abs=1e-9)
    assert scores["B"] == pytest.approx(7 / 11, abs=1e-9)
    assert scores["C"] == pytest.approx(expected_c, abs=1e-9)


def test_topsis_single_candidate_is_trivially_best():
    df = pd.DataFrame({"a": [0.5], "b": [0.5]}, index=["only"])
    scores = topsis(CandidatePool(df, ["a", "b"]), {"a": 0.5, "b": 0.5})
    assert scores["only"] == pytest.approx(1.0)


def test_topsis_empty_pool_returns_empty_series():
    df = pd.DataFrame({"a": [], "b": []})
    scores = topsis(CandidatePool(df, ["a", "b"]), {"a": 0.5, "b": 0.5})
    assert len(scores) == 0


def test_topsis_identical_rows_all_get_max_closeness_no_nan():
    df = pd.DataFrame({"a": [0.5, 0.5, 0.5], "b": [0.3, 0.3, 0.3]}, index=["x", "y", "z"])
    scores = topsis(CandidatePool(df, ["a", "b"]), {"a": 0.5, "b": 0.5})
    assert not scores.isnull().any()
    assert (scores == 1.0).all()


def test_topsis_rejects_nan_values():
    df = pd.DataFrame({"a": [0.5, float("nan")], "b": [0.5, 0.5]})
    with pytest.raises(ValueError):
        topsis(CandidatePool(df, ["a", "b"]), {"a": 0.5, "b": 0.5})


def test_rank_topsis_sorted_descending():
    df = pd.DataFrame({"attr1": [0.8, 0.4, 0.6], "attr2": [0.2, 0.9, 0.5]}, index=["A", "B", "C"])
    ranked = rank_topsis(CandidatePool(df, ["attr1", "attr2"]), {"attr1": 0.5, "attr2": 0.5})
    assert list(ranked.index) == ["B", "C", "A"]
