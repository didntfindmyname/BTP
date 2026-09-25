from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from agentic_selection.data.preprocessing import CandidatePool

from agentic_selection.baselines.skyline import skyline, skyline_then_topsis


def test_skyline_excludes_dominated_point(small_2attr_pool):
    sky = skyline(CandidatePool(small_2attr_pool, ["attr1", "attr2"]))
    assert set(sky) == {"A", "B", "C"}
    assert "D" not in sky  # D=[0.3,0.3] is dominated by C=[0.6,0.5] on both attrs


def test_skyline_all_nondominated_when_no_dominance():
    df = pd.DataFrame({"a": [1.0, 0.0], "b": [0.0, 1.0]}, index=["x", "y"])
    sky = skyline(CandidatePool(df, ["a", "b"]))
    assert set(sky) == {"x", "y"}


def test_skyline_single_row():
    df = pd.DataFrame({"a": [0.5], "b": [0.5]}, index=["only"])
    assert list(skyline(CandidatePool(df, ["a", "b"]))) == ["only"]


def test_skyline_chunking_invariant_to_result():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.random((250, 4)), columns=list("abcd"))
    full = skyline(CandidatePool(df, list("abcd")), chunk_size=10000)
    chunked = skyline(CandidatePool(df, list("abcd")), chunk_size=31)
    assert set(full) == set(chunked)


def test_skyline_rejects_nan():
    df = pd.DataFrame({"a": [0.5, float("nan")], "b": [0.5, 0.5]})
    with pytest.raises(ValueError):
        skyline(CandidatePool(df, ["a", "b"]))


def test_skyline_then_topsis_non_members_get_sentinel(small_2attr_pool):
    w = {"attr1": 0.5, "attr2": 0.5}
    scores = skyline_then_topsis(CandidatePool(small_2attr_pool, ["attr1", "attr2"]), w)
    assert scores["D"] == -1.0
    assert scores["D"] < scores[["A", "B", "C"]].min()
    # top choice must come from the skyline set
    assert scores.sort_values(ascending=False).index[0] in {"A", "B", "C"}
