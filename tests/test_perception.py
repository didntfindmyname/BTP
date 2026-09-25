from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from agentic_selection.data.preprocessing import CandidatePool

from agentic_selection.agent.perception import summarize_pool


def test_summarize_pool_basic_shape():
    df = pd.DataFrame({"a": [0.1, 0.5, 0.9], "b": [0.9, 0.5, 0.1]})
    p = summarize_pool(CandidatePool(df, ["a", "b"]))
    assert p.n_candidates == 3
    assert set(p.variance.keys()) == {"a", "b"}
    assert "a|b" in p.pairwise_correlation


def test_summarize_pool_perfect_negative_correlation_detected():
    df = pd.DataFrame({"a": [0.0, 0.5, 1.0], "b": [1.0, 0.5, 0.0]})
    p = summarize_pool(CandidatePool(df, ["a", "b"]))
    assert p.pairwise_correlation["a|b"] == pytest.approx(-1.0, abs=1e-6)
    assert p.conflict_score == pytest.approx(1.0, abs=1e-6)


def test_summarize_pool_no_conflict_when_positively_correlated():
    df = pd.DataFrame({"a": [0.0, 0.5, 1.0], "b": [0.0, 0.5, 1.0]})
    p = summarize_pool(CandidatePool(df, ["a", "b"]))
    assert p.conflict_score == pytest.approx(0.0, abs=1e-6)


def test_summarize_pool_constant_column_correlation_is_zero_not_nan():
    df = pd.DataFrame({"a": [0.5, 0.5, 0.5], "b": [0.1, 0.5, 0.9]})
    p = summarize_pool(CandidatePool(df, ["a", "b"]))
    assert p.pairwise_correlation["a|b"] == 0.0
    assert p.variance["a"] == 0.0


def test_summarize_pool_to_vector_length_matches_schema():
    df = pd.DataFrame({"a": [0.1, 0.5], "b": [0.5, 0.9], "c": [0.2, 0.4]})
    p = summarize_pool(CandidatePool(df, ["a", "b", "c"]))
    vec = p.to_vector()
    # 3 variances + 3 pairwise correlations (a|b, a|c, b|c) + 4 scalars
    assert vec.shape[0] == 3 + 3 + 4
    assert not np.isnan(vec).any()


def test_summarize_pool_missing_fraction_from_separate_pre_imputation_df():
    imputed = pd.DataFrame({"a": [0.5, 0.5], "b": [0.5, 0.5]})
    pre_imputation = pd.DataFrame({"a": [0.5, np.nan], "b": [0.5, 0.5]})
    p = summarize_pool(CandidatePool(imputed, ["a", "b"]), missing_df=pre_imputation)
    assert p.missing_fraction == pytest.approx(0.25)  # 1 missing cell out of 4


def test_summarize_pool_prompt_text_is_nonempty_string():
    df = pd.DataFrame({"a": [0.1, 0.9], "b": [0.9, 0.1]})
    p = summarize_pool(CandidatePool(df, ["a", "b"]))
    text = p.to_prompt_text()
    assert isinstance(text, str) and len(text) > 0
