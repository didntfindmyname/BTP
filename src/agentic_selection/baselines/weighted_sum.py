"""Weighted-sum multi-criteria scoring.

Assumes the input DataFrame has already been through
``agentic_selection.data.preprocessing.normalize_benefit_oriented`` so every
attribute column is in [0, 1] with *higher = better* uniformly. This module
does no direction handling of its own on purpose: mixing normalization
concerns into the scoring function is exactly the kind of implicit coupling
that makes MCDM code hard to audit, and auditability is a stated design goal
of this project (see paper §3.4).
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from agentic_selection.data.preprocessing import CandidatePool


def _weights_to_array(
    weights: Mapping[str, float], attribute_cols: Sequence[str]
) -> np.ndarray:
    missing = [c for c in attribute_cols if c not in weights]
    if missing:
        raise ValueError(
            f"weights is missing entries for attribute columns: {missing}. "
            f"Every column in attribute_cols must have a weight."
        )
    w = np.array([float(weights[c]) for c in attribute_cols], dtype=float)
    if np.any(w < -1e-9):
        raise ValueError(f"weights must be non-negative, got {dict(zip(attribute_cols, w))}")
    total = w.sum()
    if total <= 0:
        raise ValueError("weights must sum to a positive number, got all-zero (or negative) weights")
    if not np.isclose(total, 1.0, atol=1e-3):
        # Renormalize rather than raise: callers (esp. the LLM-reasoning
        # module) are expected to hand this function whatever it produced,
        # and silent-but-logged renormalization is safer than a hard crash
        # mid-experiment. Strict validation belongs in agent/validation.py,
        # which runs *before* weights ever reach here.
        w = w / total
    return w


def weighted_sum(
    pool: CandidatePool,
    weights: Mapping[str, float],
) -> pd.Series:
    """Score every row of ``df`` as a weighted sum of ``attribute_cols``.

    Parameters
    ----------
    df : DataFrame
        Candidate pool. All columns in ``attribute_cols`` must be numeric
        and already normalized to [0, 1], benefit-oriented (higher=better).
    weights : mapping of attribute name -> weight
        Need not sum exactly to 1 (will be renormalized), but must be
        non-negative and cover every column in ``attribute_cols``.
    attribute_cols : sequence of str
        Which columns of ``df`` to score over.

    Returns
    -------
    pd.Series
        Score per row, indexed like ``df``. Higher is better. Not
        rescaled to [0, 1] as a set (individual weighted sums already
        live in [0, 1] given [0,1] inputs and weights summing to 1).
    """
    if len(pool.attribute_cols) == 0:
        raise ValueError("attribute_cols must be non-empty")
    for c in pool.attribute_cols:
        if c not in pool.df.columns:
            raise KeyError(f"attribute column '{c}' not found in df.columns={list(pool.df.columns)}")
    sub = pool.df.loc[:, list(pool.attribute_cols)]
    if sub.isnull().any().any():
        bad_cols = sub.columns[sub.isnull().any()].tolist()
        raise ValueError(
            f"weighted_sum received NaN values in columns {bad_cols}; "
            f"impute or drop before scoring (see data/preprocessing.py)."
        )
    w = _weights_to_array(weights, pool.attribute_cols)
    scores = sub.to_numpy(dtype=float) @ w
    return pd.Series(scores, index=pool.df.index, name="weighted_sum_score")


def rank_weighted_sum(
    pool: CandidatePool,
    weights: Mapping[str, float],
) -> pd.DataFrame:
    """Convenience wrapper: returns df sorted best-first with a score column."""
    scores = weighted_sum(pool, weights)
    out = pool.df.copy()
    out["score"] = scores
    return out.sort_values("score", ascending=False)
