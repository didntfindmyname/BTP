"""TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution).

Implementation note (paper §3.3-3.4): classical TOPSIS normalizes the raw
decision matrix itself via vector normalization (x_ij / sqrt(sum_i x_ij^2))
and separately tracks which attributes are cost- vs benefit-type when
picking the ideal/anti-ideal points. This project instead performs
min-max normalization *and* cost-inversion once, up front, in
``data.preprocessing.normalize_benefit_oriented``, so that every attribute
reaching this function is already in [0, 1] with higher = better. Given
that precondition, the ideal point A+ is simply the per-column max of the
*weighted* matrix and the anti-ideal A- is the per-column min -- there is
no remaining cost/benefit branching to do here. This is a standard,
disclosed simplification (sometimes called "TOPSIS on a pre-normalized
matrix") and is validated against a hand-computed example in
tests/test_topsis.py so the arithmetic itself is not in question.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from agentic_selection.data.preprocessing import CandidatePool

from agentic_selection.baselines.weighted_sum import _weights_to_array


def topsis(
    pool: CandidatePool,
    weights: Mapping[str, float],
) -> pd.Series:
    """Compute TOPSIS closeness coefficients for every row of ``df``.

    Returns
    -------
    pd.Series
        Closeness coefficient CC_i = d(i, A-) / (d(i, A+) + d(i, A-)) in
        [0, 1], indexed like ``df``. Higher is better (closer to the ideal
        point, farther from the anti-ideal point).
    """
    if len(pool.attribute_cols) == 0:
        raise ValueError("attribute_cols must be non-empty")
    for c in pool.attribute_cols:
        if c not in pool.df.columns:
            raise KeyError(f"attribute column '{c}' not found in df.columns={list(pool.df.columns)}")
    if len(pool.df) == 0:
        return pd.Series([], dtype=float, name="topsis_closeness")

    sub = pool.df.loc[:, list(pool.attribute_cols)]
    if sub.isnull().any().any():
        bad_cols = sub.columns[sub.isnull().any()].tolist()
        raise ValueError(
            f"topsis received NaN values in columns {bad_cols}; "
            f"impute or drop before scoring (see data/preprocessing.py)."
        )

    w = _weights_to_array(weights, pool.attribute_cols)
    X = sub.to_numpy(dtype=float)
    V = X * w  # weighted normalized matrix, shape (n, m)

    if len(pool.df) == 1:
        # A single candidate is trivially both the ideal and anti-ideal
        # point along every attribute it doesn't share with anyone else;
        # define closeness as 1.0 (it is, uncontestedly, the best of one).
        return pd.Series([1.0], index=pool.df.index, name="topsis_closeness")

    a_plus = V.max(axis=0)
    a_minus = V.min(axis=0)

    d_plus = np.sqrt(((V - a_plus) ** 2).sum(axis=1))
    d_minus = np.sqrt(((V - a_minus) ** 2).sum(axis=1))

    denom = d_plus + d_minus
    # If a candidate is simultaneously the ideal and anti-ideal on every
    # weighted attribute (only possible when every candidate is identical
    # on all attributes that have nonzero weight), d_plus == d_minus == 0.
    # Treat this as a tie at maximal closeness for all such rows rather
    # than propagating a NaN from 0/0.
    closeness = np.where(denom > 1e-12, d_minus / np.where(denom > 1e-12, denom, 1.0), 1.0)

    return pd.Series(closeness, index=pool.df.index, name="topsis_closeness")


def rank_topsis(
    pool: CandidatePool,
    weights: Mapping[str, float],
) -> pd.DataFrame:
    """Convenience wrapper: returns df sorted best-first with a score column."""
    scores = topsis(pool, weights)
    out = pool.df.copy()
    out["score"] = scores
    return out.sort_values("score", ascending=False)
