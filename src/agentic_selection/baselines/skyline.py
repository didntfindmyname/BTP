"""Skyline (Pareto-optimal) filtering.

As with weighted_sum and topsis, this assumes attribute_cols are already
normalized to [0, 1] and benefit-oriented (higher = better) -- see
data/preprocessing.py. Candidate i dominates candidate j iff i is >= j on
every attribute in attribute_cols and strictly > j on at least one. The
skyline is the set of candidates dominated by nobody.

Memory note: a fully vectorized pairwise dominance check needs an
(n, n, m) boolean tensor, which is fine for the paper's protocol (pools
of n=20) but would use several hundred MB if run over the full QWS
dataset (n=2507) without chunking. ``skyline()`` therefore chunks over
the first axis; this changes nothing about the result, only peak memory.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from agentic_selection.baselines.topsis import topsis
from agentic_selection.data.preprocessing import CandidatePool


def skyline(
    pool: CandidatePool,
    chunk_size: int = 500,
) -> pd.Index:
    """Return the index labels of the non-dominated (skyline) rows of ``df``.

    Parameters
    ----------
    df : DataFrame
    attribute_cols : sequence of str
        Columns to compute dominance over. Assumed benefit-oriented [0, 1].
    chunk_size : int
        Rows processed per batch when computing pairwise dominance, to
        bound peak memory on large candidate pools. Does not affect the
        result, only performance/memory.

    Returns
    -------
    pd.Index
        Subset of df.index that is Pareto-optimal (non-dominated).
    """
    if len(pool.attribute_cols) == 0:
        raise ValueError("attribute_cols must be non-empty")
    for c in pool.attribute_cols:
        if c not in pool.df.columns:
            raise KeyError(f"attribute column '{c}' not found in df.columns={list(pool.df.columns)}")
    n = len(pool.df)
    if n == 0:
        return pool.df.index[:0]
    if n == 1:
        return pool.df.index

    X = pool.df.loc[:, list(pool.attribute_cols)].to_numpy(dtype=float)
    if np.isnan(X).any():
        bad_cols = pool.df.loc[:, list(pool.attribute_cols)].columns[
            pool.df.loc[:, list(pool.attribute_cols)].isnull().any()
        ].tolist()
        raise ValueError(
            f"skyline received NaN values in columns {bad_cols}; "
            f"impute or drop before filtering (see data/preprocessing.py)."
        )

    dominated = _skyline_numba(X)
    return pool.df.index[~dominated]

import numba

@numba.njit(fastmath=True)
def _skyline_numba(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    m = X.shape[1]
    dominated = np.zeros(n, dtype=numba.boolean)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(i + 1, n):
            if dominated[j]:
                continue
            i_ge_j = True
            j_ge_i = True
            for k in range(m):
                if X[i, k] < X[j, k]:
                    i_ge_j = False
                elif X[i, k] > X[j, k]:
                    j_ge_i = False
                if not i_ge_j and not j_ge_i:
                    break
            if i_ge_j and not j_ge_i:
                dominated[j] = True
            elif j_ge_i and not i_ge_j:
                dominated[i] = True
    return dominated


def skyline_then_topsis(
    pool: CandidatePool,
    weights: Mapping[str, float],
    chunk_size: int = 500,
) -> pd.Series:
    """Filter to the skyline set, then rank that subset with TOPSIS.

    Returns a full-length Series aligned to ``df.index`` so this function
    has the same contract as ``weighted_sum`` and ``topsis`` (higher =
    better, safe to argmax/sort directly), which matters because the
    agent's action module and the evaluation harness treat all three
    strategies uniformly. Non-skyline candidates are assigned a sentinel
    score of -1.0, which is guaranteed to be lower than any valid TOPSIS
    closeness coefficient (always in [0, 1]), so a plain ``argmax`` /
    ``sort_values(ascending=False)`` over the returned Series always
    prefers a skyline member when one exists.
    """
    sky_idx = skyline(pool, chunk_size=chunk_size)
    sky_pool = CandidatePool(pool.df.loc[sky_idx], pool.attribute_cols)
    sky_scores = topsis(sky_pool, weights)

    full = pd.Series(-1.0, index=pool.df.index, name="skyline_topsis_score")
    full.loc[sky_idx] = sky_scores
    return full


def rank_skyline_then_topsis(
    pool: CandidatePool,
    weights: Mapping[str, float],
) -> pd.DataFrame:
    scores = skyline_then_topsis(pool, weights)
    out = pool.df.copy()
    out["score"] = scores
    return out.sort_values("score", ascending=False)
