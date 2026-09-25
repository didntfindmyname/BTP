"""Normalization, cost-inversion, missing-value handling, and candidate-pool
sampling (paper §3.3).

Design decision (important for correctness of TOPSIS's ideal-point
semantics): normalization is computed **once, globally**, across the full
dataset being used (e.g. all 2,507 QWS rows), *before* any candidate pool
is sampled. Individual selection requests then operate on subsets of an
already-normalized dataframe. This means "how good is 0.8 throughput"
means the same thing across every experiment run on a given dataset,
while TOPSIS's ideal/anti-ideal points remain correctly instance-relative
(computed fresh from whichever candidate pool is passed to it) -- that
instance-relativity is inherent to what TOPSIS is, not something this
module tries to control.

Call order for a typical experiment:
    raw_df -> normalize_benefit_oriented(raw_df) -> normalized_df
    normalized_df -> sample_candidate_pool(normalized_df, n=20, seed=...) -> pool_df
    pool_df -> baselines.topsis(pool_df, weights, attribute_cols)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from agentic_selection.constants import COST_ATTRIBUTES, QWS_ATTRIBUTE_COLUMNS


def normalize_benefit_oriented(
    df: pd.DataFrame,
    attribute_cols: Sequence[str] = QWS_ATTRIBUTE_COLUMNS,
    cost_attributes: Iterable[str] = COST_ATTRIBUTES,
) -> pd.DataFrame:
    """Min-max normalize attribute_cols to [0, 1] and invert cost attributes.

    After this call every column in ``attribute_cols`` is "higher = better"
    uniformly, which is the precondition every function in
    ``agentic_selection.baselines`` assumes.

    A column with zero range across the whole dataset (every row has the
    same value) carries no discriminative signal; rather than divide by
    zero, such a column is set to a neutral constant (0.5) for every row.
    This is logged so it doesn't silently pass unnoticed.
    """
    cost_attributes = set(cost_attributes)
    out = df.copy()
    degenerate_cols = []
    for col in attribute_cols:
        if col not in out.columns:
            raise KeyError(f"attribute column '{col}' not found in df.columns={list(out.columns)}")
        col_min = out[col].min(skipna=True)
        col_max = out[col].max(skipna=True)
        rng = col_max - col_min
        if pd.isna(rng) or rng <= 1e-12:
            degenerate_cols.append(col)
            normalized = pd.Series(0.5, index=out.index)
            # Preserve NaNs where the raw value was missing -- normalization
            # should not manufacture data that imputation hasn't decided on.
            normalized = normalized.where(out[col].notna(), other=np.nan)
        else:
            normalized = (out[col] - col_min) / rng
        if col in cost_attributes:
            normalized = 1.0 - normalized
        out[col] = normalized
    if degenerate_cols:
        import warnings

        warnings.warn(
            f"normalize_benefit_oriented: columns with zero variance across "
            f"the dataset were set to a neutral 0.5 (no discriminative "
            f"signal): {degenerate_cols}",
            stacklevel=2,
        )
    return out


def impute_missing(
    df: pd.DataFrame,
    attribute_cols: Sequence[str] = QWS_ATTRIBUTE_COLUMNS,
    strategy: str = "column_mean",
) -> pd.DataFrame:
    """Fill missing QoS values. Call this BEFORE normalize_benefit_oriented
    if you want the imputed values to influence the global min/max, or
    AFTER if you want normalization computed only from observed values.
    Both are defensible; this project imputes after normalizing, on the
    already-[0,1]-scaled columns, so a fixed strategy (e.g. "column_mean")
    has a stable, interpretable meaning (0.5 = "prior belief was neutral"
    style, rather than depending on outlier-sensitive raw units).
    """
    out = df.copy()
    for col in attribute_cols:
        if out[col].isnull().any():
            if strategy == "column_mean":
                fill_val = out[col].mean(skipna=True)
            elif strategy == "column_median":
                fill_val = out[col].median(skipna=True)
            elif strategy == "neutral":
                fill_val = 0.5
            else:
                raise ValueError(f"unknown imputation strategy: {strategy}")
            out[col] = out[col].fillna(fill_val)
    return out


def sample_candidate_pool(
    df: pd.DataFrame,
    n: int,
    seed: int,
    replace: bool = False,
) -> pd.DataFrame:
    """Draw a reproducible random subset of n rows, instantiating one
    selection request for controlled experimentation (paper §3.3).
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not replace and n > len(df):
        raise ValueError(
            f"Cannot sample n={n} without replacement from a pool of "
            f"size {len(df)}. Pass replace=True or reduce n."
        )
    sampled_df = df.sample(n=n, random_state=seed, replace=replace).copy()
    # Note: caller should wrap in CandidatePool if needed, or sample_candidate_pool can return it.
    # To avoid circularity we just return df, and caller wraps it.
    return sampled_df

@dataclass
class CandidatePool:
    df: pd.DataFrame
    attribute_cols: Sequence[str]
    
    def to_numpy(self) -> np.ndarray:
        return self.df.loc[:, list(self.attribute_cols)].to_numpy(dtype=float)
        
    def __len__(self) -> int:
        return len(self.df)
        
    @property
    def index(self):
        return self.df.index


def inject_missingness(
    df: pd.DataFrame,
    attribute_cols: Sequence[str],
    fraction: float,
    seed: int,
) -> pd.DataFrame:
    """OPTIONAL / exploratory utility, not part of the core H1-H4 protocol
    in paper §4. Randomly masks a `fraction` of (row, attribute) cells to
    NaN, useful if you want to extend the evaluation with a robustness-to-
    incomplete-data condition (this was discussed as a candidate metric
    during project scoping but was not included in the final pre-registered
    hypotheses -- see paper §4.4). Provided here so that extension is a
    one-line addition rather than a rewrite, should you choose to pursue it.
    """
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"fraction must be in [0, 1), got {fraction}")
    rng = np.random.default_rng(seed)
    out = df.copy()
    n_rows = len(out)
    for col in attribute_cols:
        mask = rng.random(n_rows) < fraction
        out.loc[out.index[mask], col] = np.nan
    return out
