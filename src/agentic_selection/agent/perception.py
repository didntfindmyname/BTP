"""Perception module. Converts candidate pools into numerical LLM context summaries."""
# ============================== #
#       Perception Module        #
# ============================== #

# --- Imports ---
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Sequence, Protocol
import numpy as np
import pandas as pd
from agentic_selection.data.preprocessing import CandidatePool

# --- Data Structures ---
@dataclass
class PoolPerception:
    n_candidates: int
    attribute_cols: List[str]
    variance: Dict[str, float]
    pairwise_correlation: Dict[str, float]  # key "attrA|attrB" -> Pearson r
    conflict_score: float
    outlier_fraction: float
    missing_fraction: float
    prompt_text_override: str | None = None

    def to_vector(self) -> np.ndarray:
        """Fixed-length numeric feature vector for nearest-neighbor memory
        lookup. Length depends on len(attribute_cols); this is fine as
        long as a single deployment (one memory store) uses one
        consistent attribute schema, which is the intended use pattern.
        """
        var_part = np.array([self.variance[c] for c in self.attribute_cols], dtype=float)
        corr_part = np.array(list(self.pairwise_correlation.values()), dtype=float)
        scalar_part = np.array(
            [
                np.log1p(self.n_candidates),
                self.conflict_score,
                self.outlier_fraction,
                self.missing_fraction,
            ],
            dtype=float,
        )
        return np.concatenate([var_part, corr_part, scalar_part])

    def to_prompt_text(self) -> str:
        """Compact, human/LLM-readable rendering used inside the reasoning
        prompt (paper §3.6, perception summary component (b))."""
        if self.prompt_text_override is not None:
            return self.prompt_text_override
            
        var_str = ", ".join(f"{c}={v:.3f}" for c, v in self.variance.items())
        top_conflicts = sorted(
            self.pairwise_correlation.items(), key=lambda kv: kv[1]
        )[:3]
        conflict_str = (
            "; ".join(f"{k.replace('|', ' vs ')}: r={v:.2f}" for k, v in top_conflicts)
            if top_conflicts
            else "none observed"
        )
        return (
            f"Candidate pool size: {self.n_candidates}\n"
            f"Per-attribute variance (0-1 scale): {var_str}\n"
            f"Most negatively-correlated attribute pairs (potential "
            f"trade-offs): {conflict_str}\n"
            f"Overall conflict score: {self.conflict_score:.3f} "
            f"(higher = more genuine trade-offs between attributes)\n"
            f"Outlier fraction (rows with >=1 attribute outside 1.5xIQR): "
            f"{self.outlier_fraction:.3f}\n"
            f"Missing-value fraction: {self.missing_fraction:.3f}"
        )


def _outlier_row_fraction(df: pd.DataFrame, attribute_cols: Sequence[str]) -> float:
    if len(df) == 0:
        return 0.0
        
    sub = df[list(attribute_cols)]
    
    valid_counts = sub.count()
    valid_cols = valid_counts[valid_counts >= 4].index
    
    if len(valid_cols) == 0:
        return 0.0
        
    sub_valid = sub[valid_cols]
    q1 = sub_valid.quantile(0.25)
    q3 = sub_valid.quantile(0.75)
    iqr = q3 - q1
    
    valid_cols = iqr[iqr > 1e-12].index
    if len(valid_cols) == 0:
        return 0.0
        
    sub_valid = sub[valid_cols]
    q1 = q1[valid_cols]
    q3 = q3[valid_cols]
    iqr = iqr[valid_cols]
    
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    
    is_outlier = (sub_valid < lo) | (sub_valid > hi)
    return float(is_outlier.any(axis=1).mean())


def summarize_pool(
    pool: CandidatePool,
    missing_df: pd.DataFrame | None = None,
) -> PoolPerception:
    """Compute a PoolPerception summary for `pool`.

    Parameters
    ----------
    df : DataFrame
        The candidate pool, already normalized (used for variance /
        correlation / outlier computation -- these should be computed
        post-imputation so no NaNs reach downstream statistics).
    attribute_cols : sequence of str
    missing_df : DataFrame, optional
        Pass the PRE-imputation version of the pool here if you want
        missing_fraction to reflect genuine missingness rather than 0.0
        (by the time `df` reaches this function it is normally already
        imputed, per the data pipeline's contract with baselines/).
        If omitted, missing_fraction is computed from `df` itself.
    """
    attribute_cols = list(pool.attribute_cols)
    df = pool.df
    sub = df.loc[:, attribute_cols]

    variance = {c: float(sub[c].var(ddof=0)) if sub[c].notna().any() else 0.0 for c in attribute_cols}

    corr_matrix = sub.corr()
    
    pairwise_correlation: Dict[str, float] = {}
    for a, b in combinations(attribute_cols, 2):
        if sub[a].nunique() <= 1 or sub[b].nunique() <= 1:
            corr = 0.0
        else:
            corr = corr_matrix.loc[a, b]
            corr = 0.0 if pd.isna(corr) else float(corr)
        pairwise_correlation[f"{a}|{b}"] = corr

    negative_corrs = [v for v in pairwise_correlation.values() if v < 0]
    conflict_score = float(np.mean([-v for v in negative_corrs])) if negative_corrs else 0.0

    outlier_fraction = _outlier_row_fraction(sub, attribute_cols)

    ref = missing_df.loc[:, attribute_cols] if missing_df is not None else sub
    missing_fraction = float(ref.isnull().mean().mean()) if len(ref) else 0.0

    p = PoolPerception(
        n_candidates=len(df),
        attribute_cols=attribute_cols,
        variance=variance,
        pairwise_correlation=pairwise_correlation,
        conflict_score=conflict_score,
        outlier_fraction=outlier_fraction,
        missing_fraction=missing_fraction,
    )

    if len(df) == 0:
        p.prompt_text_override = "Candidate pool size: 0"
        return p
        
    lines = [f"Candidate pool size: {len(df)}", "Attribute bounds:"]
    for c in attribute_cols:
        if sub[c].notna().any():
            c_min, c_max = float(sub[c].min()), float(sub[c].max())
            lines.append(f"  - {c}: [{c_min:.3f}, {c_max:.3f}]")
        else:
            lines.append(f"  - {c}: all missing")
            
    p.prompt_text_override = "\n".join(lines)
    return p
