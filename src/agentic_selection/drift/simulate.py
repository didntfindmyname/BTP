"""Drift simulation (paper §3.8).

CORRECTION vs. the original project brief: the brief's plan was to use
"WS-DREAM's 64 time slices" directly for drift. As documented in
data/wsdream_loader.py, the dataset that actually has 64 time slices is
Dataset #2 (142 users x 4,500 services). The real archive has now been
downloaded and verified through the official Zenodo record. The original
controlled drift protocol remains the primary H2 experiment because it
provides a known degradation onset and a reproducible adaptation-lag target.

This module's PRIMARY, always-available path instead constructs
controlled, fully-documented drift scenarios on top of the real,
verified QWS (or WS-DREAM Dataset #1) static data: pick a service that
every condition being compared currently agrees is the top choice, then
apply an explicit, disclosed degradation curve to one or more of its
attributes over a sequence of synthetic "rounds", and measure how many
rounds it takes each method to stop recommending it. This is standard
practice for controlled drift-response evaluation (the alternative --
waiting for or hand-labeling a real drift event -- is not feasible on a
single-semester project timeline) and, importantly, it is fully
reproducible from a seed, unlike a dependency on an unverified 480MB
external download.

The complete secondary path in `scripts/09_run_wsdream_complete.py` evaluates
all 64 genuine Dataset #2 time slices. `wsdream2_to_drift_sequence` remains a
lower-level helper for custom temporal studies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from agentic_selection.data.preprocessing import CandidatePool


def find_common_top_choice(
    pool: CandidatePool,
    methods: Dict[str, Callable[[CandidatePool], pd.Series]],
) -> Optional[object]:
    """Find a service_id that is the #1 ranked choice under every method
    in `methods` (each a callable pool -> score Series, higher=better).
    Returns None if no such consensus candidate exists in this pool --
    callers should try a different seed/pool in that case rather than
    force a degradation target nothing actually agreed on.
    """
    top_choices = set()
    per_method_top = {}
    for name, fn in methods.items():
        scores = fn(pool)
        top_id = scores.sort_values(ascending=False).index[0]
        per_method_top[name] = top_id
        top_choices.add(top_id)
    if len(top_choices) == 1:
        return next(iter(top_choices))
    return None


def _degradation_curve(
    profile: str,
    n_rounds: int,
    degrade_start_round: int,
    magnitude: float,
) -> np.ndarray:
    """Returns an array of length n_rounds giving the multiplicative
    factor applied to the target attribute's ORIGINAL (pre-drift, benefit-
    oriented [0,1]) value at each round. 1.0 = unaffected, 0.0 = fully
    zeroed out. Rounds before degrade_start_round are always 1.0.
    """
    curve = np.ones(n_rounds)
    if profile == "sudden":
        curve[degrade_start_round:] = 1.0 - magnitude
    elif profile == "gradual":
        remaining = n_rounds - degrade_start_round
        ramp = np.linspace(0.0, magnitude, num=remaining)
        curve[degrade_start_round:] = 1.0 - ramp
    elif profile == "sudden_then_recover":
        recover_round = degrade_start_round + max(1, (n_rounds - degrade_start_round) // 2)
        curve[degrade_start_round:recover_round] = 1.0 - magnitude
        curve[recover_round:] = 1.0
    else:
        raise ValueError(f"unknown degradation profile: {profile!r}")
    return curve


@dataclass
class DriftSequence:
    pools: List[pd.DataFrame]  # one snapshot per round, same schema as input pool
    target_service_id: object
    degraded_attributes: List[str]
    degrade_start_round: int
    profile: str
    magnitude: float
    seed: int


def simulate_drift_sequence(
    pool: pd.DataFrame,
    attribute_cols: Sequence[str],
    target_service_id: object,
    degraded_attributes: Sequence[str],
    n_rounds: int = 20,
    degrade_start_round: int = 8,
    profile: str = "sudden",
    magnitude: float = 0.7,
    seed: int = 0,
) -> DriftSequence:
    """Produce n_rounds snapshots of `pool` where `target_service_id`'s
    `degraded_attributes` degrade starting at `degrade_start_round`
    according to `profile`, and every other cell (every other service,
    and this service's non-degraded attributes) is held fixed.

    `pool` must already be normalized/benefit-oriented [0,1] (see
    data/preprocessing.py) -- the degradation multiplies the *current*
    normalized value, so it remains in [0,1] throughout.
    """
    if target_service_id not in pool.index:
        raise KeyError(f"target_service_id {target_service_id!r} not found in pool index")
    for a in degraded_attributes:
        if a not in attribute_cols:
            raise ValueError(f"degraded attribute '{a}' not in attribute_cols")
    if not 0 < degrade_start_round < n_rounds:
        raise ValueError("degrade_start_round must be strictly between 0 and n_rounds")

    curve = _degradation_curve(profile, n_rounds, degrade_start_round, magnitude)
    original_values = {a: pool.loc[target_service_id, a] for a in degraded_attributes}

    pools = []
    for round_idx in range(n_rounds):
        snapshot = pool.copy()
        factor = curve[round_idx]
        for a in degraded_attributes:
            snapshot.loc[target_service_id, a] = original_values[a] * factor
        pools.append(snapshot)

    return DriftSequence(
        pools=pools,
        target_service_id=target_service_id,
        degraded_attributes=list(degraded_attributes),
        degrade_start_round=degrade_start_round,
        profile=profile,
        magnitude=magnitude,
        seed=seed,
    )


def wsdream2_to_drift_sequence(
    long_df: pd.DataFrame,
    service_ids: Sequence[int],
    value_col: str = "value",
    invert_as_cost: bool = True,
) -> Dict[int, pd.DataFrame]:
    """OPTIONAL upgrade path: build a real drift sequence from genuine
    WS-DREAM Dataset #2 data (see data/wsdream_loader.load_wsdream_dataset2_long).

    Returns a dict mapping user_id -> DataFrame of shape
    (n_time_slices, len(service_ids)), i.e. one real observed time series
    per user, for the requested services. This is intentionally a
    different (simpler, real-data) shape than DriftSequence above --
    consumed by evaluation code that wants genuine, not synthetically
    injected, drift. The complete runner uses a chunked aggregate
    representation; this helper remains available for custom long-format
    analyses.
    """
    sub = long_df[long_df["service_id"].isin(service_ids)]
    out: Dict[int, pd.DataFrame] = {}
    for user_id, group in sub.groupby("user_id"):
        pivoted = group.pivot(index="time_slice", columns="service_id", values=value_col)
        pivoted = pivoted.sort_index()
        out[user_id] = pivoted
    return out
