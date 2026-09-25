"""Evaluation protocol (paper §4.2, §4.3): orchestrates the four
conditions across task profiles, candidate pools, and (separately) the
drift sequence.

Practical note this module is built around: the full stable-condition
protocol as specified (6 profiles + ~8 held-out tasks, 30 pools each,
2 of the 4 conditions requiring a real LLM call) is on the order of
several hundred to ~1,000 LLM calls. That is real money and real wall-
clock time, and a network hiccup partway through a long run is a normal
thing to happen, not an edge case. Every run function in this module
therefore writes each completed trial to disk immediately (append-only
CSV) and, on startup, skips any (profile_key, pool_seed, condition)
combination already present in the output file. Killing the process and
re-running the same command resumes rather than restarts.
"""
from __future__ import annotations

import csv
import json
import concurrent.futures
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from agentic_selection.agent.controller import AgentController
from agentic_selection.baselines import GLOBAL_FIXED_WEIGHTS, TASK_LOOKUP_TABLE, topsis
from agentic_selection.baselines.lookup_table import get_lookup_weights
from agentic_selection.data.preprocessing import CandidatePool, sample_candidate_pool
from agentic_selection.drift.simulate import find_common_top_choice, simulate_drift_sequence
from agentic_selection.evaluation.metrics import adaptation_lag, regret
from agentic_selection.evaluation.runner import ExperimentRunner
from agentic_selection.evaluation.storage import ExperimentStorage
from agentic_selection.tasks import HELD_OUT_TASKS, TASK_PROFILES, HeldOutTask, TaskProfile

STABLE_CONDITIONS = ("global_fixed", "lookup_table", "agent_weights_only", "agent_full")
STABLE_RESULT_FIELDS = [
    "task_key",
    "task_kind",  # "profile" | "held_out"
    "task_description",
    "pool_seed",
    "condition",
    "regret",
    "fallback_triggered",
    "latency_seconds",
    "api_calls",
    "top_service_id",
    "strategy",
    "weights_json",
    "is_synthetic_data",
    "prompt_tokens",
    "completion_tokens",
]


def run_stable_protocol(
    normalized_df: pd.DataFrame,
    attribute_cols: Sequence[str],
    storage: ExperimentStorage,
    agent_controller: AgentController,
    n_pools: int = 30,
    pool_size: int = 20,
    base_seed: int = 1000,
    is_synthetic_data: bool = False,
    task_profiles: Sequence[TaskProfile] = TASK_PROFILES,
    held_out_tasks: Sequence[HeldOutTask] = HELD_OUT_TASKS,
    conditions: Sequence[str] = STABLE_CONDITIONS,
    global_fixed_weights: Optional[Dict[str, float]] = None,
    lookup_table: Optional[Dict[str, Dict[str, float]]] = None,
    lookup_weights_fn: Optional[Callable[[str], Dict[str, float]]] = None,
) -> pd.DataFrame:
    """Run the stable-condition protocol (paper §4.3) over every task
    profile plus the held-out set, `n_pools` independent pools each,
    across `conditions`. Resumable: re-running with the same
    `output_csv` skips trials already recorded there.

    `global_fixed_weights`, `lookup_table`, and `lookup_weights_fn` default
    to the QWS 9-attribute versions (baselines.GLOBAL_FIXED_WEIGHTS,
    baselines.TASK_LOOKUP_TABLE, baselines.get_lookup_weights) if not
    given -- this function is otherwise dataset-agnostic (it only touches
    `normalized_df` through `attribute_cols` and `sample_candidate_pool`),
    so passing the WS-DREAM-specific versions
    (baselines.wsdream_lookup_table) along with `attribute_cols=
    WSDREAM_ATTRIBUTE_COLUMNS` and `task_profiles=WSDREAM_TASK_PROFILES`
    runs the exact same protocol against WS-DREAM Dataset #1 instead of
    QWS -- see scripts/07_run_wsdream_validation.py.

    Returns the full accumulated results (including any prior runs
    already in output_csv) as a DataFrame.
    """
    global_fixed_weights = global_fixed_weights if global_fixed_weights is not None else GLOBAL_FIXED_WEIGHTS
    lookup_table = lookup_table if lookup_table is not None else TASK_LOOKUP_TABLE
    lookup_weights_fn = lookup_weights_fn if lookup_weights_fn is not None else get_lookup_weights

    all_tasks: List[Tuple[str, str, str]] = [
        (p.key, "profile", p.description) for p in task_profiles
    ] + [(f"held_out_{i}", "held_out", t.description) for i, t in enumerate(held_out_tasks)]

    reference_weights_by_key: Dict[str, dict] = {p.key: lookup_table[p.key] for p in task_profiles}
    for i, t in enumerate(held_out_tasks):
        reference_weights_by_key[f"held_out_{i}"] = lookup_table[t.nearest_profile_key]

    def _run_single_condition(task_key, task_kind, task_description, seed, condition, ref_weights):
        if not storage.should_run({"task_key": task_key, "pool_seed": seed, "condition": condition}):
            return

        pool_df = sample_candidate_pool(normalized_df, n=pool_size, seed=seed)
        pool = CandidatePool(pool_df, attribute_cols)

        if condition == "global_fixed":
            scores = topsis(pool, global_fixed_weights)
            fallback, latency, api_calls = False, 0.0, 0
            top_id = scores.sort_values(ascending=False).index[0]
            strategy = "topsis"
            w_json = json.dumps(global_fixed_weights)
            prompt_tokens, completion_tokens = 0, 0
        elif condition == "lookup_table":
            w = lookup_weights_fn(task_description if task_kind == "held_out" else task_key)
            scores = topsis(pool, w)
            fallback, latency, api_calls = False, 0.0, 0
            top_id = scores.sort_values(ascending=False).index[0]
            strategy = "topsis"
            w_json = json.dumps(w)
            prompt_tokens, completion_tokens = 0, 0
        elif condition in ("agent_weights_only", "agent_full"):
            strategy_override = "topsis" if condition == "agent_weights_only" else None
            decision = agent_controller.decide(
                task_description, pool, strategy_override=strategy_override
            )
            scores = decision.ranking
            fallback = decision.fallback_triggered
            latency = decision.latency_seconds
            api_calls = decision.api_calls
            top_id = decision.top_service_id()
            strategy = decision.strategy
            w_json = json.dumps(decision.weights)
            prompt_tokens = decision.prompt_tokens
            completion_tokens = decision.completion_tokens
        else:
            raise ValueError(f"unknown condition: {condition}")

        r = regret(pool.df, scores, ref_weights, attribute_cols)
        row = {
            "task_key": task_key,
            "task_kind": task_kind,
            "task_description": task_description,
            "pool_seed": seed,
            "condition": condition,
            "regret": r,
            "fallback_triggered": fallback,
            "latency_seconds": latency,
            "api_calls": api_calls,
            "top_service_id": top_id,
            "strategy": strategy,
            "weights_json": w_json,
            "is_synthetic_data": is_synthetic_data,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        storage.record(row)

    trials = []
    for task_key, task_kind, task_description in all_tasks:
        ref_weights = reference_weights_by_key[task_key]
        for pool_i in range(n_pools):
            seed = base_seed + pool_i
            for condition in conditions:
                trials.append(
                    lambda tk=task_key, tkind=task_kind, td=task_description, s=seed, c=condition, rw=ref_weights: _run_single_condition(tk, tkind, td, s, c, rw)
                )

    ExperimentRunner(max_workers=10).execute(trials)
    return storage.load_all()


DRIFT_CONDITIONS = ("global_fixed", "lookup_table", "agent_weights_only", "agent_full")
DRIFT_RESULT_FIELDS = [
    "task_key",
    "trial_seed",
    "condition",
    "target_service_id",
    "degrade_start_round",
    "n_rounds",
    "profile",
    "lag_rounds",
    "censored",
    "trace_json",
    "is_synthetic_data",
    "n_switches",
]


def run_drift_protocol(
    normalized_df: pd.DataFrame,
    attribute_cols: Sequence[str],
    storage: ExperimentStorage,
    agent_controller: AgentController,
    n_trials: int = 5,
    pool_size: int = 20,
    n_rounds: int = 16,
    degrade_start_round: int = 6,
    degradation_profile: str = "sudden",
    degradation_magnitude: float = 0.75,
    static_reevaluation_period: int = 5,
    max_seed_attempts_for_consensus: int = 25,
    base_seed: int = 5000,
    is_synthetic_data: bool = False,
    task_profiles: Sequence[TaskProfile] = TASK_PROFILES,
    degraded_attributes_by_profile: Dict[str, List[str]] | None = None,
    conditions: Sequence[str] = DRIFT_CONDITIONS,
) -> pd.DataFrame:
    """Run the drift experiment (paper §3.8, §4.3) over every task
    profile, `n_trials` independent drift scenarios each. Resumable like
    run_stable_protocol.

    For each trial: sample a pool, find a service both the global-fixed
    and lookup-table baselines currently agree is the #1 choice (trying
    up to `max_seed_attempts_for_consensus` seeds -- not every random
    pool has a clean consensus candidate, which is expected and simply
    skipped), degrade that service's dominant attribute(s) for this
    profile starting at `degrade_start_round`, then measure each
    condition's adaptation lag. Static conditions (global_fixed,
    lookup_table) are re-evaluated only every `static_reevaluation_period`
    rounds; agent conditions are re-evaluated every round -- see
    evaluation/metrics.py's module docstring for why this asymmetry is
    the correct, non-trivial way to operationalize this comparison.
    """
    def _run_single_drift_trial(profile, trial_i, default_degraded):
        task_key = profile.key
        trial_seed = base_seed + trial_i

        if all(not storage.should_run({"task_key": task_key, "trial_seed": trial_seed, "condition": c}) for c in conditions):
            return

        pool = None
        target = None
        for attempt in range(max_seed_attempts_for_consensus):
            candidate_seed = trial_seed * 1000 + attempt
            pool_df = sample_candidate_pool(normalized_df, n=pool_size, seed=candidate_seed)
            candidate_pool = CandidatePool(pool_df, attribute_cols)
            methods = {
                "global_fixed": lambda p: topsis(p, GLOBAL_FIXED_WEIGHTS),
                "lookup_table": lambda p: topsis(p, TASK_LOOKUP_TABLE[profile.key]),
            }
            t = find_common_top_choice(candidate_pool, methods)
            if t is not None:
                pool, target = candidate_pool, t
                break
        if pool is None:
            print(
                f"[drift] profile={profile.key} trial={trial_i}: no consensus "
                f"target found in {max_seed_attempts_for_consensus} seed attempts, skipping"
            )
            return

        seq = simulate_drift_sequence(
            pool.df,
            attribute_cols,
            target_service_id=target,
            degraded_attributes=default_degraded,
            n_rounds=n_rounds,
            degrade_start_round=degrade_start_round,
            profile=degradation_profile,
            magnitude=degradation_magnitude,
            seed=trial_seed,
        )

        for condition in conditions:
            if not storage.should_run({"task_key": task_key, "trial_seed": trial_seed, "condition": condition}):
                continue

            if condition == "global_fixed":
                decide_fn = lambda p: topsis(CandidatePool(p, attribute_cols), GLOBAL_FIXED_WEIGHTS).sort_values(ascending=False).index[0]
                reeval = static_reevaluation_period
            elif condition == "lookup_table":
                w = TASK_LOOKUP_TABLE[profile.key]
                decide_fn = lambda p, w=w: topsis(CandidatePool(p, attribute_cols), w).sort_values(ascending=False).index[0]
                reeval = static_reevaluation_period
            elif condition in ("agent_weights_only", "agent_full", "rag_agent"):
                strategy_override = "topsis" if condition == "agent_weights_only" else None
                ctrl = rag_controller if condition == "rag_agent" and rag_controller else agent_controller
                decide_fn = lambda p, so=strategy_override, c=ctrl: c.decide(
                    profile.description, CandidatePool(p, attribute_cols), strategy_override=so
                ).top_service_id()
                reeval = None
            else:
                raise ValueError(f"unknown condition: {condition}")

            result = adaptation_lag(
                seq.pools, target, degrade_start_round, decide_fn, reevaluation_period=reeval
            )

            trace = result.active_recommendation_trace
            n_switches = sum(1 for i in range(1, len(trace)) if trace[i] != trace[i-1]) if trace else 0
            
            row = {
                "task_key": task_key,
                "trial_seed": trial_seed,
                "condition": condition,
                "target_service_id": target,
                "degrade_start_round": degrade_start_round,
                "n_rounds": n_rounds,
                "profile": degradation_profile,
                "lag_rounds": result.lag_rounds if result.lag_rounds is not None else "",
                "censored": result.censored,
                "trace_json": str(trace),
                "is_synthetic_data": is_synthetic_data,
                "n_switches": n_switches,
            }
            storage.record(row)

    trials = []
    for profile in task_profiles:
        default_degraded = (degraded_attributes_by_profile or {}).get(
            profile.key, profile.dominant_attributes[:2] or profile.dominant_attributes
        )
        for trial_i in range(n_trials):
            trials.append(
                lambda p=profile, t_i=trial_i, d=default_degraded: _run_single_drift_trial(p, t_i, d)
            )

    ExperimentRunner(max_workers=10).execute(trials)
    return storage.load_all()
