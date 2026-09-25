"""Agent Controller Module.

Orchestrates the Perceive -> Reason -> Act -> Remember loop for the agent.
"""
# ============================== #
#       Controller Module        #
# ============================== #

# --- Imports ---
from __future__ import annotations
import time
import sqlite3
import json
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Protocol
import pandas as pd
from agentic_selection.agent.llm_backends import LLMBackend
from agentic_selection.agent.memory import MemoryStore, format_digest, make_record
from agentic_selection.agent.perception import PoolPerception, summarize_pool
from agentic_selection.agent.reasoning import (
    DEFAULT_TOOL_MENU,
    LLMOutputParseError,
    ReasoningStrategy,
    DirectWeightReasoner,
)
from agentic_selection.agent.validation import ValidationResult, validate_agent_output
from agentic_selection.baselines import skyline_then_topsis, topsis, weighted_sum

from agentic_selection.data.preprocessing import CandidatePool

# --- Action Dispatch ---

ACTION_DISPATCH: Dict[str, Callable[[CandidatePool, dict], pd.Series]] = {
    "weighted_sum": weighted_sum,
    "topsis": topsis,
    "skyline_then_topsis": skyline_then_topsis,
}


@dataclass
class AgentDecision:
    record_id: str
    task_description: str
    perception: PoolPerception
    weights: Dict[str, float]
    strategy: str
    justification: str
    fallback_triggered: bool
    fallback_reason: Optional[str]
    ranking: pd.Series  # scores, best-first is NOT guaranteed here; see .top()
    latency_seconds: float
    api_calls: int
    raw_llm_text: Optional[str]
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def top(self, k: int = 1) -> pd.Index:
        return self.ranking.sort_values(ascending=False).index[:k]

    def top_service_id(self) -> object:
        return self.top(1)[0]





class SQLiteCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

        # A threading lock is required here to ensure multiple worker 
        # threads don't corrupt the DB during concurrent cache writes.
        self._lock = threading.Lock()
        conn = sqlite3.connect(self.path)
        try:
            conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, val TEXT)")
        finally:
            conn.close()
            
    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            conn = sqlite3.connect(self.path)
            try:
                cur = conn.execute("SELECT val FROM cache WHERE key = ?", (key,))
                row = cur.fetchone()
                return json.loads(row[0]) if row else None
            finally:
                conn.close()
                
    def set(self, key: str, val: dict) -> None:
        with self._lock:
            conn = sqlite3.connect(self.path)
            try:
                conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(val)))
                conn.commit()
            finally:
                conn.close()


class AgentController:
    """Stateful across calls only via its MemoryStore -- everything else
    (backend, attribute schema, tool menu) is fixed at construction time.
    """

    def __init__(
        self,
        backend: LLMBackend,
        attribute_cols: Sequence[str],
        memory_path,
        tool_menu: Sequence[str] = DEFAULT_TOOL_MENU,
        k_memory: int = 3,
        reasoning_strategy: Optional[ReasoningStrategy] = None,
    ):
        self.backend = backend
        self.attribute_cols = list(attribute_cols)
        self.memory = MemoryStore(memory_path)
        self.tool_menu = tuple(tool_menu)
        self.k_memory = k_memory
        self.reasoning_strategy = reasoning_strategy or DirectWeightReasoner()
        cache_path = Path(memory_path).parent / "llm_cache.db"
        self._llm_cache = SQLiteCache(cache_path)

    def decide(
        self,
        task_description: str,
        candidate_pool: CandidatePool,
        strategy_override: Optional[str] = None,
        use_memory: bool = True,
    ) -> AgentDecision:
        """Run one full perceive-reason-act-remember cycle.

        Parameters
        ----------
        strategy_override : str, optional
            If given, force this strategy regardless of what the LLM (or
            the fallback) chose, while still using the LLM-inferred
            weights. This implements the "Agent (weights only)" ablation
            condition in paper Table 4.2 -- pass strategy_override="topsis"
            to reproduce that row; leave it None for the "Agent (full)" row.
        use_memory : bool
            If False, skip both memory retrieval and logging this
            decision to memory -- useful for a memory-free ablation to
            isolate memory's marginal contribution (paper §6, "Discuss
            ... whether memory-informed decisions measurably outperformed
            memory-free ones").
        """
        original_index = candidate_pool.index
        perception = summarize_pool(candidate_pool)

        digest = ""
        if use_memory:
            neighbors = self.memory.nearest(perception.to_vector(), k=self.k_memory, query_task=task_description)
            digest = format_digest(neighbors)

        t0 = time.time()
        api_calls = 0
        prompt_tokens = 0
        completion_tokens = 0
        cache_key = json.dumps((task_description, perception.to_prompt_text(), digest))
        
        cached_result = self._llm_cache.get(cache_key)
        if cached_result is not None:
            parsed, raw_text, usage_dict = cached_result["parsed"], cached_result["raw_text"], cached_result["usage_dict"]
            prompt_tokens = usage_dict.get("prompt_tokens", 0)
            completion_tokens = usage_dict.get("completion_tokens", 0)
        else:
            try:
                parsed, raw_text, usage_dict = self.reasoning_strategy.decide(
                    self.backend, task_description, perception, self.attribute_cols, digest, self.tool_menu
                )
                api_calls = 1
                prompt_tokens = usage_dict.get("prompt_tokens", 0)
                completion_tokens = usage_dict.get("completion_tokens", 0)
                self._llm_cache.set(cache_key, {"parsed": parsed, "raw_text": raw_text, "usage_dict": usage_dict})
            except LLMOutputParseError as e:
                parsed, raw_text = None, e.raw_text
                api_calls = 1
                usage_dict = {"prompt_tokens": 0, "completion_tokens": 0}
                self._llm_cache.set(cache_key, {"parsed": parsed, "raw_text": raw_text, "usage_dict": usage_dict})
        latency = time.time() - t0

        validated: ValidationResult = validate_agent_output(
            parsed, task_description, self.attribute_cols, self.tool_menu
        )

        effective_strategy = strategy_override or validated.strategy
        action_fn = ACTION_DISPATCH.get(effective_strategy)
        if action_fn is None:
            raise ValueError(
                f"strategy '{effective_strategy}' is not in ACTION_DISPATCH "
                f"{list(ACTION_DISPATCH)}"
            )
        ranking = action_fn(candidate_pool, validated.weights)

        record = make_record(
            task_description=task_description,
            perception_vector=perception.to_vector(),
            weights=validated.weights,
            strategy=validated.strategy,  # log what the agent *chose*, even under override
            justification=validated.justification,
            fallback_triggered=validated.fallback_triggered,
            fallback_reason=validated.fallback_reason,
        )
        if use_memory:
            self.memory.append(record)

        if len(ranking) < len(original_index):
            padded_ranking = pd.Series(0.0, index=original_index)
            padded_ranking.update(ranking)
            ranking = padded_ranking

        return AgentDecision(
            record_id=record.record_id,
            task_description=task_description,
            perception=perception,
            weights=validated.weights,
            strategy=effective_strategy,
            justification=validated.justification,
            fallback_triggered=validated.fallback_triggered,
            fallback_reason=validated.fallback_reason,
            ranking=ranking,
            latency_seconds=latency,
            api_calls=api_calls,
            raw_llm_text=raw_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
