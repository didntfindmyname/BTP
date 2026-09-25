from __future__ import annotations
import concurrent.futures
from typing import Callable, Sequence

class ExperimentRunner:
    """Manages thread pooling and execution of independent trials."""
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers

    def execute(self, trials: Sequence[Callable[[], None]]) -> None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(t) for t in trials]
            for future in concurrent.futures.as_completed(futures):
                future.result()
