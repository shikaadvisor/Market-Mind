"""
utils/cost_tracker.py
======================
Risk 10 fix: Hard daily cost cap. Every LLM call is estimated before firing.
If the cap would be exceeded the pipeline refuses to run and logs a warning.

Usage:
    tracker = CostTracker(daily_cap_usd=20.0, log_dir=Path("logs"))
    with tracker.track("tier2", input_tokens=400, output_tokens=300):
        result = await t2_sim.run_async(...)
    print(tracker.summary())
"""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from utils.model_registry import estimate_cost

logger = logging.getLogger("marketmind.cost")


class BudgetExceeded(Exception):
    """Raised when a planned call would exceed the daily cap."""


@dataclass
class CallRecord:
    role:          str
    input_tokens:  int
    output_tokens: int
    cost_usd:      float
    timestamp:     str = field(default_factory=lambda: datetime.utcnow().isoformat())


class CostTracker:
    """
    Tracks estimated LLM spend per calendar day.
    Acts as a pre-flight guard: raises BudgetExceeded before any API call
    if the estimated cost would push today's total past daily_cap_usd.
    """

    def __init__(self, daily_cap_usd: float = 20.0, log_dir: Optional[Path] = None):
        self.daily_cap   = daily_cap_usd
        self.log_dir     = log_dir or Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self._log_path   = self.log_dir / "cost_log.jsonl"
        self._today      = date.today().isoformat()
        self._today_cost = self._load_today_cost()

    def _load_today_cost(self) -> float:
        if not self._log_path.exists():
            return 0.0
        today_total = 0.0
        with open(self._log_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("timestamp","")[:10] == self._today:
                        today_total += float(rec.get("cost_usd", 0))
                except Exception:
                    pass
        return today_total

    def _refresh_day(self) -> None:
        today = date.today().isoformat()
        if today != self._today:
            self._today      = today
            self._today_cost = 0.0

    def check(self, role: str, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost of a planned call. Raises BudgetExceeded if it would
        push today's total over the cap. Returns estimated cost in USD.
        """
        self._refresh_day()
        cost = estimate_cost(role, input_tokens, output_tokens)
        if self._today_cost + cost > self.daily_cap:
            msg = (f"Daily budget ${self.daily_cap:.2f} would be exceeded. "
                   f"Used: ${self._today_cost:.4f}  Planned: ${cost:.4f}")
            logger.error(msg)
            raise BudgetExceeded(msg)
        return cost

    def record(self, role: str, input_tokens: int, output_tokens: int) -> float:
        """Record an actual call (post-completion)."""
        self._refresh_day()
        cost = estimate_cost(role, input_tokens, output_tokens)
        self._today_cost += cost
        rec = CallRecord(role=role, input_tokens=input_tokens,
                         output_tokens=output_tokens, cost_usd=cost)
        with open(self._log_path, "a") as f:
            f.write(json.dumps({
                "role": rec.role, "input_tokens": rec.input_tokens,
                "output_tokens": rec.output_tokens, "cost_usd": round(cost, 6),
                "timestamp": rec.timestamp,
            }) + "\n")
        return cost

    @property
    def today_spent(self) -> float:
        self._refresh_day()
        return self._today_cost

    @property
    def today_remaining(self) -> float:
        return max(0.0, self.daily_cap - self.today_spent)

    def summary(self) -> str:
        return (f"Cost today: ${self.today_spent:.4f} / ${self.daily_cap:.2f}  "
                f"({self.today_remaining:.4f} remaining)")

    def estimate_run_cost(self, n_universes: int = 4, n_tier2: int = 50,
                          n_tier1: int = 3, n_desk_agents: int = 5) -> float:
        """
        Estimate total cost for one full pipeline run.
        Conservative: uses per-token estimates from model_registry.
        """
        t2_cost   = n_universes * n_tier2 * estimate_cost("tier2",  350, 80)
        t1_cost   = n_universes * n_tier1 * estimate_cost("tier1", 1200, 400)
        desk_cost = (
            estimate_cost("desk_macro",      900, 300) +
            estimate_cost("desk_technical",  900, 300) +
            estimate_cost("desk_sentiment", 1200, 400) +
            estimate_cost("desk_risk",       700, 200) +
            estimate_cost("desk_devil",      600, 200) +
            estimate_cost("coordinator",    2000, 600)
        )
        return round(t2_cost + t1_cost + desk_cost, 4)
