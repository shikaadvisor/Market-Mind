"""
desk/agents/base.py
===================
Shared infrastructure for all trading desk agents.

Defines:
- DeskState: the LangGraph state that flows through the desk graph
- AgentOutput: typed output each agent writes to state
- BaseAgent: common async call pattern, mock support, error handling
- Prompt utilities shared across agents
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, TypedDict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# LangGraph state — the shared blackboard all agents read and write
# ---------------------------------------------------------------------------

class DeskState(TypedDict, total=False):
    """
    The state object that flows through the LangGraph desk graph.
    Each agent reads what it needs and writes its contribution.
    The coordinator reads everything and produces the final signal.

    TypedDict with total=False means all fields are optional at any
    point in the graph — nodes only see what has been written so far.
    """

    # ── Inputs (set by pipeline before graph runs) ────────────────────────
    instrument:          str
    exchange:            str
    asset_class:         str
    point_in_time_date:  str        # ISO date string — enforces forward-only
    market_data:         dict       # OHLCV, OI, volume, technical levels
    simulation_brief:    str        # SimulationReport.sentiment_interpreter_brief
    simulation_metrics:  dict       # raw SimulationReport metrics dict
    event_description:   str        # the seed event string

    # ── Agent outputs (written by each agent node) ────────────────────────
    macro_view:          dict       # MacroAgent output
    technical_view:      dict       # TechnicalAgent output
    risk_params:         dict       # RiskAgent output — sole writer of levels
    sentiment_view:      dict       # SentimentAgent output

    # ── Final output (written by coordinator) ─────────────────────────────
    final_signal:        dict       # raw dict for TradingSignal(**final_signal)
    error:               str        # any fatal error that aborted the graph


# ---------------------------------------------------------------------------
# Typed output structure each agent writes
# ---------------------------------------------------------------------------

@dataclass
class MacroOutput:
    regime:           str           # RegimeType value
    direction:        str           # "long" | "short" | "neutral"
    conviction:       float         # 0–1
    macro_context:    str           # narrative (min 80 chars)
    key_factors:      list[str]     # 1–4 macro drivers
    fii_outlook:      str           # "inflow" | "outflow" | "neutral"
    rate_environment: str           # "tightening" | "easing" | "on_hold"
    warning_flags:    list[str]     # macro risks to flag
    raw:              str           # raw LLM response


@dataclass
class TechnicalOutput:
    direction:        str           # "long" | "short" | "neutral"
    conviction:       float
    technical_view:   str           # narrative (min 80 chars)
    entry_low:        float         # entry zone bottom
    entry_high:       float         # entry zone top
    stop_loss:        float         # hard stop
    target_1:         float         # first target
    target_2:         float | None  # second target (optional)
    key_levels:       list[str]     # support/resistance being watched
    pattern:          str           # chart pattern or structure
    raw:              str


@dataclass
class RiskOutput:
    approved:             bool          # False = do_not_trade veto
    max_position_pct:     float         # 0–1, e.g. 0.04 = 4%
    invalidation_level:   float         # price where thesis is dead
    do_not_trade_flags:   list[str]     # hard vetoes
    warning_flags:        list[str]     # caution flags (non-veto)
    risk_reward:          float         # computed R:R
    conviction_modifier:  float         # -0.3 to 0.0 — risk can only REDUCE conviction
    rationale:            str
    raw:                  str


@dataclass
class SentimentOutput:
    crowd_direction:      str           # "bullish" | "bearish" | "neutral" | "mixed"
    crowd_conviction:     float         # how strongly to weight the sim
    simulation_weight:    float         # 0.10–0.30 recommended weight
    crowd_narrative:      str           # translated crowd story (min 80 chars)
    key_crowd_insight:    str           # the single most actionable crowd signal
    contrarian_signal:    bool          # True if crowd and institutions diverge strongly
    raw:                  str


# ---------------------------------------------------------------------------
# Shared prompt utilities
# ---------------------------------------------------------------------------

POINT_IN_TIME_INSTRUCTION = (
    "CRITICAL: You are reasoning as of {date} only. "
    "You have no knowledge of market events after this date. "
    "Do not reference any prices, events, or outcomes beyond {date}."
)


def build_market_context(state: DeskState) -> str:
    """Format market data dict into readable prompt context."""
    md = state.get("market_data", {})
    if not md:
        return "Market data: not provided — use your knowledge of current market conditions."

    lines = ["Current market snapshot:"]
    for k, v in md.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def clean_json(raw: str) -> str:
    """Strip markdown fences if the model disobeys the 'no fences' instruction."""
    return raw.replace("```json", "").replace("```", "").strip()


# ---------------------------------------------------------------------------
# Base async agent
# ---------------------------------------------------------------------------

class BaseAgent:
    """
    Common async call pattern for all desk agents.
    Handles: client creation, mock injection, retry on parse error,
    latency logging, fallback on total failure.
    """

    def __init__(
        self,
        model:       str,
        temperature: float = 0.2,
        max_tokens:  int   = 1500,
        mock:        bool  = False,
        mock_client: Any   = None,
    ):
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.mock        = mock
        self._mock_client = mock_client

    def _get_client(self, api_key: str, provider: str = "openai") -> Any:
        if self.mock and self._mock_client:
            return self._mock_client

        if provider == "openai":
            try:
                from openai import AsyncOpenAI
                return AsyncOpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("pip install openai")

        if provider == "anthropic":
            try:
                from anthropic import AsyncAnthropic
                return AsyncAnthropic(api_key=api_key)
            except ImportError:
                raise ImportError("pip install anthropic")

        raise ValueError(f"Unknown provider: {provider}")

    async def _call_openai(
        self,
        client:        Any,
        system_prompt: str,
        user_prompt:   str,
    ) -> str:
        response = await client.chat.completions.create(
            model       = self.model,
            temperature = self.temperature,
            max_tokens  = self.max_tokens,
            messages    = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    async def _call_anthropic(
        self,
        client:        Any,
        system_prompt: str,
        user_prompt:   str,
    ) -> str:
        response = await client.messages.create(
            model      = self.model,
            max_tokens = self.max_tokens,
            system     = system_prompt,
            messages   = [{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text.strip()

    def _parse_float(self, v: Any, default: float = 0.5) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return default

    def _parse_positive_float(self, v: Any, default: float = 0.0) -> float:
        try:
            result = float(v)
            return result if result > 0 else default
        except (TypeError, ValueError):
            return default
