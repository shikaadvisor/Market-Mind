"""
desk/agents/technical.py
========================
The Technical Analyst — runs parallel to the Macro Strategist.

Reads price action, structure, and key levels to produce:
- A directional view with entry zone (low/high, not a single price)
- Stop loss and target levels
- Pattern/structure identification
- Key levels the desk should watch

This agent owns the price levels. The Risk Manager will receive these
and apply risk rules on top — but the raw levels come from here.

Model: GPT-4o
Runs: in parallel with Macro Strategist
Writes to state: technical_view (TechnicalOutput)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from desk.agents.base import (
    BaseAgent, DeskState, TechnicalOutput,
    POINT_IN_TIME_INSTRUCTION, build_market_context, clean_json,
)


SYSTEM_PROMPT = """You are the Technical Analyst on a sophisticated trading desk.
Your role: analyse price structure, key levels, and chart patterns to produce
actionable entry zones, stops, and targets.

Your analytical framework:
- Market structure: higher highs/lows (uptrend), lower highs/lows (downtrend), range
- Key levels: support/resistance, previous swing highs/lows, round numbers, VWAP
- Volume: confirm breakouts/breakdowns, spot divergences
- Momentum: RSI, MACD state (not exact values — directional read)
- Patterns: flags, wedges, head and shoulders, base formations
- Open interest and futures positioning for index/futures instruments

Entry zone philosophy:
- Never give a single entry price — always a zone (low to high)
- Zone width should reflect the instrument's typical volatility:
  * Equity indices: 0.2–0.6% of price  (Nifty, SPX, DAX)
  * Energy futures: 0.8–2.5% of price  (CL, BZ, NG)
  * Precious metals: 0.3–1.0% of price (GC, SI)
  * Base metals: 0.8–2.0% of price     (HG, AL)
  * Agriculture: 1.0–3.0% of price     (ZW, ZC, ZS)
  * FX majors: 0.15–0.40% of price     (EURUSD, GBPUSD)
  * EM FX: 0.20–0.60% of price         (USDINR, USDMXN)
  * Rates/bonds: 0.10–0.30% of price   (ZN, ZB)
  * Crypto: 1.5–5.0% of price          (BTCUSD, ETHUSD)
- Stop must be beyond a clear structure level — not arbitrary
- For commodities, use contract price units (e.g. $/barrel, cents/bushel)

Output requirements:
- All price levels must be realistic for THIS SPECIFIC INSTRUMENT and its units
- Entry zone width must reflect current volatility (use ATR context if available)
- Risk/reward to target_1 must be >= 1.5
- Be specific about pattern/structure, including commodity-specific signals
  (e.g. "contango structure breaking", "failed test of key moving average")

Respond ONLY with valid JSON. No preamble. No markdown fences."""


def _build_user_prompt(state: DeskState) -> str:
    pit   = state.get("point_in_time_date", str(__import__("datetime").date.today()))
    event = state.get("event_description", "General technical analysis")
    mctx  = build_market_context(state)
    macro = state.get("macro_view", {})
    macro_dir = macro.get("direction", "neutral") if macro else "neutral"

    return f"""{POINT_IN_TIME_INSTRUCTION.format(date=pit)}

Instrument: {state.get('instrument', 'NIFTY50')} ({state.get('exchange', 'NSE')})
Event/catalyst: {event}
Macro desk view: {macro_dir} (use as context, not constraint)

{mctx}

Perform technical analysis and provide your assessment.
Respond ONLY with this JSON structure:
{{
  "direction": "long" | "short" | "neutral",
  "conviction": 0.0-1.0,
  "technical_view": "your technical narrative — structure, pattern, key levels, and why this is the setup (80-200 words)",
  "entry_low": <float — lower bound of entry zone>,
  "entry_high": <float — upper bound of entry zone, must be > entry_low>,
  "stop_loss": <float — hard stop, must be on the other side of entry zone>,
  "target_1": <float — first target, R:R to here must be >= 1.5>,
  "target_2": <float or null — second target, further extension>,
  "key_levels": ["level 1 description", "level 2 description", "level 3 description"],
  "pattern": "name of the chart pattern or structure driving the view"
}}"""


class TechnicalAgent(BaseAgent):
    """Technical Analyst desk agent."""

    def __init__(self, mock: bool = False, mock_client: Any = None):
        super().__init__(
            model       = "gpt-4o",
            temperature = 0.15,   # most deterministic — levels should be consistent
            max_tokens  = 900,
            mock        = mock,
            mock_client = mock_client,
        )

    async def run(self, state: DeskState, api_key: str) -> TechnicalOutput:
        client      = self._get_client(api_key)
        user_prompt = _build_user_prompt(state)

        try:
            raw    = await self._call_openai(client, SYSTEM_PROMPT, user_prompt)
            parsed = json.loads(clean_json(raw))

            entry_low  = self._parse_positive_float(parsed.get("entry_low"),  0.0)
            entry_high = self._parse_positive_float(parsed.get("entry_high"), 0.0)
            stop_loss  = self._parse_positive_float(parsed.get("stop_loss"),  0.0)
            target_1   = self._parse_positive_float(parsed.get("target_1"),   0.0)
            t2_raw     = parsed.get("target_2")
            target_2   = self._parse_positive_float(t2_raw, 0.0) if t2_raw else None

            return TechnicalOutput(
                direction    = parsed.get("direction", "neutral").lower(),
                conviction   = self._parse_float(parsed.get("conviction", 0.5)),
                technical_view = str(parsed.get("technical_view", ""))[:1000],
                entry_low    = entry_low,
                entry_high   = entry_high,
                stop_loss    = stop_loss,
                target_1     = target_1,
                target_2     = target_2,
                key_levels   = list(parsed.get("key_levels", []))[:5],
                pattern      = str(parsed.get("pattern", ""))[:80],
                raw          = raw,
            )
        except Exception as e:
            return TechnicalOutput(
                direction="neutral", conviction=0.25,
                technical_view=f"Technical analysis unavailable: {e}",
                entry_low=0.0, entry_high=0.0,
                stop_loss=0.0, target_1=0.0, target_2=None,
                key_levels=["analysis_error"],
                pattern="error",
                raw=str(e),
            )

    def run_sync(self, state: DeskState, api_key: str) -> TechnicalOutput:
        return asyncio.run(self.run(state, api_key))


# ---------------------------------------------------------------------------
# Mock client
# ---------------------------------------------------------------------------

class _MockTechnicalClient:
    @property
    def chat(self): return self
    @property
    def completions(self): return self

    async def create(self, model, messages, temperature, max_tokens, **kw):
        user_msg = messages[-1]["content"] if messages else ""
        is_bearish = any(w in user_msg.lower()
                         for w in ["hot inflation", "breakdown", "selling", "short"])

        if is_bearish:
            payload = {
                "direction":     "short",
                "conviction":    0.74,
                "technical_view": (
                    "NIFTY50 has broken below the critical 24,000 support level on above-average "
                    "volume, confirming a bearish structural shift. The prior swing low at 23,800 "
                    "is now acting as resistance. RSI is in bearish momentum territory below 45. "
                    "MACD has crossed bearish on the daily. The measured move from the "
                    "head-and-shoulders breakdown targets 23,200. Sell rallies into 23,900–24,000 "
                    "with stop above 24,150 (prior support-turned-resistance)."
                ),
                "entry_low":  23850.0,
                "entry_high": 23980.0,
                "stop_loss":  24160.0,
                "target_1":   23400.0,
                "target_2":   23100.0,
                "key_levels": [
                    "24,000 — prior support, now resistance",
                    "23,800 — recent swing low, critical defence",
                    "23,200 — measured move target from H&S",
                ],
                "pattern": "head_and_shoulders_breakdown",
            }
        else:
            payload = {
                "direction":    "long",
                "conviction":   0.80,
                "technical_view": (
                    "NIFTY50 is forming a strong base at the 24,000–24,200 demand zone after "
                    "a healthy 8% correction from the highs. Price has respected the 200-day "
                    "moving average on a closing basis — a key sign of structural strength. "
                    "Volume on down days has been declining (bearish momentum exhaustion). "
                    "The daily RSI is turning up from 42 — a historically reliable buy signal. "
                    "Entry on any dip into the 24,200–24,350 zone with a stop below the "
                    "recent swing low at 23,900. First target at the prior high of 24,900."
                ),
                "entry_low":  24200.0,
                "entry_high": 24350.0,
                "stop_loss":  23880.0,
                "target_1":   24900.0,
                "target_2":   25400.0,
                "key_levels": [
                    "24,000 — 200-DMA and major structural support",
                    "24,900 — prior high and first target",
                    "23,900 — stop level, recent swing low",
                ],
                "pattern": "base_formation_at_200dma_support",
            }

        class _M:
            content = json.dumps(payload)
        class _C:
            message = _M()
        class _R:
            choices = [_C()]
        return _R()


if __name__ == "__main__":
    state: DeskState = {
        "instrument":         "NIFTY50",
        "exchange":           "NSE",
        "point_in_time_date": str(__import__("datetime").date.today()),
        "event_description":  "RBI holds at 6.5% — dovish tone",
        "macro_view":         {"direction": "long"},
        "market_data": {
            "nifty_spot": 24280,
            "nifty_52w_high": 26277,
            "nifty_52w_low": 21964,
            "india_vix": 13.2,
        },
    }

    agent  = TechnicalAgent(mock=True, mock_client=_MockTechnicalClient())
    result = asyncio.run(agent.run(state, api_key="mock"))

    print(f"\n── Technical Analyst ──")
    print(f"Direction:  {result.direction}  conviction={result.conviction:.0%}")
    print(f"Entry zone: {result.entry_low} – {result.entry_high}")
    print(f"Stop:       {result.stop_loss}  Target 1: {result.target_1}  Target 2: {result.target_2}")
    print(f"Pattern:    {result.pattern}")
    print(f"Key levels: {result.key_levels}")
    print(f"View:\n  {result.technical_view[:200]}...")
