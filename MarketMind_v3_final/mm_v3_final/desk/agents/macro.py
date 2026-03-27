"""
desk/agents/macro.py
====================
The Macro Strategist — first agent to run on the trading desk.

Analyses the global and domestic macro environment and produces a
directional view with regime classification. Its output sets the
macro context that all subsequent agents can reference.

Model: GPT-4o
Runs: in parallel with Technical Analyst
Writes to state: macro_view (MacroOutput)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from desk.agents.base import (
    BaseAgent, DeskState, MacroOutput,
    POINT_IN_TIME_INSTRUCTION, build_market_context, clean_json,
)


def _build_system_prompt(asset_class: str, instrument: str) -> str:
    """Build asset-class-aware macro system prompt."""

    # Asset-class specific analytical lens
    asset_lens = {
        "equity_index": f"equity index ({instrument}). Focus on: growth expectations, earnings, flows, risk appetite.",
        "equity_stock":  f"individual equity ({instrument}). Focus on: sector macro, earnings cycle, valuation regime.",
        "commodity_energy": f"energy commodity ({instrument}). Focus on: global growth proxy, OPEC policy, USD strength, inventory cycles.",
        "commodity_metal":  f"metals ({instrument}). Focus on: real rates (gold), China industrial demand (copper/base metals), supply disruption.",
        "commodity_agri":   f"agricultural commodity ({instrument}). Focus on: weather patterns, USDA supply/demand balance, USD, biofuel policy.",
        "commodity_soft":   f"soft commodity ({instrument}). Focus on: seasonal supply cycles, emerging market demand, weather.",
        "fx_major":     f"major FX pair ({instrument}). Focus on: rate differentials, central bank divergence, current account, PPP.",
        "fx_em":        f"EM FX ({instrument}). Focus on: carry attractiveness, political risk, capital flows, dollar cycle.",
        "fixed_income": f"rates/bonds ({instrument}). Focus on: central bank path, inflation expectations, supply/demand for duration.",
        "volatility":   f"volatility ({instrument}). Focus on: macro uncertainty, event risk calendar, risk appetite regime.",
        "crypto":       f"crypto asset ({instrument}). Focus on: risk appetite, regulatory environment, institutional flow, macro liquidity.",
    }
    ac_key = next((k for k in asset_lens if asset_class.startswith(k)), "equity_index")
    lens = asset_lens[ac_key]

    return f"""You are the Macro Strategist on a sophisticated global trading desk.
Your role: analyse the macro environment for {lens}

Your analytical framework (apply what is relevant to this asset class):
- Global rates: US Fed path, real rates, DXY direction, global central bank divergence
- Growth cycle: PMI, GDP expectations, China cycle, EM growth differentials
- Commodity macro: supply/demand balances, OPEC policy, inventory cycles, weather (if relevant)
- Cross-asset flows: fund flows, positioning, carry trade conditions, risk appetite
- Inflation regime: CPI/PCE trends, commodity pass-through, wage growth

Output requirements:
- Be specific to THIS asset class — not generic equity commentary
- Reference actual current indicators
- Identify the single most important macro driver right now

Respond ONLY with valid JSON. No preamble. No markdown fences. No extra keys."""


def _build_user_prompt(state: DeskState) -> str:
    pit = state.get("point_in_time_date", str(__import__("datetime").date.today()))
    event = state.get("event_description", "No specific event — general market analysis")
    market_ctx = build_market_context(state)
    asset_class = state.get("asset_class", "equity_index")
    instrument  = state.get("instrument", "UNKNOWN")

    return f"""{POINT_IN_TIME_INSTRUCTION.format(date=pit)}

Instrument: {instrument} | Asset class: {asset_class} | Exchange: {state.get('exchange', '')}
Event/catalyst: {event}

{market_ctx}

Analyse the macro environment FOR THIS SPECIFIC ASSET CLASS and provide your assessment.
Respond ONLY with this JSON structure:
{{
  "regime": "trending_up" | "trending_down" | "range_bound" | "high_volatility" | "event_driven",
  "direction": "long" | "short" | "neutral",
  "conviction": 0.0-1.0,
  "macro_context": "your macro narrative — what the environment looks like and why it matters (80-200 words)",
  "key_factors": ["factor 1", "factor 2", "factor 3"],
  "fii_outlook": "inflow" | "outflow" | "neutral",
  "rate_environment": "tightening" | "easing" | "on_hold",
  "warning_flags": ["any macro risks to flag, empty list if none"]
}}"""


class MacroAgent(BaseAgent):
    """Macro Strategist desk agent."""

    def __init__(self, mock: bool = False, mock_client: Any = None):
        super().__init__(
            model       = "gpt-4o",
            temperature = 0.20,
            max_tokens  = 900,
            mock        = mock,
            mock_client = mock_client,
        )

    async def run(self, state: DeskState, api_key: str) -> MacroOutput:
        client      = self._get_client(api_key)
        user_prompt = _build_user_prompt(state)
        system_prompt = _build_system_prompt(
            state.get("asset_class", "equity_index"),
            state.get("instrument", "UNKNOWN"),
        )

        try:
            raw    = await self._call_openai(client, system_prompt, user_prompt)
            parsed = json.loads(clean_json(raw))

            return MacroOutput(
                regime           = parsed.get("regime", "range_bound"),
                direction        = parsed.get("direction", "neutral").lower(),
                conviction       = self._parse_float(parsed.get("conviction", 0.5)),
                macro_context    = str(parsed.get("macro_context", ""))[:1000],
                key_factors      = list(parsed.get("key_factors", []))[:4],
                fii_outlook      = parsed.get("fii_outlook", "neutral"),
                rate_environment = parsed.get("rate_environment", "on_hold"),
                warning_flags    = list(parsed.get("warning_flags", [])),
                raw              = raw,
            )
        except Exception as e:
            return MacroOutput(
                regime="range_bound", direction="neutral", conviction=0.30,
                macro_context=f"Macro analysis unavailable: {e}",
                key_factors=["analysis_error"],
                fii_outlook="neutral", rate_environment="on_hold",
                warning_flags=["macro_agent_error"],
                raw=str(e),
            )

    def run_sync(self, state: DeskState, api_key: str) -> MacroOutput:
        return asyncio.run(self.run(state, api_key))


# ---------------------------------------------------------------------------
# Mock client for testing
# ---------------------------------------------------------------------------

class _MockMacroClient:
    @property
    def chat(self): return self
    @property
    def completions(self): return self

    async def create(self, model, messages, temperature, max_tokens, **kw):
        user_msg = messages[-1]["content"] if messages else ""
        is_bearish = any(w in user_msg.lower()
                         for w in ["hot inflation", "breakdown", "selling", "fear"])

        payload = {
            "regime":   "trending_down" if is_bearish else "trending_up",
            "direction": "short" if is_bearish else "long",
            "conviction": 0.65 if is_bearish else 0.72,
            "macro_context": (
                "The global macro environment is turning hostile for risk assets. "
                "US inflation surprising to the upside forces the Fed to delay cuts, "
                "strengthening the dollar and pressuring EM flows. India is relatively "
                "insulated by domestic demand but FII outflows of ₹8,000–12,000 crore "
                "weekly are a headwind. RBI is on hold with a hawkish tilt. "
                "The rate differential narrows, making India less attractive to carry traders."
                if is_bearish else
                "Domestic macro remains supportive. RBI's dovish pivot removes near-term "
                "rate uncertainty and boosts credit growth expectations. FII flows are turning "
                "neutral to positive as global risk appetite improves. GST collections at "
                "₹1.7 lakh crore signal robust domestic consumption. DII inflows via SIPs "
                "provide a steady bid. The macro regime is trending upward."
            ),
            "key_factors": (
                ["us_inflation_above_expectations", "fed_rate_cut_delay", "fii_outflows_accelerating"]
                if is_bearish else
                ["rbi_dovish_pivot", "dii_sip_flows_strong", "domestic_consumption_resilient"]
            ),
            "fii_outlook": "outflow" if is_bearish else "inflow",
            "rate_environment": "tightening" if is_bearish else "easing",
            "warning_flags": (
                ["fed_higher_for_longer", "usd_strength_EM_headwind"]
                if is_bearish else []
            ),
        }

        class _M:
            content = json.dumps(payload)
        class _C:
            message = _M()
        class _R:
            choices = [_C()]
        return _R()


if __name__ == "__main__":
    import asyncio
    state: DeskState = {
        "instrument":         "NIFTY50",
        "exchange":           "NSE",
        "asset_class":        "equity",
        "point_in_time_date": str(__import__("datetime").date.today()),
        "event_description":  "US CPI 3.4% vs 2.9% expected — hot inflation surprise",
        "market_data": {
            "nifty_spot":      24180,
            "fii_net_weekly":  -8200,
            "india_vix":       14.8,
            "usd_inr":         84.2,
        },
    }

    agent  = MacroAgent(mock=True, mock_client=_MockMacroClient())
    result = asyncio.run(agent.run(state, api_key="mock"))

    print(f"\n── Macro Strategist ──")
    print(f"Regime:    {result.regime}")
    print(f"Direction: {result.direction}  conviction={result.conviction:.0%}")
    print(f"FII:       {result.fii_outlook}  rates={result.rate_environment}")
    print(f"Context:\n  {result.macro_context[:200]}...")
    print(f"Factors:   {result.key_factors}")
    print(f"Warnings:  {result.warning_flags}")
