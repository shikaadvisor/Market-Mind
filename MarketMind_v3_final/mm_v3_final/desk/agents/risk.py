"""
desk/agents/risk.py
===================
The Risk Manager — the only agent that can VETO the signal entirely.

Runs AFTER Macro and Technical have written their views. Reads both,
applies hard risk rules, and writes RiskOutput including:
- Position sizing (max_position_pct)
- Invalidation level (beyond the stop — where the thesis is structurally dead)
- do_not_trade_flags (hard vetoes — coordinator cannot override these)
- warning_flags (soft cautions — coordinator considers but can override)
- conviction_modifier (can only REDUCE the final conviction, never increase)

Design rule: the Risk Manager is the only agent on the desk with
unilateral veto power. If do_not_trade_flags is non-empty, the signal
is blocked regardless of how high the conviction is. This is intentional.

Model: GPT-4o
Runs: AFTER Macro + Technical (needs their outputs)
Writes to state: risk_params (RiskOutput)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from desk.agents.base import (
    BaseAgent, DeskState, RiskOutput,
    POINT_IN_TIME_INSTRUCTION, clean_json,
)


SYSTEM_PROMPT = """You are the Risk Manager on a sophisticated trading desk.
Your role: evaluate proposed trades, apply hard risk rules, and either approve
or veto the signal. You are the last line of defence before a signal reaches the trader.

Your responsibilities:
1. Position sizing: calculate maximum position size based on conviction and risk
2. Invalidation level: the price where the thesis is structurally dead (beyond the stop)
3. Hard vetoes (do_not_trade_flags): conditions that block the trade entirely:
   - Upcoming high-impact events (RBI policy, budget, US FOMC within 2 days)
   - Earnings announcement within 48 hours for stock-specific trades
   - Position limit already reached in correlated instruments
   - R:R below 1.5 to target_1
   - Extremely high VIX (above 25 for index trades) suggesting panic environment
   - Both macro and technical views disagree on direction
4. Warning flags (non-veto): caution signals the coordinator should consider:
   - Low liquidity session risk
   - Upcoming weekend gap risk
   - High correlation with existing positions

Position sizing rules:
- Base size: 2–5% of portfolio per signal
- Reduce to 1–2% if conviction below 0.65 or warning flags present
- Never exceed 5% in a single signal
- Conviction modifier: 0.0 (no change) to -0.3 (reduce conviction significantly)
  — you can only reduce, never increase the desk's conviction

Be rigorous. Markets punish optimism more than caution.

Respond ONLY with valid JSON. No preamble. No markdown fences."""


def _build_user_prompt(state: DeskState) -> str:
    from config import cfg
    pit   = state.get("point_in_time_date", str(__import__("datetime").date.today()))
    event = state.get("event_description", "")
    macro = state.get("macro_view", {})
    tech  = state.get("technical_view", {})
    md    = state.get("market_data", {})

    macro_dir = macro.get("direction", "neutral") if macro else "neutral"
    macro_conv = macro.get("conviction", 0.5) if macro else 0.5
    macro_warn = macro.get("warning_flags", []) if macro else []

    tech_dir  = tech.get("direction", "neutral") if tech else "neutral"
    tech_conv = tech.get("conviction", 0.5) if tech else 0.5
    entry_low  = tech.get("entry_low",  0.0) if tech else 0.0
    entry_high = tech.get("entry_high", 0.0) if tech else 0.0
    stop       = tech.get("stop_loss",  0.0) if tech else 0.0
    target1    = tech.get("target_1",   0.0) if tech else 0.0

    # Compute R:R for the risk manager to evaluate
    rr = 0.0
    if stop > 0 and target1 > 0 and entry_low > 0:
        entry_mid = (entry_low + entry_high) / 2
        risk_pts  = abs(entry_mid - stop)
        reward_pts = abs(target1 - entry_mid)
        rr = round(reward_pts / risk_pts, 2) if risk_pts > 0 else 0.0

    direction_agree = macro_dir == tech_dir
    asset_class = state.get("asset_class", "equity_index")
    # Vol index: use relevant vol measure for the asset class
    vol_index = (md.get("india_vix") or md.get("equity_vix") or
                 md.get("vix") or md.get("hist_vol_20") or 15.0)
    # High-vol asset classes have different panic thresholds
    vix_panic_threshold = {
        "equity_index": 25, "equity_stock": 30,
        "commodity_energy": 55, "commodity_metal": 30,
        "commodity_agri": 35, "commodity_soft": 40,
        "fx_major": 15, "fx_em": 20,
        "fixed_income": 15, "volatility": 40, "crypto": 80,
    }.get(asset_class, 25)

    return f"""{POINT_IN_TIME_INSTRUCTION.format(date=pit)}

Instrument: {state.get('instrument', 'NIFTY50')} | Asset class: {asset_class} | Exchange: {state.get('exchange', '')}
Event: {event}

DESK VIEWS FOR RISK EVALUATION:
- Macro Strategist: {macro_dir}, conviction={macro_conv:.0%}
  Macro warnings: {macro_warn}
- Technical Analyst: {tech_dir}, conviction={tech_conv:.0%}
  Entry zone: {entry_low} – {entry_high}
  Stop loss: {stop}
  Target 1: {target1}
  Computed R:R: {rr:.2f}
- Macro and Technical AGREE on direction: {direction_agree}

MARKET RISK PARAMETERS:
- Current vol measure: {vol_index:.1f} (panic threshold for {asset_class}: {vix_panic_threshold})
- Min R:R threshold: {cfg.MIN_RISK_REWARD}
- Max position size: {cfg.MAX_POSITION_SIZE_PCT:.0%}
- Asset class volatility note: size positions conservatively for {asset_class}
  (commodities/crypto warrant smaller position sizes than equity indices)

Evaluate this proposed trade and respond ONLY with this JSON:
{{
  "approved": true | false,
  "max_position_pct": 0.01-0.05,
  "invalidation_level": <float — price where thesis is structurally dead, beyond stop>,
  "do_not_trade_flags": ["list hard vetoes here, empty if approved"],
  "warning_flags": ["list soft cautions here, empty if none"],
  "risk_reward": {rr},
  "conviction_modifier": -0.30 to 0.0 (negative only — reduce conviction if needed, 0.0 if no change),
  "rationale": "your risk assessment rationale (60-150 words)"
}}"""


class RiskAgent(BaseAgent):
    """Risk Manager desk agent."""

    def __init__(self, mock: bool = False, mock_client: Any = None):
        super().__init__(
            model       = "gpt-4o",
            temperature = 0.10,   # most deterministic of all agents — rules, not creativity
            max_tokens  = 700,
            mock        = mock,
            mock_client = mock_client,
        )

    async def run(self, state: DeskState, api_key: str) -> RiskOutput:
        client      = self._get_client(api_key)
        user_prompt = _build_user_prompt(state)

        try:
            raw    = await self._call_openai(client, SYSTEM_PROMPT, user_prompt)
            parsed = json.loads(clean_json(raw))

            # Clamp conviction_modifier to [-0.30, 0.00]
            mod = float(parsed.get("conviction_modifier", 0.0))
            mod = max(-0.30, min(0.0, mod))

            # Clamp position size
            pos = float(parsed.get("max_position_pct", 0.03))
            pos = max(0.01, min(0.05, pos))

            do_not_trade = list(parsed.get("do_not_trade_flags", []))
            approved     = bool(parsed.get("approved", True)) and len(do_not_trade) == 0

            # Hard rule: override approval if flags present
            if do_not_trade:
                approved = False

            # Hard rule: R:R below minimum always triggers a flag
            rr = float(parsed.get("risk_reward", 0.0))
            from config import cfg
            if 0 < rr < cfg.MIN_RISK_REWARD and "rr_below_minimum" not in do_not_trade:
                do_not_trade.append(f"rr_below_minimum_{rr:.1f}")
                approved = False

            return RiskOutput(
                approved             = approved,
                max_position_pct     = pos,
                invalidation_level   = self._parse_positive_float(
                    parsed.get("invalidation_level"), 0.0
                ),
                do_not_trade_flags   = do_not_trade,
                warning_flags        = list(parsed.get("warning_flags", [])),
                risk_reward          = rr,
                conviction_modifier  = mod,
                rationale            = str(parsed.get("rationale", ""))[:600],
                raw                  = raw,
            )
        except Exception as e:
            return RiskOutput(
                approved=False,
                max_position_pct=0.02,
                invalidation_level=0.0,
                do_not_trade_flags=["risk_agent_error"],
                warning_flags=[str(e)[:100]],
                risk_reward=0.0,
                conviction_modifier=-0.30,
                rationale=f"Risk agent error: {e}",
                raw=str(e),
            )

    def run_sync(self, state: DeskState, api_key: str) -> RiskOutput:
        return asyncio.run(self.run(state, api_key))


# ---------------------------------------------------------------------------
# Mock client
# ---------------------------------------------------------------------------

class _MockRiskClient:
    @property
    def chat(self): return self
    @property
    def completions(self): return self

    async def create(self, model, messages, temperature, max_tokens, **kw):
        import re
        user_msg = messages[-1]["content"] if messages else ""

        # Extract R:R from prompt
        rr_match = re.search(r"Computed R:R: ([0-9.]+)", user_msg)
        rr = float(rr_match.group(1)) if rr_match else 2.3

        # Extract direction agreement
        agree = "True" in user_msg and "AGREE" in user_msg

        do_not_trade = []
        warning_flags = []
        approved = True
        mod = 0.0

        if rr < 1.5:
            do_not_trade.append(f"rr_below_minimum_{rr:.1f}")
            approved = False
        if not agree:
            do_not_trade.append("macro_technical_direction_conflict")
            approved = False
            mod = -0.20

        if not do_not_trade:
            warning_flags = ["monitor_fii_flow_daily"]
            mod = -0.05

        payload = {
            "approved":            approved,
            "max_position_pct":    0.04 if approved else 0.02,
            "invalidation_level":  23700.0,
            "do_not_trade_flags":  do_not_trade,
            "warning_flags":       warning_flags,
            "risk_reward":         rr,
            "conviction_modifier": mod,
            "rationale": (
                f"R:R of {rr:.2f} {'meets' if rr >= 1.5 else 'fails'} minimum threshold. "
                f"Macro and technical {'agree' if agree else 'DISAGREE'} on direction. "
                f"Position sized at {'4%' if approved else '2%'} given "
                f"{'acceptable' if approved else 'elevated'} risk profile. "
                f"Stop is beyond clear structure — invalidation level at 23,700."
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
    state: DeskState = {
        "instrument":         "NIFTY50",
        "exchange":           "NSE",
        "asset_class":        "equity",
        "point_in_time_date": str(__import__("datetime").date.today()),
        "event_description":  "RBI holds at 6.5% — dovish tone",
        "macro_view":  {"direction": "long", "conviction": 0.72, "warning_flags": []},
        "technical_view": {
            "direction": "long", "conviction": 0.80,
            "entry_low": 24200, "entry_high": 24350,
            "stop_loss": 23880, "target_1": 24900,
        },
        "market_data": {"india_vix": 13.2},
    }

    agent  = RiskAgent(mock=True, mock_client=_MockRiskClient())
    result = asyncio.run(agent.run(state, api_key="mock"))

    print(f"\n── Risk Manager ──")
    print(f"Approved:      {result.approved}")
    print(f"Position size: {result.max_position_pct:.0%}")
    print(f"R:R:           {result.risk_reward:.2f}")
    print(f"Conv modifier: {result.conviction_modifier:+.2f}")
    print(f"Vetoes:        {result.do_not_trade_flags}")
    print(f"Warnings:      {result.warning_flags}")
    print(f"Rationale:     {result.rationale}")
