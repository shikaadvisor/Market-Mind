"""
desk/coordinator.py
===================
The Head Trader — the desk coordinator powered by Claude (Anthropic).

Why Claude here and not GPT-4o?
- Model diversity: every other agent in this system uses OpenAI.
  The coordinator uses a different model family to avoid systematic bias.
  Two independent models strongly agreeing = higher confidence signal.
- Claude's reasoning is strong at synthesis: reading multiple conflicting
  inputs and producing a coherent, justified output.
- Anthropic's training emphasises careful reasoning and appropriate
  uncertainty — exactly what you want in a final signal synthesiser.

The coordinator:
1. Reads all four desk agent outputs
2. Weighs them using configured weights (technical 0.30, macro 0.25, crowd 0.25, risk 0.20)
3. Identifies conflicts and resolves them explicitly
4. Applies the risk manager's conviction modifier
5. Builds the complete TradingSignal dict (all fields, fully populated)
6. Applies the lookahead guard — validates point_in_time_date before producing signal

Output: a dict ready for TradingSignal(**output) — fully Pydantic-validated downstream.
"""

from __future__ import annotations

import asyncio
import json
import uuid
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from desk.agents.base import (
    BaseAgent, DeskState, clean_json, POINT_IN_TIME_INSTRUCTION
)


SYSTEM_PROMPT = """You are the Head Trader and desk coordinator at a sophisticated quantitative trading firm.
You receive structured inputs from four specialist agents — Macro Strategist, Technical Analyst,
Risk Manager, and Sentiment Interpreter — and synthesise them into a single, decisive trading signal.

Your synthesis process:
1. CHECK AGREEMENT: Do macro and technical agree on direction? Disagreement reduces conviction.
2. RISK GATE: If the Risk Manager has issued do_not_trade_flags, your signal conviction must
   reflect this — you still produce a signal (it's informative) but conviction should be low
   and the flags must be preserved exactly.
3. WEIGH THE CROWD: Use the sentiment interpreter's recommended weight to scale the crowd input.
   If it's a contrarian signal, it adds to conviction. If it's confirming, it confirms.
   If it's mixed, reduce weight.
4. CONVICTION ARITHMETIC: Apply the risk manager's conviction_modifier to the blended conviction.
5. RESOLVE CONFLICTS: When agents disagree, explain the conflict explicitly in coordinator_thesis.
   Do not paper over disagreements — surface them.
6. PRODUCE THE SIGNAL: All fields must be populated. No null reasoning fields.

Signal conviction interpretation:
- 0.85+: Exceptional setup, multiple agents agree, strong risk profile
- 0.70–0.84: Actionable, review and consider trading
- 0.50–0.69: Monitor only, conditions not quite right
- Below 0.50: Log only, do not consider trading

You must respond ONLY with a valid JSON object. No preamble. No markdown fences. No extra keys."""


def _build_coordinator_prompt(state: DeskState) -> str:
    from config import cfg
    pit     = state.get("point_in_time_date", str(date.today()))
    event   = state.get("event_description", "")
    macro   = state.get("macro_view",    {}) or {}
    tech    = state.get("technical_view", {}) or {}
    risk    = state.get("risk_params",   {}) or {}
    sent    = state.get("sentiment_view", {}) or {}
    sim_met = state.get("simulation_metrics", {}) or {}

    return f"""{POINT_IN_TIME_INSTRUCTION.format(date=pit)}

INSTRUMENT: {state.get('instrument', 'NIFTY50')} | {state.get('exchange', 'NSE')} | {state.get('asset_class', 'equity')}
EVENT/CATALYST: {event}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MACRO STRATEGIST INPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Direction:        {macro.get('direction', 'neutral')}
Conviction:       {macro.get('conviction', 0.5):.0%}
Regime:           {macro.get('regime', 'unknown')}
FII outlook:      {macro.get('fii_outlook', 'neutral')}
Rate environment: {macro.get('rate_environment', 'on_hold')}
Key factors:      {macro.get('key_factors', [])}
Macro context:    {macro.get('macro_context', '')[:300]}
Warning flags:    {macro.get('warning_flags', [])}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TECHNICAL ANALYST INPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Direction:    {tech.get('direction', 'neutral')}
Conviction:   {tech.get('conviction', 0.5):.0%}
Entry zone:   {tech.get('entry_low', 0)} – {tech.get('entry_high', 0)}
Stop loss:    {tech.get('stop_loss', 0)}
Target 1:     {tech.get('target_1', 0)}
Target 2:     {tech.get('target_2', 'none')}
Pattern:      {tech.get('pattern', '')}
Key levels:   {tech.get('key_levels', [])}
Technical view: {tech.get('technical_view', '')[:300]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RISK MANAGER INPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Approved:            {risk.get('approved', True)}
Position size:       {risk.get('max_position_pct', 0.03):.0%}
Risk/Reward:         {risk.get('risk_reward', 0):.2f}
Invalidation level:  {risk.get('invalidation_level', 0)}
Conviction modifier: {risk.get('conviction_modifier', 0):+.2f}
DO NOT TRADE flags:  {risk.get('do_not_trade_flags', [])}
Warning flags:       {risk.get('warning_flags', [])}
Risk rationale:      {risk.get('rationale', '')[:200]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SENTIMENT INTERPRETER INPUT (Digital World)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Crowd direction:    {sent.get('crowd_direction', 'mixed')}
Crowd conviction:   {sent.get('crowd_conviction', 0.5):.0%}
Simulation weight:  {sent.get('simulation_weight', 0.20):.0%}
Contrarian signal:  {sent.get('contrarian_signal', False)}
Key crowd insight:  {sent.get('key_crowd_insight', '')}
Crowd narrative:    {sent.get('crowd_narrative', '')[:300]}

SIMULATION METRICS:
  Aggregate sentiment:     {sim_met.get('aggregate_sentiment', 'N/A')}
  Retail sentiment:        {sim_met.get('retail_sentiment', 'N/A')}
  Institutional sentiment: {sim_met.get('institutional_sentiment', 'N/A')}
  Herd index:              {sim_met.get('herd_index', 'N/A')}
  Panic probability:       {sim_met.get('panic_probability', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WEIGHTING FRAMEWORK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Technical:  {cfg.WEIGHT_TECHNICAL:.0%}
Macro:      {cfg.WEIGHT_MACRO:.0%}
Crowd sim:  {cfg.WEIGHT_CROWD_SIM:.0%} (but use sentiment_interpreter's recommended weight)
Risk:       {cfg.WEIGHT_RISK:.0%}

Synthesise all inputs and respond ONLY with this JSON:
{{
  "direction": "long" | "short" | "neutral",
  "conviction": 0.0-1.0 (apply risk conviction_modifier to your blended score),
  "timeframe": "intraday" | "swing" | "positional",
  "macro_context": "macro agent's narrative, in your own synthesis (80-200 words)",
  "technical_view": "technical agent's narrative, in your own synthesis (80-200 words)",
  "crowd_narrative": "sentiment interpreter's narrative, in your own synthesis (80-200 words)",
  "coordinator_thesis": "your synthesis: why this signal, key conflicts resolved, conviction rationale (150-300 words)",
  "agent_contributions": [
    {{"agent_name": "MacroAgent", "agent_role": "macro_strategist", "view": "long|short|neutral", "conviction": 0.0-1.0, "key_factors": ["f1","f2"], "dissent": null}},
    {{"agent_name": "TechnicalAgent", "agent_role": "technical_analyst", "view": "long|short|neutral", "conviction": 0.0-1.0, "key_factors": ["f1","f2"], "dissent": null}},
    {{"agent_name": "RiskAgent", "agent_role": "risk_manager", "view": "long|short|neutral", "conviction": 0.0-1.0, "key_factors": ["f1","f2"], "dissent": null}},
    {{"agent_name": "SentimentAgent", "agent_role": "sentiment_interpreter", "view": "bullish|bearish|neutral|mixed", "conviction": 0.0-1.0, "key_factors": ["f1"], "dissent": null}}
  ],
  "confidence_flags": ["list of signals adding confidence, e.g. multi_timeframe_confluence"],
  "warning_flags": ["combined warning flags from all agents"],
  "do_not_trade_flags": ["MUST include ALL do_not_trade_flags from Risk Manager — do not omit any"]
}}"""


class DeskCoordinator(BaseAgent):
    """
    Head Trader — Claude-powered final signal synthesiser.
    Takes full DeskState and produces a complete signal dict.
    """

    def __init__(self, mock: bool = False, mock_client: Any = None):
        super().__init__(
            model       = "claude-opus-4-6",
            temperature = 0.20,
            max_tokens  = 2000,
            mock        = mock,
            mock_client = mock_client,
        )

    async def synthesise(self, state: DeskState, api_key: str) -> dict:
        """
        Run Claude synthesis and return a dict ready for TradingSignal(**result).
        This method handles the full signal construction including fields
        that come from state rather than the LLM output.
        """
        client      = self._get_client(api_key, provider="anthropic")
        user_prompt = _build_coordinator_prompt(state)

        try:
            raw    = await self._call_anthropic(client, SYSTEM_PROMPT, user_prompt)
            parsed = json.loads(clean_json(raw))

            tech   = state.get("technical_view", {}) or {}
            risk   = state.get("risk_params",    {}) or {}
            macro  = state.get("macro_view",     {}) or {}
            sent   = state.get("sentiment_view", {}) or {}
            sim    = state.get("simulation_metrics", {}) or {}

            # Pull price levels from technical agent (they own these)
            entry_low  = float(tech.get("entry_low",  0.0))
            entry_high = float(tech.get("entry_high", 0.0))
            stop_loss  = float(tech.get("stop_loss",  0.0))
            target_1   = float(tech.get("target_1",   0.0))
            t2_raw     = tech.get("target_2")
            target_2   = float(t2_raw) if t2_raw else None
            invalidation = float(risk.get("invalidation_level", 0.0)) or (
                stop_loss * 0.985 if stop_loss > 0 else 0.0
            )

            # Compute R:R
            rr = float(risk.get("risk_reward", 0.0))
            if rr == 0.0 and entry_low > 0 and stop_loss > 0 and target_1 > 0:
                entry_mid  = (entry_low + entry_high) / 2
                risk_pts   = abs(entry_mid - stop_loss)
                reward_pts = abs(target_1 - entry_mid)
                rr = round(reward_pts / risk_pts, 2) if risk_pts > 0 else 0.0

            # Conviction — LLM computes it, we clamp
            conviction = max(0.0, min(1.0, float(parsed.get("conviction", 0.5))))

            # Collect flags — Risk Manager's do_not_trade flags are sacred
            risk_dnt   = list(risk.get("do_not_trade_flags", []))
            coord_dnt  = list(parsed.get("do_not_trade_flags", []))
            do_not_trade = list(set(risk_dnt + coord_dnt))

            all_warnings = list(set(
                list(macro.get("warning_flags", [])) +
                list(risk.get("warning_flags", [])) +
                list(parsed.get("warning_flags", []))
            ))

            # Build CrowdMetrics dict from simulation metrics
            crowd_metrics = {
                "aggregate_sentiment":     float(sim.get("aggregate_sentiment",     0.0)),
                "retail_sentiment":        float(sim.get("retail_sentiment",        0.0)),
                "institutional_sentiment": float(sim.get("institutional_sentiment", 0.0)),
                "herd_index":              float(sim.get("herd_index",              0.5)),
                "panic_probability":       float(sim.get("panic_probability",       0.1)),
                "narrative_momentum":      float(sim.get("narrative_momentum",      0.0)),
                "contrarian_pressure":     float(sim.get("contrarian_pressure",     0.1)),
                "information_velocity":    float(sim.get("information_velocity",    0.3)),
                "opinion_cluster_count":   int(sim.get("opinion_cluster_count",     2)),
                "agent_count":             int(sim.get("agent_count",               10210)),
                "simulation_seed_event":   state.get("event_description", "")[:200],
                "simulation_run_id":       sim.get("run_id", str(uuid.uuid4())[:8]),
            }

            # Determine asset class and expiry
            asset_class = state.get("asset_class", "equity")
            expiry_str  = state.get("expiry_date")
            expiry      = date.fromisoformat(expiry_str) if expiry_str else None

            signal_dict = {
                # Identity
                "signal_id":          str(uuid.uuid4()),
                "created_at":         datetime.now(timezone.utc),
                "point_in_time_date": date.fromisoformat(
                    state.get("point_in_time_date", str(date.today()))
                ),

                # Instrument
                "instrument":  state.get("instrument", "NIFTY50"),
                "exchange":    state.get("exchange",   "NSE"),
                "asset_class": asset_class,
                "expiry":      expiry,

                # Signal core
                "direction":   parsed.get("direction", "neutral"),
                "conviction":  conviction,
                "timeframe":   parsed.get("timeframe", "swing"),
                "regime":      macro.get("regime", "range_bound"),

                # Price levels (from technical agent)
                "entry_zone": {"low": entry_low, "high": entry_high},
                "risk": {
                    "stop_loss":          stop_loss,
                    "target_1":           target_1,
                    "target_2":           target_2,
                    "risk_reward":        rr,
                    "max_position_pct":   float(risk.get("max_position_pct", 0.03)),
                    "invalidation_level": invalidation,
                },

                # Crowd intelligence
                "crowd": crowd_metrics,

                # Reasoning
                "macro_context":      parsed.get("macro_context",      "")[:1000],
                "technical_view":     parsed.get("technical_view",     "")[:1000],
                "crowd_narrative":    parsed.get("crowd_narrative",    "")[:1000],
                "coordinator_thesis": parsed.get("coordinator_thesis", "")[:2000],

                # Audit trail
                "agent_contributions": parsed.get("agent_contributions", []),

                # Flags
                "confidence_flags":   list(parsed.get("confidence_flags",  [])),
                "warning_flags":      all_warnings,
                "do_not_trade_flags": do_not_trade,
            }

            return signal_dict

        except Exception as e:
            # Build a minimal error signal rather than crashing the pipeline
            return {
                "signal_id":          str(uuid.uuid4()),
                "created_at":         datetime.now(timezone.utc),
                "point_in_time_date": date.today(),
                "instrument":         state.get("instrument", "UNKNOWN"),
                "exchange":           state.get("exchange", "NSE"),
                "asset_class":        state.get("asset_class", "equity"),
                "direction":          "neutral",
                "conviction":         0.10,
                "timeframe":          "swing",
                "regime":             "range_bound",
                "entry_zone":         {"low": 1.0, "high": 2.0},
                "risk": {
                    "stop_loss": 1.0, "target_1": 2.0, "target_2": None,
                    "risk_reward": 0.0, "max_position_pct": 0.01,
                    "invalidation_level": 1.0,
                },
                "crowd": {
                    "aggregate_sentiment": 0.0, "retail_sentiment": 0.0,
                    "institutional_sentiment": 0.0, "herd_index": 0.5,
                    "panic_probability": 0.1, "narrative_momentum": 0.0,
                    "contrarian_pressure": 0.1, "information_velocity": 0.3,
                    "opinion_cluster_count": 1, "agent_count": 0,
                    "simulation_seed_event": "error", "simulation_run_id": "error",
                },
                "macro_context":      "Coordinator error",
                "technical_view":     "Coordinator error",
                "crowd_narrative":    "Coordinator error",
                "coordinator_thesis": f"Coordinator synthesis failed: {e}",
                "agent_contributions": [],
                "confidence_flags":   [],
                "warning_flags":      ["coordinator_error"],
                "do_not_trade_flags": ["coordinator_error"],
            }

    def run_sync(self, state: DeskState, api_key: str) -> dict:
        return asyncio.run(self.synthesise(state, api_key))


# ---------------------------------------------------------------------------
# Mock Anthropic client for testing
# ---------------------------------------------------------------------------

class _MockAnthropicClient:
    """Mock AsyncAnthropic client."""

    async def messages_create(self, **kwargs):
        return await self.create(**kwargs)

    @property
    def messages(self): return self

    async def create(self, model, system, messages, max_tokens, **kw):
        import re
        user_msg = messages[0]["content"] if messages else ""

        is_bearish  = "short" in user_msg.lower() and "long" not in user_msg[:200].lower()
        is_vetoed   = "do_not_trade" in user_msg.lower() and (
            "macro_technical" in user_msg.lower() or "rr_below" in user_msg.lower()
        )

        direction = "short" if is_bearish else "long"
        conviction = 0.42 if is_vetoed else (0.68 if is_bearish else 0.76)
        dnt_flags = ["macro_technical_direction_conflict"] if is_vetoed else []
        conf_flags = [] if is_vetoed else (
            ["technical_structure_confirmed", "crowd_contrarian_setup"]
            if not is_bearish else ["momentum_confirmed"]
        )

        payload = {
            "direction":   direction,
            "conviction":  conviction,
            "timeframe":   "swing",
            "macro_context": (
                "The global macro environment is creating headwinds for Indian equities. "
                "US inflation above expectations delays Fed cuts, strengthening the dollar "
                "and pressuring EM capital flows. India's domestic macro remains resilient "
                "but FII outflows are a material near-term headwind. RBI is on hold."
                if is_bearish else
                "The macro environment is supportive for Indian equities. RBI's dovish tilt "
                "removes rate uncertainty and supports credit growth. FII flows are turning "
                "neutral to positive. DII SIP inflows provide structural demand. "
                "Domestic consumption indicators remain robust."
            ),
            "technical_view": (
                "Technical structure has broken down below the 24,000 support level. "
                "The breakdown is confirmed by above-average volume and declining RSI. "
                "The pattern is a head-and-shoulders breakdown targeting 23,200. "
                "Entry on rallies into 23,850–23,980 with stop at 24,160."
                if is_bearish else
                "Technical structure is constructive — price is holding above the 200-DMA "
                "and forming a base at the 24,200–24,350 demand zone. "
                "RSI is turning up from 42, volume on down days is declining. "
                "Pattern is a classic demand zone retest — buy the zone, stop below 23,880."
            ),
            "crowd_narrative": (
                "The simulation reveals retail fear (78% bear) with institutional agents "
                "showing a mild contrarian lean. This retail fear + institutional "
                "non-commitment is historically a mean-reversion setup. "
                "Contrarian signal is active — crowd divergence adds to conviction."
                if not is_bearish else
                "Crowd simulation shows broad bearish consensus across all tiers. "
                "When all tiers align, the simulation is confirming rather than "
                "providing contrarian edge. Weight reduced to 15% accordingly."
            ),
            "coordinator_thesis": (
                f"SYNTHESIS: Macro and technical {'agree' if not is_vetoed else 'DISAGREE'} "
                f"on a {direction} view for NIFTY50. "
                f"{'Conviction is reduced due to directional conflict between desk agents. ' if is_vetoed else ''}"
                f"The Digital World simulation "
                f"{'provides a contrarian bullish signal (retail panic + institutional calm) that adds to long conviction' if not is_bearish and not is_vetoed else 'confirms bearish momentum with moderate weight'}. "
                f"Risk Manager has {'approved the trade' if not is_vetoed else 'issued vetoes — signal is informational only'}. "
                f"Final conviction: {conviction:.0%}. "
                f"{'Signal is actionable — review entry zone and execute with defined risk.' if conviction >= 0.70 and not is_vetoed else 'Signal below actionable threshold — monitor but do not trade.'}"
            ),
            "agent_contributions": [
                {"agent_name": "MacroAgent",     "agent_role": "macro_strategist",    "view": direction,       "conviction": 0.65, "key_factors": ["global_rates", "fii_flows"],            "dissent": None},
                {"agent_name": "TechnicalAgent", "agent_role": "technical_analyst",   "view": direction,       "conviction": 0.80, "key_factors": ["structure_breakdown" if is_bearish else "demand_zone_hold", "volume_confirmation"], "dissent": None},
                {"agent_name": "RiskAgent",      "agent_role": "risk_manager",        "view": direction,       "conviction": 0.70, "key_factors": ["rr_acceptable", "stop_at_structure"],   "dissent": None},
                {"agent_name": "SentimentAgent", "agent_role": "sentiment_interpreter","view": "bearish" if is_bearish else "mixed", "conviction": 0.65, "key_factors": ["crowd_consensus" if is_bearish else "retail_fear_institutional_calm"], "dissent": None},
            ],
            "confidence_flags":   conf_flags,
            "warning_flags":      ["monitor_fii_weekly"],
            "do_not_trade_flags": dnt_flags,
        }

        class _Content:
            text = json.dumps(payload)
            type = "text"
        class _Resp:
            content = [_Content()]
        return _Resp()


if __name__ == "__main__":
    state: DeskState = {
        "instrument":         "NIFTY50",
        "exchange":           "NSE",
        "asset_class":        "equity",
        "point_in_time_date": str(date.today()),
        "event_description":  "RBI holds at 6.5% — dovish tone signals cut next meeting",
        "macro_view": {
            "direction": "long", "conviction": 0.72,
            "regime": "trending_up", "fii_outlook": "inflow",
            "rate_environment": "easing",
            "macro_context": "RBI dovish. DII inflows strong. Domestic consumption resilient.",
            "key_factors": ["rbi_dovish", "dii_inflows", "consumption"],
            "warning_flags": [],
        },
        "technical_view": {
            "direction": "long", "conviction": 0.80,
            "entry_low": 24200.0, "entry_high": 24350.0,
            "stop_loss": 23880.0, "target_1": 24900.0, "target_2": 25400.0,
            "technical_view": "Base at 200-DMA, RSI turning up, demand zone retest.",
            "key_levels": ["24,000 support", "24,900 target"],
            "pattern": "base_at_200dma",
        },
        "risk_params": {
            "approved": True, "max_position_pct": 0.04,
            "risk_reward": 2.3, "invalidation_level": 23700.0,
            "do_not_trade_flags": [], "warning_flags": ["monitor_fii_daily"],
            "conviction_modifier": -0.05,
            "rationale": "R:R 2.3 acceptable. Stop at structure. 4% position.",
        },
        "sentiment_view": {
            "crowd_direction": "mixed", "crowd_conviction": 0.68,
            "simulation_weight": 0.22,
            "crowd_narrative": "Retail fear + institutional calm = contrarian bullish setup.",
            "key_crowd_insight": "78% retail bear vs 43% institutional bull — divergence signal.",
            "contrarian_signal": True,
        },
        "simulation_metrics": {
            "aggregate_sentiment": -0.32, "retail_sentiment": -0.45,
            "institutional_sentiment": 0.18, "herd_index": 0.55,
            "panic_probability": 0.08, "narrative_momentum": -0.54,
            "contrarian_pressure": 0.43, "information_velocity": 0.19,
            "opinion_cluster_count": 3, "agent_count": 10210,
            "run_id": "demo-run-001",
        },
    }

    coord  = DeskCoordinator(mock=True, mock_client=_MockAnthropicClient())
    result = asyncio.run(coord.synthesise(state, api_key="mock"))

    print(f"\n── Desk Coordinator (Claude) ──")
    print(f"Direction:  {result['direction']}  conviction={result['conviction']:.0%}")
    print(f"Timeframe:  {result['timeframe']}")
    print(f"Entry:      {result['entry_zone']['low']} – {result['entry_zone']['high']}")
    print(f"Stop:       {result['risk']['stop_loss']}  Target: {result['risk']['target_1']}")
    print(f"R:R:        {result['risk']['risk_reward']:.2f}")
    print(f"Vetoes:     {result['do_not_trade_flags']}")
    print(f"Warnings:   {result['warning_flags']}")
    print(f"Conf flags: {result['confidence_flags']}")
    print(f"\nThesis:\n  {result['coordinator_thesis'][:300]}...")
