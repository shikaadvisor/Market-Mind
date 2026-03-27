"""
desk/coordinator_v3.py
=======================
Risk 1 fix: Two-pass coordinator with devil's advocate integration.

Pass 1: Standard synthesis from all four desk agents (macro, technical,
        sentiment, risk). Produces a draft signal dict.

Pass 2 (conditional): If devil's advocate counterargument_strength >= 0.70,
        coordinator receives its own draft + the counterargument and must:
        - Explicitly rebut the counterargument, OR
        - Lower conviction to max(draft_conviction, recommended_cap), OR
        - Flip direction if the counterargument is overwhelming (strength > 0.90)

The two-pass design adds one LLM call (~$0.008) only when the devil finds
a genuine weakness. Weak counterarguments (strength < 0.70) skip pass 2.

Risk 9 fix: Model name read from model_registry, never hardcoded.
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

from desk.agents.base import BaseAgent, DeskState, clean_json, POINT_IN_TIME_INSTRUCTION
from desk.agents.devil_advocate import DevilsAdvocateAgent


SYSTEM_PROMPT = """You are the Head Trader and desk coordinator at a sophisticated quantitative trading firm.
You receive structured inputs from five specialist agents — Macro Strategist, Technical Analyst,
Risk Manager, Sentiment Interpreter, and a Devil's Advocate — and synthesise them into one signal.

Your synthesis process:
1. CHECK AGREEMENT: Do macro and technical agree on direction? Disagreement reduces conviction.
2. RISK GATE: If Risk Manager issued do_not_trade_flags, conviction must reflect this.
3. WEIGH THE CROWD: Use sentiment interpreter's recommended weight for simulation data.
4. ADDRESS THE DEVIL: If a devil's advocate counterargument is present (strength >= 0.70),
   you MUST explicitly rebut it in coordinator_thesis or lower your conviction accordingly.
   You cannot paper over a strong counterargument — surface it, address it, explain why
   you are proceeding despite it (or lower conviction if you cannot rebut it).
5. CONVICTION ARITHMETIC: Apply risk manager's conviction_modifier to blended conviction.
6. RESOLVE CONFLICTS: Explain disagreements explicitly. Never paper over them.

Signal conviction:
- 0.85+: Exceptional setup, multiple agents agree, devil's advocate rebutted
- 0.70-0.84: Actionable, review and consider
- 0.50-0.69: Monitor only
- Below 0.50: Log only, do not trade

Respond ONLY with valid JSON. No preamble. No markdown fences. No extra keys."""


def _build_prompt(state: DeskState, devil: dict | None = None) -> str:
    from config import cfg
    pit   = state.get("point_in_time_date", str(date.today()))
    macro = state.get("macro_view",    {}) or {}
    tech  = state.get("technical_view", {}) or {}
    risk  = state.get("risk_params",   {}) or {}
    sent  = state.get("sentiment_view", {}) or {}
    sim   = state.get("simulation_metrics", {}) or {}

    devil_section = ""
    if devil and devil.get("triggered"):
        devil_section = f"""
DEVIL'S ADVOCATE (counterargument_strength={devil['counterargument_strength']:.2f} — MUST ADDRESS):
Counterargument: {devil['counterargument'][:500]}
Key risk ignored: {devil['key_risk']}
Ignored data: {devil['ignored_data_points']}
Recommended conviction cap: {devil['recommended_conviction_cap']:.0%}
"""
    elif devil:
        devil_section = f"\nDEVIL'S ADVOCATE (strength={devil.get('counterargument_strength',0):.2f} — weak, noted only): {devil.get('key_risk','')}\n"

    return f"""{POINT_IN_TIME_INSTRUCTION.format(date=pit)}

INSTRUMENT: {state.get('instrument','NIFTY50')} | {state.get('exchange','NSE')} | {state.get('asset_class','equity')}
EVENT: {state.get('event_description','')}
{devil_section}
MACRO STRATEGIST:
  Direction={macro.get('direction','neutral')} conviction={macro.get('conviction',0.5):.0%}
  Regime={macro.get('regime','unknown')}  Rate={macro.get('rate_environment','on_hold')}
  Context: {macro.get('macro_context','')[:200]}
  Warnings: {macro.get('warning_flags',[])}

TECHNICAL ANALYST:
  Direction={tech.get('direction','neutral')} conviction={tech.get('conviction',0.5):.0%}
  Entry={tech.get('entry_low',0)}–{tech.get('entry_high',0)}  Stop={tech.get('stop_loss',0)}
  Target1={tech.get('target_1',0)}  Target2={tech.get('target_2','none')}
  View: {tech.get('technical_view','')[:200]}

RISK MANAGER:
  Approved={risk.get('approved',True)}  Position={risk.get('max_position_pct',0.03):.0%}
  R:R={risk.get('risk_reward',0):.2f}  Conviction modifier={risk.get('conviction_modifier',0):+.2f}
  DO NOT TRADE: {risk.get('do_not_trade_flags',[])}
  Warnings: {risk.get('warning_flags',[])}

SENTIMENT INTERPRETER:
  Crowd={sent.get('crowd_direction','mixed')} conviction={sent.get('crowd_conviction',0.5):.0%}
  Sim weight={sent.get('simulation_weight',0.20):.0%}  Contrarian={sent.get('contrarian_signal',False)}
  Universe divergence flagged: {sent.get('universe_divergence_flag',False)}
  Confidence boost: {sent.get('confidence_boost_flag',False)}
  Insight: {sent.get('key_crowd_insight','')}

WEIGHTS: Technical={cfg.WEIGHT_TECHNICAL:.0%}  Macro={cfg.WEIGHT_MACRO:.0%}  Crowd={cfg.WEIGHT_CROWD_SIM:.0%}  Risk={cfg.WEIGHT_RISK:.0%}

Synthesise and respond ONLY with this JSON:
{{
  "direction": "long"|"short"|"neutral",
  "conviction": 0.0-1.0,
  "timeframe": "intraday"|"swing"|"positional",
  "macro_context": "macro synthesis (80-200 words)",
  "technical_view": "technical synthesis (80-200 words)",
  "crowd_narrative": "crowd synthesis (80-200 words)",
  "coordinator_thesis": "full synthesis including devil's advocate rebuttal if triggered (150-300 words)",
  "devil_rebuttal": "explicit rebuttal of counterargument or 'N/A'",
  "agent_contributions": [
    {{"agent_name":"MacroAgent","agent_role":"macro_strategist","view":"long|short|neutral","conviction":0.0-1.0,"key_factors":["f1","f2"],"dissent":null}},
    {{"agent_name":"TechnicalAgent","agent_role":"technical_analyst","view":"long|short|neutral","conviction":0.0-1.0,"key_factors":["f1","f2"],"dissent":null}},
    {{"agent_name":"RiskAgent","agent_role":"risk_manager","view":"long|short|neutral","conviction":0.0-1.0,"key_factors":["f1","f2"],"dissent":null}},
    {{"agent_name":"SentimentAgent","agent_role":"sentiment_interpreter","view":"bullish|bearish|neutral|mixed","conviction":0.0-1.0,"key_factors":["f1"],"dissent":null}},
    {{"agent_name":"DevilsAdvocate","agent_role":"adversarial_checker","view":"long|short|neutral","conviction":0.0-1.0,"key_factors":["counterargument_summary"],"dissent":"rebuttal or null"}}
  ],
  "confidence_flags": ["list signals adding confidence"],
  "warning_flags": ["combined warning flags from all agents"],
  "do_not_trade_flags": ["MUST include ALL do_not_trade_flags from Risk Manager"]
}}"""


class DeskCoordinatorV3(BaseAgent):

    def __init__(self, mock: bool = False, mock_client: Any = None):
        from utils.model_registry import get
        m = get("coordinator")
        super().__init__(
            model       = m.name,  # Risk 9 fix: from registry, not hardcoded
            temperature = 0.20,
            max_tokens  = 2000,
            mock        = mock,
            mock_client = mock_client,
        )
        self._devil = DevilsAdvocateAgent(mock=mock, mock_client=mock_client)

    async def synthesise(self, state: DeskState, api_key: str, anthropic_key: str) -> dict:
        client = self._get_client(anthropic_key, provider="anthropic")

        # ── Pass 1: standard synthesis ────────────────────────────────────────
        user_prompt_p1 = _build_prompt(state, devil=None)
        try:
            raw1   = await self._call_anthropic(client, SYSTEM_PROMPT, user_prompt_p1)
            draft  = json.loads(clean_json(raw1))
        except Exception as e:
            return _error_signal(state, f"Coordinator pass 1 failed: {e}")

        # ── Devil's advocate ──────────────────────────────────────────────────
        openai_client = self._get_client(api_key, provider="openai")
        try:
            devil = await self._devil.run(state, draft, api_key)
        except Exception as e:
            devil = {"counterargument_strength": 0.0, "triggered": False, "key_risk": str(e)}

        # ── Pass 2 (only if devil triggered) ─────────────────────────────────
        if devil.get("triggered", False):
            state_with_devil = {**state, "_devil_output": devil}
            user_prompt_p2 = _build_prompt(state_with_devil, devil=devil)
            try:
                raw2  = await self._call_anthropic(client, SYSTEM_PROMPT, user_prompt_p2)
                draft = json.loads(clean_json(raw2))
                # Enforce conviction cap from devil's advocate
                cap = float(devil.get("recommended_conviction_cap", 1.0))
                draft_conv = float(draft.get("conviction", 0.5))
                if draft_conv > cap:
                    draft["conviction"] = cap
                    draft.setdefault("warning_flags", []).append(
                        f"conviction_capped_by_devil_advocate:{cap:.2f}"
                    )
            except Exception as e:
                draft.setdefault("warning_flags", []).append(f"coordinator_pass2_failed:{e}")

        return _build_signal_dict(draft, state, devil)

    def run_sync(self, state: DeskState, api_key: str, anthropic_key: str) -> dict:
        return asyncio.run(self.synthesise(state, api_key, anthropic_key))


def _build_signal_dict(parsed: dict, state: DeskState, devil: dict) -> dict:
    tech  = state.get("technical_view", {}) or {}
    risk  = state.get("risk_params",   {}) or {}
    macro = state.get("macro_view",    {}) or {}
    sim   = state.get("simulation_metrics", {}) or {}

    entry_low  = float(tech.get("entry_low",  0.0))
    entry_high = float(tech.get("entry_high", 0.0))
    stop_loss  = float(tech.get("stop_loss",  0.0))
    target_1   = float(tech.get("target_1",   0.0))
    t2_raw     = tech.get("target_2")
    target_2   = float(t2_raw) if t2_raw else None

    invalidation = float(risk.get("invalidation_level", 0.0)) or (stop_loss * 0.985 if stop_loss > 0 else 0.0)
    rr = float(risk.get("risk_reward", 0.0))
    if rr == 0.0 and entry_low > 0 and stop_loss > 0 and target_1 > 0:
        mid  = (entry_low + entry_high) / 2
        risk_pts   = abs(mid - stop_loss)
        reward_pts = abs(target_1 - mid)
        rr = round(reward_pts / risk_pts, 2) if risk_pts > 0 else 0.0

    conviction = max(0.0, min(1.0, float(parsed.get("conviction", 0.5))))

    risk_dnt  = list(risk.get("do_not_trade_flags", []))
    coord_dnt = list(parsed.get("do_not_trade_flags", []))
    do_not_trade = list(set(risk_dnt + coord_dnt))

    all_warnings = list(set(
        list(macro.get("warning_flags", [])) +
        list(risk.get("warning_flags", [])) +
        list(parsed.get("warning_flags", []))
    ))

    # Devil's advocate metadata
    if devil.get("triggered"):
        all_warnings.append(f"devil_triggered_strength:{devil.get('counterargument_strength',0):.2f}")

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
        "agent_count":             int(sim.get("agent_count",               10000)),
        "universe_divergence":     float(sim.get("universe_divergence",     0.0)),
        "simulation_seed_event":   state.get("event_description", "")[:200],
        "simulation_run_id":       sim.get("run_id", str(uuid.uuid4())[:8]),
    }

    thesis = str(parsed.get("coordinator_thesis", ""))
    if devil.get("triggered") and parsed.get("devil_rebuttal"):
        thesis += f"\n\nDEVIL'S ADVOCATE REBUTTAL: {parsed['devil_rebuttal']}"

    return {
        "signal_id":          str(uuid.uuid4()),
        "created_at":         datetime.now(timezone.utc),
        "point_in_time_date": date.fromisoformat(state.get("point_in_time_date", str(date.today()))),
        "instrument":  state.get("instrument", "UNKNOWN"),
        "exchange":    state.get("exchange", "NSE"),
        "asset_class": state.get("asset_class", "equity"),
        "expiry":      None,
        "direction":   parsed.get("direction", "neutral"),
        "conviction":  conviction,
        "timeframe":   parsed.get("timeframe", "swing"),
        "regime":      macro.get("regime", "range_bound"),
        "entry_zone":  {"low": entry_low, "high": entry_high},
        "risk": {
            "stop_loss": stop_loss, "target_1": target_1, "target_2": target_2,
            "risk_reward": rr, "max_position_pct": float(risk.get("max_position_pct", 0.03)),
            "invalidation_level": invalidation,
        },
        "crowd":               crowd_metrics,
        "macro_context":       str(parsed.get("macro_context",   ""))[:1000],
        "technical_view":      str(parsed.get("technical_view",  ""))[:1000],
        "crowd_narrative":     str(parsed.get("crowd_narrative", ""))[:1000],
        "coordinator_thesis":  thesis[:2000],
        "agent_contributions": parsed.get("agent_contributions", []),
        "confidence_flags":    list(parsed.get("confidence_flags",  [])),
        "warning_flags":       all_warnings,
        "do_not_trade_flags":  do_not_trade,
        "devil_advocate": {
            "strength":   devil.get("counterargument_strength", 0.0),
            "triggered":  devil.get("triggered", False),
            "key_risk":   devil.get("key_risk", ""),
        },
    }


def _error_signal(state: DeskState, error: str) -> dict:
    return {
        "signal_id": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc),
        "point_in_time_date": date.today(),
        "instrument": state.get("instrument","UNKNOWN"), "exchange": state.get("exchange","NSE"),
        "asset_class": state.get("asset_class","equity"), "expiry": None,
        "direction":"neutral","conviction":0.10,"timeframe":"swing","regime":"range_bound",
        "entry_zone":{"low":1.0,"high":2.0},
        "risk":{"stop_loss":1.0,"target_1":2.0,"target_2":None,"risk_reward":0.0,
                "max_position_pct":0.01,"invalidation_level":1.0},
        "crowd":{"aggregate_sentiment":0,"retail_sentiment":0,"institutional_sentiment":0,
                 "herd_index":0.5,"panic_probability":0.1,"narrative_momentum":0,
                 "contrarian_pressure":0.1,"information_velocity":0.3,
                 "opinion_cluster_count":1,"agent_count":0,"universe_divergence":0,
                 "simulation_seed_event":"error","simulation_run_id":"error"},
        "macro_context":"Coordinator error","technical_view":"Coordinator error",
        "crowd_narrative":"Coordinator error","coordinator_thesis":f"Error: {error}",
        "agent_contributions":[],
        "confidence_flags":[],"warning_flags":["coordinator_error"],
        "do_not_trade_flags":["coordinator_error"],
        "devil_advocate":{"strength":0,"triggered":False,"key_risk":""},
    }
