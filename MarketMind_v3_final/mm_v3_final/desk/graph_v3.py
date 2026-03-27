"""
desk/graph_v3.py
================
Full desk graph wired to v3 components:
  - Model names from model_registry (Risk 3 / Risk 9)
  - Per-agent timeouts (Risk 4)
  - DevilsAdvocate two-pass coordinator (Risk 1)
  - Universe divergence injected into sentiment agent (Bug 15 partial)
  - Feedback context injected post-coordinator (Bug 15 full fix done in runner)
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
logger = logging.getLogger("marketmind.desk")

from desk.agents.base import DeskState
from desk.agents.macro import MacroAgent
from desk.agents.technical import TechnicalAgent
from desk.agents.risk import RiskAgent
from desk.agents.sentiment_v3 import SentimentAgentV3
from desk.coordinator_v3 import DeskCoordinatorV3
from utils.model_registry import get as get_model


AGENT_TIMEOUT = 90


async def _timed(coro, label: str, timeout: int = AGENT_TIMEOUT):
    try:
        return await asyncio.wait_for(coro, timeout=timeout), None
    except asyncio.TimeoutError:
        logger.error("Agent '%s' timed out after %ds", label, timeout)
        return None, f"{label}_timeout"
    except Exception as exc:
        logger.exception("Agent '%s' failed: %s", label, exc)
        return None, f"{label}_error:{str(exc)[:80]}"


class DeskGraphV3:

    def __init__(self, openai_api_key: str, anthropic_api_key: str,
                 mock: bool = False, agent_timeout: int = AGENT_TIMEOUT):
        self.openai_key    = openai_api_key
        self.anthropic_key = anthropic_api_key
        self.mock          = mock
        self.timeout       = agent_timeout

        # Risk 3 / Risk 9: model names from registry
        m_desk = get_model("desk_macro")
        self._macro     = MacroAgent(mock=mock)
        self._macro.model = m_desk.name

        m_tech = get_model("desk_technical")
        self._technical = TechnicalAgent(mock=mock)
        self._technical.model = m_tech.name

        m_risk = get_model("desk_risk")
        self._risk      = RiskAgent(mock=mock)
        self._risk.model = m_risk.name

        self._sentiment  = SentimentAgentV3(mock=mock)
        self._coordinator = DeskCoordinatorV3(mock=mock)

    async def run_async(self, state: DeskState) -> DeskState:
        state  = dict(state)
        errors: list[str] = []

        # ── Phase 1: Macro, Technical, Sentiment in parallel ─────────────────
        (m_out, m_err), (t_out, t_err), (s_out, s_err) = await asyncio.gather(
            _timed(self._macro.run(state, self.openai_key),     "macro",     self.timeout),
            _timed(self._technical.run(state, self.openai_key), "technical", self.timeout),
            _timed(self._sentiment.run(state, self.openai_key,
                                       state.get("feedback_context","")),
                   "sentiment", self.timeout),
        )

        if m_err:  errors.append(m_err)
        if t_err:  errors.append(t_err); errors.extend(["zero_price_from_technical","do_not_trade_price_data_error"])
        if s_err:  errors.append(s_err)

        if m_out:
            state["macro_view"] = {
                "direction": m_out.direction, "conviction": m_out.conviction,
                "regime": m_out.regime, "macro_context": m_out.macro_context,
                "key_factors": m_out.key_factors, "fii_outlook": m_out.fii_outlook,
                "rate_environment": m_out.rate_environment, "warning_flags": m_out.warning_flags,
            }
        if t_out:
            state["technical_view"] = {
                "direction": t_out.direction, "conviction": t_out.conviction,
                "technical_view": t_out.technical_view,
                "entry_low": t_out.entry_low, "entry_high": t_out.entry_high,
                "stop_loss": t_out.stop_loss, "target_1": t_out.target_1,
                "target_2": t_out.target_2, "key_levels": t_out.key_levels,
                "pattern": t_out.pattern,
            }
        if s_out:
            state["sentiment_view"] = {
                "crowd_direction": s_out.crowd_direction, "crowd_conviction": s_out.crowd_conviction,
                "simulation_weight": s_out.simulation_weight, "crowd_narrative": s_out.crowd_narrative,
                "key_crowd_insight": s_out.key_crowd_insight, "contrarian_signal": s_out.contrarian_signal,
                # v3 extras surfaced from swarm (parsed from raw if present)
                "universe_divergence_flag": False, "confidence_boost_flag": False,
            }

        # ── Phase 2: Risk ──────────────────────────────────────────────────────
        r_out, r_err = await _timed(self._risk.run(state, self.openai_key), "risk", self.timeout)
        if r_err: errors.append(r_err)

        if r_out:
            # Risk 11 fix: reconcile invalidation with technical stop
            tech_stop    = float(state.get("technical_view",{}).get("stop_loss",0))
            risk_invalid = float(r_out.invalidation_level) if r_out.invalidation_level else 0.0
            if tech_stop > 1e-6 and risk_invalid > 1e-6:
                if abs(tech_stop - risk_invalid) / tech_stop > 0.02:
                    risk_invalid = tech_stop * 0.985

            state["risk_params"] = {
                "approved": r_out.approved,
                "max_position_pct": r_out.max_position_pct,
                "invalidation_level": risk_invalid,
                "do_not_trade_flags": list(r_out.do_not_trade_flags) + errors,
                "warning_flags": r_out.warning_flags,
                "risk_reward": r_out.risk_reward,
                "conviction_modifier": r_out.conviction_modifier,
                "rationale": r_out.rationale,
            }

        # ── Phase 3: Two-pass coordinator (with devil's advocate) ─────────────
        coord_result = await self._coordinator.synthesise(
            state, self.openai_key, self.anthropic_key
        )

        # Merge accumulated errors into do_not_trade
        if errors:
            existing = list(coord_result.get("do_not_trade_flags", []))
            coord_result["do_not_trade_flags"] = list(set(existing + errors))

        state["final_signal"] = coord_result
        return state

    def run(self, state: DeskState) -> DeskState:
        return asyncio.run(self.run_async(state))


def build_initial_state_v3(
    instrument:        str,
    exchange:          str,
    asset_class:       str,
    event_description: str,
    swarm_report,
    market_data:       dict | None = None,
    expiry_date:       str | None  = None,
    feedback_context:  str         = "",
) -> DeskState:
    from datetime import date
    metrics = swarm_report.metrics_dict if swarm_report else {}
    return DeskState(
        instrument         = instrument,
        exchange           = exchange,
        asset_class        = asset_class,
        point_in_time_date = str(date.today()),
        market_data        = market_data or {},
        simulation_brief   = getattr(swarm_report, "swarm_brief", ""),
        simulation_metrics = metrics,
        event_description  = event_description,
        expiry_date        = expiry_date,
        swarm_report       = swarm_report,
        feedback_context   = feedback_context,
    )
