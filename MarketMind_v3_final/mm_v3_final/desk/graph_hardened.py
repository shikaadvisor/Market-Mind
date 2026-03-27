"""
desk/graph_hardened.py  — HARDENED DESK GRAPH
==============================================
Fixes applied vs v1.0:

FIX-4  Per-agent timeout — asyncio.wait_for() on each agent call
FIX-9  Model name sourced from config.py, not hardcoded in DeskCoordinator
FIX-10 Agent fallback surfaced honestly — price-zero outputs get flagged,
       not silently passed through
FIX-11 Risk / Technical invalidation reconciliation

Drop-in: replace DeskGraph import with HardenedDeskGraph in pipeline/runner.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger("marketmind.desk")

AGENT_TIMEOUT = 90   # seconds per agent call


async def _with_timeout(coro, label: str, timeout: int = AGENT_TIMEOUT):
    """
    Wraps a coroutine with a timeout, returning (result, error_str).
    Never raises — always returns a tuple.
    """
    try:
        result = await asyncio.wait_for(coro, timeout=timeout)
        return result, None
    except asyncio.TimeoutError:
        logger.error("Agent '%s' timed out after %ds", label, timeout)
        return None, f"{label}_timeout"
    except Exception as exc:
        logger.exception("Agent '%s' failed: %s", label, exc)
        return None, f"{label}_error:{str(exc)[:80]}"


def _zero_price_fallback_flags(agent_name: str) -> list[str]:
    return [f"zero_price_from_{agent_name}", "do_not_trade_price_data_error"]


class HardenedDeskGraph:
    """
    Trading desk graph with per-agent timeouts, honest error propagation,
    and model name sourced from config.
    """

    def __init__(
        self,
        openai_api_key:    str,
        anthropic_api_key: str,
        mock:              bool = False,
        agent_timeout:     int  = AGENT_TIMEOUT,
    ):
        self.openai_key    = openai_api_key
        self.anthropic_key = anthropic_api_key
        self.mock          = mock
        self.timeout       = agent_timeout

        # FIX-9: read model from config, not hardcoded
        from config import cfg
        self._coordinator_model = cfg.ANTHROPIC_MODEL_COORD

        from desk.agents.macro     import MacroAgent
        from desk.agents.technical import TechnicalAgent
        from desk.agents.risk      import RiskAgent
        from desk.agents.sentiment import SentimentAgent
        from desk.coordinator      import DeskCoordinator

        self._macro     = MacroAgent(mock=mock)
        self._technical = TechnicalAgent(mock=mock)
        self._sentiment = SentimentAgent(mock=mock)
        self._risk      = RiskAgent(mock=mock)

        # FIX-9: coordinator uses model from config
        self._coordinator = DeskCoordinator(mock=mock)
        self._coordinator.model = self._coordinator_model

    async def run_async(self, state: dict) -> dict:
        from desk.agents.base import DeskState

        # ── FIX-4: run macro / technical / sentiment with individual timeouts
        macro_res, macro_err = await _with_timeout(
            self._macro.run(state, self.openai_key), "macro", self.timeout
        )
        tech_res, tech_err = await _with_timeout(
            self._technical.run(state, self.openai_key), "technical", self.timeout
        )
        sent_res, sent_err = await _with_timeout(
            self._sentiment.run(state, self.openai_key), "sentiment", self.timeout
        )

        # Build state from whatever succeeded
        extra_dnt_flags: list[str] = []

        if macro_res is None:
            logger.warning("Macro agent failed (%s) — using neutral fallback", macro_err)
            extra_dnt_flags.append(macro_err or "macro_failed")
        if tech_res is None:
            logger.warning("Technical agent failed (%s) — price levels will be zero", tech_err)
            extra_dnt_flags.extend(_zero_price_fallback_flags("technical"))
        if sent_res is None:
            logger.warning("Sentiment agent failed (%s)", sent_err)
            extra_dnt_flags.append(sent_err or "sentiment_failed")

        # Write partial results into state
        if macro_res:
            state = {**state, "macro_view": {
                "direction":        macro_res.direction,
                "conviction":       macro_res.conviction,
                "regime":           macro_res.regime,
                "macro_context":    macro_res.macro_context,
                "key_factors":      macro_res.key_factors,
                "fii_outlook":      macro_res.fii_outlook,
                "rate_environment": macro_res.rate_environment,
                "warning_flags":    macro_res.warning_flags,
            }}
        if tech_res:
            state = {**state, "technical_view": {
                "direction":      tech_res.direction,
                "conviction":     tech_res.conviction,
                "technical_view": tech_res.technical_view,
                "entry_low":      tech_res.entry_low,
                "entry_high":     tech_res.entry_high,
                "stop_loss":      tech_res.stop_loss,
                "target_1":       tech_res.target_1,
                "target_2":       tech_res.target_2,
                "key_levels":     tech_res.key_levels,
                "pattern":        tech_res.pattern,
            }}
        if sent_res:
            state = {**state, "sentiment_view": {
                "crowd_direction":  sent_res.crowd_direction,
                "crowd_conviction": sent_res.crowd_conviction,
                "simulation_weight": sent_res.simulation_weight,
                "crowd_narrative":  sent_res.crowd_narrative,
                "key_crowd_insight": sent_res.key_crowd_insight,
                "contrarian_signal": sent_res.contrarian_signal,
            }}

        # ── Risk agent (needs macro + technical) ─────────────────────────────
        risk_res, risk_err = await _with_timeout(
            self._risk.run(state, self.openai_key), "risk", self.timeout
        )
        if risk_res is None:
            logger.warning("Risk agent failed (%s)", risk_err)
            extra_dnt_flags.append(risk_err or "risk_failed")
        else:
            # FIX-11: Reconcile invalidation_level with technical stop_loss
            # Risk's invalidation_level wins only if technical stop_loss is zero
            tech_stop = float(state.get("technical_view", {}).get("stop_loss", 0.0))
            risk_invalid = float(risk_res.invalidation_level) if risk_res.invalidation_level else 0.0

            if tech_stop > 1e-6 and risk_invalid > 1e-6 and abs(tech_stop - risk_invalid) / tech_stop > 0.02:
                logger.warning(
                    "Invalidation level mismatch: tech_stop=%.4f  risk_invalid=%.4f — using tech_stop",
                    tech_stop, risk_invalid,
                )
                risk_invalid = tech_stop * 0.985  # slight buffer below structure

            state = {**state, "risk_params": {
                "approved":           risk_res.approved,
                "max_position_pct":   risk_res.max_position_pct,
                "invalidation_level": risk_invalid,
                "do_not_trade_flags": list(risk_res.do_not_trade_flags) + extra_dnt_flags,
                "warning_flags":      risk_res.warning_flags,
                "risk_reward":        risk_res.risk_reward,
                "conviction_modifier": risk_res.conviction_modifier,
                "rationale":          risk_res.rationale,
            }}

        # ── Coordinator ───────────────────────────────────────────────────────
        coord_result = await self._coordinator.synthesise(state, self.anthropic_key)

        # Merge any extra do_not_trade flags accumulated from agent failures
        if extra_dnt_flags:
            existing = list(coord_result.get("do_not_trade_flags", []))
            coord_result["do_not_trade_flags"] = list(set(existing + extra_dnt_flags))

        state["final_signal"] = coord_result
        return state
