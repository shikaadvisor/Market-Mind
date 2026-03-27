"""
desk/agents/devil_advocate.py
==============================
Risk 1 fix: Structural counterargument agent.

The coordinator's draft signal is passed to this agent BEFORE finalisation.
It is explicitly instructed to:
    1. Find the strongest single reason the signal is WRONG
    2. Identify which data points were IGNORED by the main agents
    3. Rate the counterargument strength (0=weak, 1=devastating)

If counterargument_strength >= 0.70, the coordinator receives it as an
additional input and must explicitly address it in coordinator_thesis.
This is not a veto — it is forced steelmanning.

The agent is intentionally adversarial. It is the only agent on the desk
whose job is to disagree. It runs after the coordinator's first draft
and its output is injected into the coordinator's second (final) call.

Two-pass coordinator design:
    Pass 1: coordinator sees macro + technical + sentiment + risk (standard)
    Pass 2: coordinator sees its own draft + devil's counterargument
            → must explicitly rebut or lower conviction

Cost: ~1 extra gpt-4o call per pipeline run (~$0.008).
Worth it: one genuine counterargument caught = one avoided loss.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from desk.agents.base import BaseAgent, DeskState, clean_json


SYSTEM_PROMPT = """You are the Devil's Advocate on a trading desk.
Your ONLY job: find the strongest possible reason the proposed signal is WRONG.

You are not balanced. You are not fair. You are adversarial by design.
The other agents have already made a bull or bear case. You make the OPPOSITE case
as hard as you can, using only evidence that the other agents IGNORED or MINIMISED.

Rules:
- Do NOT agree with the signal direction under any circumstance
- Focus on risks, ignored data, and structural weaknesses in the argument
- If the signal is long, argue why it should be short or neutral
- If the signal is short, argue why it should be long or neutral
- Be specific: cite actual data points that contradict the signal
- Do not use weasel words like "might" or "could" — be direct and aggressive

Your output is used to force the coordinator to steelman the opposite view.
A weak counterargument (score < 0.50) will be noted and dismissed.
A strong counterargument (score >= 0.70) requires the coordinator to
explicitly rebut it or lower conviction.

Respond ONLY with valid JSON. No preamble. No markdown fences."""


def _build_prompt(state: DeskState, coordinator_draft: dict) -> str:
    direction   = coordinator_draft.get("direction", "neutral")
    conviction  = coordinator_draft.get("conviction", 0.5)
    thesis      = coordinator_draft.get("coordinator_thesis", "")
    macro_ctx   = coordinator_draft.get("macro_context", "")
    tech_view   = coordinator_draft.get("technical_view", "")
    crowd_narr  = coordinator_draft.get("crowd_narrative", "")
    warning_f   = coordinator_draft.get("warning_flags", [])
    dnt_flags   = coordinator_draft.get("do_not_trade_flags", [])

    instrument  = state.get("instrument", "UNKNOWN")
    asset_class = state.get("asset_class", "equity")
    event       = state.get("event_description", "")
    mdata       = state.get("market_data", {})

    return f"""INSTRUMENT: {instrument} ({asset_class})
EVENT: {event}

PROPOSED SIGNAL (what you must ATTACK):
  Direction:  {direction.upper()}
  Conviction: {conviction:.0%}
  Thesis:     {thesis[:400]}

WHAT THE AGENTS SAID:
  Macro:     {macro_ctx[:300]}
  Technical: {tech_view[:300]}
  Crowd:     {crowd_narr[:200]}
  Warnings already flagged: {warning_f}
  Do-not-trade flags: {dnt_flags}

MARKET DATA AVAILABLE (look for what was IGNORED):
{json.dumps(mdata, indent=2)[:600]}

Find the strongest counterargument. What did every agent get wrong or ignore?
Respond ONLY with this JSON:
{{
  "counterargument": "your aggressive case against the signal (100-200 words)",
  "ignored_data_points": ["specific data or fact the agents glossed over", "..."],
  "key_risk": "the single most dangerous overlooked risk in 20 words",
  "counterargument_strength": 0.0-1.0,
  "recommended_conviction_cap": 0.0-1.0
}}"""


class DevilsAdvocateAgent(BaseAgent):

    def __init__(self, mock: bool = False, mock_client: Any = None):
        from utils.model_registry import get
        m = get("desk_devil")
        super().__init__(
            model       = m.name,
            temperature = m.temperature,
            max_tokens  = m.max_tokens,
            mock        = mock,
            mock_client = mock_client,
        )

    async def run(
        self,
        state:              DeskState,
        coordinator_draft:  dict,
        api_key:            str,
    ) -> dict:
        """
        Returns a devil's advocate dict with counterargument_strength.
        If strength < 0.40, treated as a non-event (weak counterargument).
        If strength >= 0.70, coordinator must address it.
        """
        client      = self._get_client(api_key)
        user_prompt = _build_prompt(state, coordinator_draft)

        try:
            raw    = await self._call_openai(client, SYSTEM_PROMPT, user_prompt)
            parsed = json.loads(clean_json(raw))
            strength = float(parsed.get("counterargument_strength", 0.5))
            cap      = float(parsed.get("recommended_conviction_cap", 1.0))
            return {
                "counterargument":         str(parsed.get("counterargument", ""))[:1000],
                "ignored_data_points":     list(parsed.get("ignored_data_points", []))[:5],
                "key_risk":                str(parsed.get("key_risk", ""))[:200],
                "counterargument_strength": round(min(1.0, max(0.0, strength)), 3),
                "recommended_conviction_cap": round(min(1.0, max(0.0, cap)), 3),
                "triggered":               strength >= 0.70,
            }
        except Exception as e:
            return {
                "counterargument": f"Devil's advocate unavailable: {e}",
                "ignored_data_points": [],
                "key_risk": "devil_advocate_error",
                "counterargument_strength": 0.0,
                "recommended_conviction_cap": 1.0,
                "triggered": False,
            }
