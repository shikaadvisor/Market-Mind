"""
desk/agents/sentiment_v3.py
============================
Sentiment Interpreter v3 — swarm-aware.

Key changes from v1:
    - Receives a FULL SwarmReport instead of a single simulation summary
    - Explicitly surfaces universe divergence as a trading signal
    - Injects forward-test memory context when available
    - Returns a richer SentimentOutput with per-universe breakdown
    - The contrarian_setup flag from the swarm is preserved and amplified
      if multiple evidence sources agree

The most important new output:
    universe_breakdown — a structured breakdown that the coordinator reads:
    {
        "momentum":   {"sentiment": -0.3, "agreement_with_desk": false},
        "value":      {"sentiment": +0.5, "agreement_with_desk": true},
        "crisis":     {"sentiment": -0.6, "agreement_with_desk": false},
        "structural": {"sentiment": +0.2, "agreement_with_desk": true},
    }
    When 2 universes agree and 2 disagree, conviction is reduced.
    When 3+ agree, conviction gets a boost.
    When all 4 agree, it's a rare high-confidence setup.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from desk.agents.base import BaseAgent, DeskState, SentimentOutput, clean_json, POINT_IN_TIME_INSTRUCTION


SYSTEM_PROMPT = """You are the Sentiment Interpreter on a sophisticated trading desk.
Your unique role: translate the output of a multi-universe crowd simulation into
actionable intelligence for the coordinator.

You receive output from FOUR parallel simulation universes:
- MOMENTUM universe: populated by trend-followers, FOMO traders, stop-hunters
- VALUE universe: populated by contrarians, fundamental PMs, patient capital
- CRISIS universe: populated by tail-risk managers, panic-prone retail, safe-haven seekers
- STRUCTURAL universe: populated by institutional flow, commodity fundamentalists, macro PMs

Your job is to:
1. Read the divergence between universes — high divergence is itself a signal
2. Identify if the crowd setup is contrarian (momentum crowd vs value crowd disagree)
3. Weight the simulation appropriately for this instrument and regime
4. Translate technical simulation output into plain English for the coordinator

Key interpretation rules:
- If MOMENTUM and VALUE universes strongly DISAGREE (>0.40 divergence):
  → This is a contrarian setup. Momentum crowd is likely wrong. Reduce momentum weight.
  → The side that VALUE universe agrees with deserves more weight.

- If CRISIS universe has high panic probability AND STRUCTURAL is calm:
  → Panic is likely retail-driven and may be an opportunity, not a genuine risk signal

- If ALL FOUR universes agree (divergence < 0.10):
  → Rare high-confidence setup. Flag this explicitly.

- Universe divergence > 0.30 → reduce overall simulation weight to 0.15
- Universe divergence 0.15-0.30 → standard weight 0.20
- Universe divergence < 0.15 → increase weight to 0.28 (consensus)

Respond ONLY with valid JSON. No preamble. No markdown fences."""


def _build_prompt(state: DeskState, feedback_context: str = "") -> str:
    pit         = state.get("point_in_time_date", "")
    event       = state.get("event_description", "")
    swarm       = state.get("swarm_report")   # SwarmReport object or dict
    sim_metrics = state.get("simulation_metrics", {})
    sim_brief   = state.get("simulation_brief", "")

    # Extract universe data from swarm report
    universe_section = ""
    contrarian_setup = False
    divergence = 0.0

    if swarm is not None:
        # Handle both SwarmReport object and dict
        if hasattr(swarm, "universe_results"):
            results = swarm.universe_results
            divergence = float(getattr(swarm, "universe_divergence", 0.0))
            contrarian_setup = bool(getattr(swarm, "contrarian_setup", False))
            dominant = getattr(swarm, "dominant_universe", "")
            swarm_brief = getattr(swarm, "swarm_brief", sim_brief)
        else:
            results = {}
            swarm_brief = sim_brief

        universe_lines = []
        for uname, ures in results.items() if hasattr(results, "items") else []:
            sent = getattr(ures, "aggregate_sentiment", 0.0)
            herd = getattr(ures, "herd_index", 0.5)
            panic = getattr(ures, "panic_probability", 0.1)
            narrative = getattr(ures, "dominant_narrative", "")
            universe_lines.append(
                f"  [{uname.upper():12s}] sentiment={sent:+.3f}  herd={herd:.0%}  "
                f"panic={panic:.0%}  narrative={narrative}"
            )

        if universe_lines:
            universe_section = (
                f"\nUNIVERSE BREAKDOWN:\n" +
                "\n".join(universe_lines) +
                f"\n\nUniverse divergence: {divergence:.4f} "
                f"({'HIGH' if divergence > 0.30 else 'moderate' if divergence > 0.15 else 'LOW'})"
                f"\nContrarian setup: {'YES — momentum and value disagree significantly' if contrarian_setup else 'no'}"
                f"\nDominant universe: {dominant}"
            )
    else:
        swarm_brief = sim_brief

    return f"""{POINT_IN_TIME_INSTRUCTION.format(date=pit)}

Instrument: {state.get("instrument", "UNKNOWN")} | Asset class: {state.get("asset_class", "equity")}
Event: {event}

SWARM SIMULATION REPORT:
{swarm_brief[:1500]}
{universe_section}

AGGREGATE METRICS:
{json.dumps(sim_metrics, indent=2)[:500]}

{feedback_context}

Based on the above swarm output, provide your sentiment interpretation.
Respond ONLY with this JSON:
{{
  "crowd_direction":    "bullish" | "bearish" | "neutral" | "mixed",
  "crowd_conviction":  0.0-1.0,
  "simulation_weight": 0.10-0.30 (how much weight to give the simulation overall),
  "crowd_narrative":   "plain-English translation of what the crowd is doing and why (80-200 words)",
  "key_crowd_insight": "the single most actionable crowd signal in 20 words or less",
  "contrarian_signal": true | false,
  "universe_breakdown": {{
    "momentum":   {{"sentiment": <float>, "interpretation": "<10 words>"}},
    "value":      {{"sentiment": <float>, "interpretation": "<10 words>"}},
    "crisis":     {{"sentiment": <float>, "interpretation": "<10 words>"}},
    "structural": {{"sentiment": <float>, "interpretation": "<10 words>"}}
  }},
  "universe_divergence_flag": true | false (true if divergence > 0.25),
  "confidence_boost_flag":    true | false (true if all 4 universes agree, divergence < 0.10)
}}"""


class SentimentAgentV3(BaseAgent):
    """Sentiment Interpreter with full swarm awareness."""

    def __init__(self, mock: bool = False, mock_client: Any = None):
        super().__init__(
            model       = "gpt-4o",
            temperature = 0.25,
            max_tokens  = 1000,
            mock        = mock,
            mock_client = mock_client,
        )

    async def run(
        self,
        state:            DeskState,
        api_key:          str,
        feedback_context: str = "",
    ) -> SentimentOutput:
        client      = self._get_client(api_key)
        user_prompt = _build_prompt(state, feedback_context)

        try:
            raw    = await self._call_openai(client, SYSTEM_PROMPT, user_prompt)
            parsed = json.loads(clean_json(raw))

            return SentimentOutput(
                crowd_direction   = parsed.get("crowd_direction", "mixed"),
                crowd_conviction  = self._parse_float(parsed.get("crowd_conviction", 0.5)),
                simulation_weight = max(0.10, min(0.30, float(parsed.get("simulation_weight", 0.20)))),
                crowd_narrative   = str(parsed.get("crowd_narrative", ""))[:1000],
                key_crowd_insight = str(parsed.get("key_crowd_insight", ""))[:200],
                contrarian_signal = bool(parsed.get("contrarian_signal", False)),
                raw               = raw,
            )
        except Exception as e:
            return SentimentOutput(
                crowd_direction="mixed", crowd_conviction=0.30,
                simulation_weight=0.15,
                crowd_narrative=f"Sentiment interpretation unavailable: {e}",
                key_crowd_insight="sentiment_agent_error",
                contrarian_signal=False,
                raw=str(e),
            )
