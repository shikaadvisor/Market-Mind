"""
desk/agents/sentiment.py
========================
The Sentiment Interpreter — the bridge between the Digital World and the desk.

Reads the SimulationReport brief (produced by the simulation coordinator)
and translates it into a structured, desk-ready crowd intelligence view.

This agent does NOT look at charts or macro. Its entire job is:
1. Read what 10,210 agents produced
2. Identify the signal within the noise
3. Assess whether it's a contrarian or confirming input
4. Recommend a weight for the coordinator to apply
5. Surface the single most actionable crowd insight

Design principle: the Sentiment Interpreter must resist the temptation
to just summarise the brief. It must INTERPRET — add value by identifying
the pattern, assessing the historical analogue, and making a recommendation.

Model: GPT-4o
Runs: in parallel with Macro + Technical (only needs simulation data)
Writes to state: sentiment_view (SentimentOutput)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from desk.agents.base import (
    BaseAgent, DeskState, SentimentOutput,
    POINT_IN_TIME_INSTRUCTION, clean_json,
)


SYSTEM_PROMPT = """You are the Sentiment Interpreter on a sophisticated trading desk.
Your role: read the Digital World simulation output — the emergent behaviour of 10,000+
AI agents representing real market participants — and translate it into actionable
crowd intelligence for the desk coordinator.

You are NOT a summary bot. You add value by:
1. Pattern recognition: does this crowd behaviour match a known historical analogue?
   (e.g. "retail panic + institutional buy = March 2020 bottom pattern")
2. Divergence detection: the most powerful signals come from institutional vs retail splits
3. Weight recommendation: how much should the coordinator rely on this simulation?
   — High weight (0.25–0.30): clear divergence, high narrative confidence
   — Moderate weight (0.15–0.20): mixed signals, moderate confidence
   — Low weight (0.10): noisy, contradictory, or low-confidence simulation
4. Contrarian identification: if the crowd is overwhelmingly one-sided, flag it

Key insight: markets tend to mean-revert when retail herding is extreme and
institutions are positioned opposite. The herd is often right in the middle of
a trend but wrong at the extremes.

Respond ONLY with valid JSON. No preamble. No markdown fences."""


def _build_user_prompt(state: DeskState) -> str:
    pit    = state.get("point_in_time_date", str(__import__("datetime").date.today()))
    brief  = state.get("simulation_brief", "No simulation data available.")
    metrics = state.get("simulation_metrics", {})
    event  = state.get("event_description", "")

    metrics_str = ""
    if metrics:
        metrics_str = (
            f"\nRaw simulation metrics:\n"
            f"  aggregate_sentiment:     {metrics.get('aggregate_sentiment', 'N/A')}\n"
            f"  retail_sentiment:        {metrics.get('retail_sentiment', 'N/A')}\n"
            f"  institutional_sentiment: {metrics.get('institutional_sentiment', 'N/A')}\n"
            f"  herd_index:              {metrics.get('herd_index', 'N/A')}\n"
            f"  panic_probability:       {metrics.get('panic_probability', 'N/A')}\n"
            f"  narrative_momentum:      {metrics.get('narrative_momentum', 'N/A')}\n"
            f"  contrarian_pressure:     {metrics.get('contrarian_pressure', 'N/A')}\n"
        )

    return f"""{POINT_IN_TIME_INSTRUCTION.format(date=pit)}

Event: {event}
{metrics_str}

SIMULATION REPORT:
{brief}

Interpret this simulation output for the desk coordinator.
Respond ONLY with this JSON:
{{
  "crowd_direction": "bullish" | "bearish" | "neutral" | "mixed",
  "crowd_conviction": 0.0-1.0,
  "simulation_weight": 0.10-0.30,
  "crowd_narrative": "your interpretation of what the crowd is doing and why it matters for trading (80-200 words)",
  "key_crowd_insight": "the single most actionable crowd signal in one sentence (max 25 words)",
  "contrarian_signal": true | false,
  "historical_analogue": "brief description of a similar crowd pattern from market history (max 30 words)"
}}"""


class SentimentAgent(BaseAgent):
    """Sentiment Interpreter desk agent."""

    def __init__(self, mock: bool = False, mock_client: Any = None):
        super().__init__(
            model       = "gpt-4o",
            temperature = 0.25,
            max_tokens  = 800,
            mock        = mock,
            mock_client = mock_client,
        )

    async def run(self, state: DeskState, api_key: str) -> SentimentOutput:
        client      = self._get_client(api_key)
        user_prompt = _build_user_prompt(state)

        try:
            raw    = await self._call_openai(client, SYSTEM_PROMPT, user_prompt)
            parsed = json.loads(clean_json(raw))

            weight = float(parsed.get("simulation_weight", 0.20))
            weight = max(0.10, min(0.30, weight))

            return SentimentOutput(
                crowd_direction   = parsed.get("crowd_direction", "mixed").lower(),
                crowd_conviction  = self._parse_float(parsed.get("crowd_conviction", 0.5)),
                simulation_weight = weight,
                crowd_narrative   = str(parsed.get("crowd_narrative", ""))[:1000],
                key_crowd_insight = str(parsed.get("key_crowd_insight", ""))[:150],
                contrarian_signal = bool(parsed.get("contrarian_signal", False)),
                raw               = raw,
            )
        except Exception as e:
            return SentimentOutput(
                crowd_direction="mixed", crowd_conviction=0.30,
                simulation_weight=0.10,
                crowd_narrative=f"Sentiment interpretation unavailable: {e}",
                key_crowd_insight="Simulation data could not be parsed.",
                contrarian_signal=False,
                raw=str(e),
            )

    def run_sync(self, state: DeskState, api_key: str) -> SentimentOutput:
        return asyncio.run(self.run(state, api_key))


# ---------------------------------------------------------------------------
# Mock client
# ---------------------------------------------------------------------------

class _MockSentimentClient:
    @property
    def chat(self): return self
    @property
    def completions(self): return self

    async def create(self, model, messages, temperature, max_tokens, **kw):
        user_msg = messages[-1]["content"] if messages else ""

        is_fear   = "fear" in user_msg.lower() or "-0." in user_msg
        is_contra = "contrarian" in user_msg.lower() or (
            "institutional_sentiment" in user_msg and
            float(user_msg.split("institutional_sentiment:")[1].split("\n")[0].strip())
            > 0 if "institutional_sentiment:" in user_msg else False
        )

        if is_fear and is_contra:
            payload = {
                "crowd_direction":    "bearish",
                "crowd_conviction":   0.72,
                "simulation_weight":  0.28,
                "crowd_narrative": (
                    "The simulation reveals a classic retail fear vs institutional divergence. "
                    "78% of the 10,000-agent retail crowd is bearish with a herd index of 55%, "
                    "indicating correlated fear-driven selling. However, Tier 1 institutional "
                    "agents (deep thinkers representing hedge funds, macro PMs, and contrarian "
                    "investors) show a bullish lean — 43% bull vs 60% bear, with significant "
                    "neutral positioning indicating caution rather than conviction selling. "
                    "This retail fear + institutional non-commitment pattern is historically "
                    "a mean-reversion setup, particularly when panic probability is below 15%."
                ),
                "key_crowd_insight": (
                    "Retail herd (78% bear) diverges from institutions (43% bull) — "
                    "classic contrarian long setup."
                ),
                "contrarian_signal":    True,
                "historical_analogue": "Similar to Oct 2022 and Dec 2023 capitulation lows — "
                                       "retail panic + institutional accumulation preceded sharp recoveries.",
            }
        elif is_fear:
            payload = {
                "crowd_direction":    "bearish",
                "crowd_conviction":   0.65,
                "simulation_weight":  0.18,
                "crowd_narrative": (
                    "The simulation shows broad bearish consensus across all tiers — retail, "
                    "informed crowd, and institutional agents are all positioned defensively. "
                    "This unanimous alignment reduces the contrarian value of the simulation "
                    "signal. When all tiers agree, the simulation is confirming rather than "
                    "providing new information. Use with moderate weight as a momentum "
                    "confirmation rather than a contrarian edge."
                ),
                "key_crowd_insight": (
                    "All tiers bearish — confirming momentum signal, low contrarian value."
                ),
                "contrarian_signal":    False,
                "historical_analogue": "Consensus bearish environments like Sep 2022 "
                                       "can persist for weeks before bottoming.",
            }
        else:
            payload = {
                "crowd_direction":    "bullish",
                "crowd_conviction":   0.68,
                "simulation_weight":  0.22,
                "crowd_narrative": (
                    "The simulation shows bullish momentum building across tiers. "
                    "The retail crowd (68% bull) and informed crowd (64% bull) are aligned, "
                    "while institutional deep thinkers are 50% bull with notable neutrals — "
                    "indicating institutions are constructive but not aggressively positioned. "
                    "Low panic probability and moderate herding suggest orderly enthusiasm "
                    "rather than irrational exuberance. The narrative momentum (+0.38) "
                    "indicates the positive story is still spreading, not peaking."
                ),
                "key_crowd_insight": (
                    "Bullish consensus with institutional confirmation — "
                    "momentum signal, not contrarian."
                ),
                "contrarian_signal":    False,
                "historical_analogue": "Constructive crowd with institutional support — "
                                       "similar to early stages of 2023 and 2024 bull runs.",
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
        "point_in_time_date": str(__import__("datetime").date.today()),
        "event_description":  "US CPI 3.4% vs 2.9% — hot inflation",
        "simulation_brief":   (
            "Aggregate sentiment: -0.32 (FEAR). "
            "Retail 78% bear, institutional 43% bull. "
            "Herd index 55%. Panic prob 8%. Contrarian pressure 0%."
        ),
        "simulation_metrics": {
            "aggregate_sentiment":     -0.32,
            "retail_sentiment":        -0.45,
            "institutional_sentiment":  0.18,
            "herd_index":               0.55,
            "panic_probability":        0.08,
            "narrative_momentum":       -0.54,
            "contrarian_pressure":      0.43,
        },
    }

    agent  = SentimentAgent(mock=True, mock_client=_MockSentimentClient())
    result = asyncio.run(agent.run(state, api_key="mock"))

    print(f"\n── Sentiment Interpreter ──")
    print(f"Crowd direction:  {result.crowd_direction}")
    print(f"Crowd conviction: {result.crowd_conviction:.0%}")
    print(f"Sim weight:       {result.simulation_weight:.0%}")
    print(f"Contrarian:       {result.contrarian_signal}")
    print(f"Key insight:      {result.key_crowd_insight}")
    print(f"Narrative:\n  {result.crowd_narrative[:200]}...")
