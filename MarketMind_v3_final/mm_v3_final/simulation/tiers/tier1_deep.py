"""
simulation/tiers/tier1_deep.py
===============================
The deep thinkers — 10 GPT-4o agents with full reasoning chains.

These are the most sophisticated agents in the simulation. Each represents
a genuinely different class of market participant: global macro PM, domestic
contrarian, quant PM, EM specialist, event-driven activist, volatility
strategist, credit analyst, long-only value PM, distressed specialist,
systematic CTA.

Design choices:
- Full GPT-4o, not mini. These agents THINK, not just react.
- Structured JSON with rich fields: directional view, conviction, 3-month
  price target, scenario analysis, tail risk, cross-asset signals.
- Run concurrently (asyncio) but fewer agents so latency is fine (~8-15s).
- Each agent also receives: Tier 3 crowd state + Tier 2 aggregate sentiment.
  Deep thinkers know what the crowd is doing — just like real smart money.
- Temperature 0.4: lower randomness. These are "experts" — less noise,
  more signal.
- Cost: ~10 calls × ~1,200 tokens avg = ~12,000 tokens ≈ $0.12 at GPT-4o rates.

Run standalone:
    python simulation/tiers/tier1_deep.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from simulation.personas import TIER1_PERSONAS, Persona
from simulation.tiers.tier2_async import Tier2Result
from simulation.tiers.tier3_mesa import MarketEvent, Tier3Result


# ---------------------------------------------------------------------------
# Rich response schema for Tier 1
# ---------------------------------------------------------------------------

@dataclass
class Tier1AgentResponse:
    persona_name:        str
    stance:              str         # "bullish" | "bearish" | "neutral"
    conviction:          float       # 0.0 – 1.0
    sentiment_score:     float       # -1.0 – 1.0
    three_month_target:  str         # e.g. "24,500–25,200" or "bearish, no target"
    key_bull_factor:     str         # strongest bullish point
    key_bear_factor:     str         # strongest bearish point
    tail_risk:           str         # the thing that keeps them up at night
    crowd_assessment:    str         # their read on the crowd (contrarian or confirming)
    leading_indicator:   str         # the one metric they're watching most
    full_reasoning:      str         # the complete reasoning chain (for audit)
    raw_response:        str         # raw LLM output
    is_fallback:         bool = False


@dataclass
class Tier1Result:
    responses:           list[Tier1AgentResponse]
    n_agents:            int
    n_successful:        int
    n_fallback:          int
    elapsed_sec:         float
    bullish_pct:         float
    bearish_pct:         float
    neutral_pct:         float
    avg_conviction:      float
    avg_sentiment:       float
    contrarian_pct:      float       # % of agents explicitly fading the crowd
    notable_behaviors:   list[str] = field(default_factory=list)

    def to_tier_snapshot_dict(self) -> dict:
        return {
            "tier":             1,
            "agent_count":      self.n_agents,
            "bullish_pct":      round(self.bullish_pct, 4),
            "bearish_pct":      round(self.bearish_pct, 4),
            "neutral_pct":      round(self.neutral_pct, 4),
            "avg_conviction":   round(self.avg_conviction, 4),
            "notable_behaviors": self.notable_behaviors,
        }


# ---------------------------------------------------------------------------
# Prompt builder for Tier 1
# ---------------------------------------------------------------------------

def _build_tier1_prompt(
    persona:        Persona,
    event:          MarketEvent,
    tier3_snapshot: dict,
    tier2_result:   Tier2Result | None,
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for a Tier 1 deep thinker."""

    # Build crowd intelligence section — Tier 1 agents see the full crowd picture
    crowd_section = (
        f"CROWD INTELLIGENCE (what the market is doing):\n"
        f"- Price shock: {event.price_shock:+.1%}\n"
        f"- News sentiment: {event.news_sentiment:+.2f}\n"
        f"- Uncertainty level: {event.uncertainty:.0%}\n"
        f"- Retail crowd (10,000 agents): "
        f"  {tier3_snapshot['bullish_pct']:.0%} bull / "
        f"  {tier3_snapshot['bearish_pct']:.0%} bear\n"
        f"- Retail herding: {tier3_snapshot.get('herd_index', 0.5):.0%} "
        f"(1.0 = full herd)\n"
        f"- Retail panic: {tier3_snapshot.get('panic_pct', 0.0):.1%} in panic mode\n"
    )

    if tier2_result is not None:
        crowd_section += (
            f"- Informed crowd (200 semi-sophisticated agents): "
            f"  {tier2_result.bullish_pct:.0%} bull / "
            f"  {tier2_result.bearish_pct:.0%} bear  "
            f"  (avg conviction: {tier2_result.avg_conviction:.2f})\n"
        )
        if tier2_result.notable_behaviors:
            crowd_section += (
                f"- Informed crowd top themes: "
                + ", ".join(tier2_result.notable_behaviors[:2]) + "\n"
            )

    user_prompt = (
        f"Market event: {event.description}\n\n"
        f"{crowd_section}\n"
        f"Apply your specific analytical framework to this situation.\n"
        f"Be rigorous, specific, and authentic to your persona.\n\n"
        f"Respond ONLY with a JSON object in this exact format — "
        f"no preamble, no markdown fences:\n"
        f'{{\n'
        f'  "stance": "bullish" | "bearish" | "neutral",\n'
        f'  "conviction": 0.0-1.0,\n'
        f'  "three_month_target": "price range or directional description",\n'
        f'  "key_bull_factor": "strongest bullish argument (max 25 words)",\n'
        f'  "key_bear_factor": "strongest bearish argument (max 25 words)",\n'
        f'  "tail_risk": "the scenario that would prove you badly wrong (max 20 words)",\n'
        f'  "crowd_assessment": "contrarian_fade | confirming | mixed",\n'
        f'  "leading_indicator": "the one metric you are watching most closely (max 15 words)",\n'
        f'  "full_reasoning": "your complete analytical reasoning (150-300 words)"\n'
        f'}}'
    )

    return persona.system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Single agent call
# ---------------------------------------------------------------------------

async def _call_tier1_agent(
    client:         Any,
    persona:        Persona,
    event:          MarketEvent,
    tier3_snapshot: dict,
    tier2_result:   Tier2Result | None,
    semaphore:      asyncio.Semaphore,
    temperature:    float = 0.40,
    max_tokens:     int   = 1_200,
) -> Tier1AgentResponse:

    system_prompt, user_prompt = _build_tier1_prompt(
        persona, event, tier3_snapshot, tier2_result
    )

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model       = "gpt-4o",
                temperature = temperature,
                max_tokens  = max_tokens,
                messages    = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            raw   = response.choices[0].message.content.strip()
            clean = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)

            stance     = parsed.get("stance", "neutral").lower()
            conviction = float(parsed.get("conviction", 0.5))
            conviction = max(0.0, min(1.0, conviction))

            if stance == "bullish":
                sentiment_score = conviction
            elif stance == "bearish":
                sentiment_score = -conviction
            else:
                sentiment_score = 0.0

            crowd_assessment = parsed.get("crowd_assessment", "mixed").lower()

            return Tier1AgentResponse(
                persona_name       = persona.name,
                stance             = stance,
                conviction         = conviction,
                sentiment_score    = sentiment_score,
                three_month_target = str(parsed.get("three_month_target", ""))[:80],
                key_bull_factor    = str(parsed.get("key_bull_factor", ""))[:120],
                key_bear_factor    = str(parsed.get("key_bear_factor", ""))[:120],
                tail_risk          = str(parsed.get("tail_risk", ""))[:100],
                crowd_assessment   = crowd_assessment,
                leading_indicator  = str(parsed.get("leading_indicator", ""))[:80],
                full_reasoning     = str(parsed.get("full_reasoning", ""))[:1500],
                raw_response       = raw,
                is_fallback        = False,
            )

        except Exception as e:
            return Tier1AgentResponse(
                persona_name       = persona.name,
                stance             = "neutral",
                conviction         = 0.35,
                sentiment_score    = 0.0,
                three_month_target = "unknown",
                key_bull_factor    = "parse error",
                key_bear_factor    = f"error: {str(e)[:60]}",
                tail_risk          = "unknown",
                crowd_assessment   = "mixed",
                leading_indicator  = "unknown",
                full_reasoning     = f"FALLBACK — error: {str(e)}",
                raw_response       = f"ERROR: {str(e)}",
                is_fallback        = True,
            )


# ---------------------------------------------------------------------------
# Tier 1 runner
# ---------------------------------------------------------------------------

class Tier1Simulation:
    """
    Runs all Tier 1 deep thinker agents concurrently.
    Each agent is one of the 10 unique persona types.
    """

    def __init__(
        self,
        temperature: float = 0.40,
        max_tokens:  int   = 1_200,
        verbose:     bool  = True,
    ):
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.verbose     = verbose
        self._all_personas = TIER1_PERSONAS
        self.personas    = TIER1_PERSONAS   # default; overridden per asset class

    async def run_async(
        self,
        event:         MarketEvent,
        tier3_result:  Tier3Result,
        tier2_result:  Tier2Result | None,
        api_key:       str,
    ) -> Tier1Result:

        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("pip install openai")

        # Select personas relevant to this asset class
        from simulation.personas import get_tier1_personas_for
        asset_class = getattr(tier3_result, "_asset_class", "equity_index")
        self.personas = get_tier1_personas_for(asset_class)
        if len(self.personas) < 3:
            self.personas = self._all_personas[:10]

        client    = AsyncOpenAI(api_key=api_key)
        semaphore = asyncio.Semaphore(len(self.personas))

        t3_snap = tier3_result.to_tier_snapshot_dict()
        t3_snap["herd_index"] = tier3_result.final_snapshot.herd_index
        t3_snap["panic_pct"]  = tier3_result.final_snapshot.panic_agents_pct

        t0 = time.perf_counter()

        if self.verbose:
            print(f"  [Tier1] Firing {len(self.personas)} GPT-4o deep thinkers "
                  f"concurrently...")

        tasks = [
            _call_tier1_agent(
                client, persona, event, t3_snap, tier2_result,
                semaphore, self.temperature, self.max_tokens,
            )
            for persona in self.personas
        ]
        responses: list[Tier1AgentResponse] = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0

        # Aggregate
        n_bull     = sum(1 for r in responses if r.stance == "bullish")
        n_bear     = sum(1 for r in responses if r.stance == "bearish")
        n_neutral  = sum(1 for r in responses if r.stance == "neutral")
        n_fallback = sum(1 for r in responses if r.is_fallback)
        n          = len(responses)

        sentiments  = [r.sentiment_score for r in responses]
        convictions = [r.conviction      for r in responses]

        # Contrarian: agents explicitly fading the crowd
        contrarian_count = sum(
            1 for r in responses if r.crowd_assessment == "contrarian_fade"
        )

        # Notable behaviors: pull key insights from deep thinkers
        notable = []
        for r in responses:
            if r.is_fallback:
                continue
            if r.crowd_assessment == "contrarian_fade":
                notable.append(
                    f"{r.persona_name}: contrarian {r.stance} "
                    f"(conviction={r.conviction:.0%}) — {r.key_bull_factor[:40]}"
                    if r.stance == "bullish" else
                    f"{r.persona_name}: contrarian {r.stance} "
                    f"(conviction={r.conviction:.0%}) — {r.key_bear_factor[:40]}"
                )
            if r.conviction >= 0.75:
                notable.append(
                    f"HIGH CONVICTION: {r.persona_name} is {r.stance} "
                    f"at {r.conviction:.0%}"
                )

        result = Tier1Result(
            responses          = responses,
            n_agents           = n,
            n_successful       = n - n_fallback,
            n_fallback         = n_fallback,
            elapsed_sec        = elapsed,
            bullish_pct        = n_bull    / n,
            bearish_pct        = n_bear    / n,
            neutral_pct        = n_neutral / n,
            avg_conviction     = sum(convictions) / n,
            avg_sentiment      = sum(sentiments)  / n,
            contrarian_pct     = contrarian_count / n,
            notable_behaviors  = notable[:6],   # cap at 6
        )

        if self.verbose:
            self._print_summary(result)

        return result

    def run(
        self,
        event:        MarketEvent,
        tier3_result: Tier3Result,
        tier2_result: Tier2Result | None,
        api_key:      str,
    ) -> Tier1Result:
        return asyncio.run(
            self.run_async(event, tier3_result, tier2_result, api_key)
        )

    def _print_summary(self, r: Tier1Result) -> None:
        print(f"\n  {'─'*56}")
        print(f"  Tier 1 Complete — {r.n_agents} deep thinkers")
        print(f"  {'─'*56}")
        print(f"  Runtime:         {r.elapsed_sec:.1f}s")
        print(f"  Stance:          {r.bullish_pct:.0%} bull / "
              f"{r.bearish_pct:.0%} bear / {r.neutral_pct:.0%} neutral")
        print(f"  Avg sentiment:   {r.avg_sentiment:+.4f}")
        print(f"  Avg conviction:  {r.avg_conviction:.4f}")
        print(f"  Contrarian:      {r.contrarian_pct:.0%} fading the crowd")
        print(f"  Successful:      {r.n_successful}/{r.n_agents}")
        if r.notable_behaviors:
            print(f"  Key insights:")
            for b in r.notable_behaviors[:4]:
                print(f"    · {b[:80]}")
        print()
        print(f"  Individual stances:")
        for resp in r.responses:
            flag = "⚡" if resp.conviction >= 0.75 else " "
            ct   = "↙CONTRA" if resp.crowd_assessment == "contrarian_fade" else "       "
            fb   = " [fallback]" if resp.is_fallback else ""
            print(f"    {flag} {resp.persona_name:<38} "
                  f"{resp.stance:<8} {conviction_bar(resp.conviction)} "
                  f"{ct}{fb}")
        print(f"  {'─'*56}\n")


def conviction_bar(c: float) -> str:
    filled = int(c * 8)
    return "█" * filled + "░" * (8 - filled)


# ---------------------------------------------------------------------------
# Standalone demo with mock client
# ---------------------------------------------------------------------------

class _MockAsyncOpenAI:
    def __init__(self, **kwargs): pass

    @property
    def chat(self): return self

    @property
    def completions(self): return self

    async def create(self, model, messages, temperature, max_tokens, **kwargs):
        await asyncio.sleep(0.05 + 0.03 * (hash(str(messages)) % 10))

        user_msg = messages[-1]["content"] if messages else ""
        is_bearish = "hot inflation" in user_msg.lower() or \
                     "breakdown" in user_msg.lower()

        persona_name = ""
        for msg in messages:
            if msg["role"] == "system":
                # Extract persona from system prompt first line
                first_line = msg["content"].split("\n")[0]
                persona_name = first_line[:40]
                break

        # Deep thinkers are more contrarian — more bullish when event is bearish
        is_contrarian_persona = any(
            word in persona_name.lower()
            for word in ["contrarian", "distressed", "value", "systematic"]
        )

        if is_bearish and is_contrarian_persona:
            stance = "bullish"
            conviction = round(0.60 + hash(persona_name) % 20 / 100, 2)
            crowd_assessment = "contrarian_fade"
        elif is_bearish:
            stance = "bearish"
            conviction = round(0.55 + hash(persona_name) % 25 / 100, 2)
            crowd_assessment = "confirming"
        else:
            stance = "bullish"
            conviction = round(0.58 + hash(persona_name) % 22 / 100, 2)
            crowd_assessment = "confirming"

        conviction = min(0.95, conviction)

        payload = {
            "stance":            stance,
            "conviction":        conviction,
            "three_month_target": "24,200–25,500" if stance == "bullish" else "22,800–23,800",
            "key_bull_factor":   "Domestic consumption resilient; RBI policy supportive of growth",
            "key_bear_factor":   "Fed policy uncertainty creates sustained FII outflow risk",
            "tail_risk":         "Sudden credit event or geopolitical escalation beyond India",
            "crowd_assessment":  crowd_assessment,
            "leading_indicator": "FII net flow weekly trend and USD/INR rate",
            "full_reasoning": (
                f"As a sophisticated market participant, I see this event as "
                f"{'a temporary dislocation that creates opportunity' if is_contrarian_persona and is_bearish else 'a genuine risk that warrants caution'}. "
                f"The crowd is {'overreacting to a global macro signal that India can absorb' if is_contrarian_persona and is_bearish else 'correctly pricing in the risk'}. "
                f"My conviction is {conviction:.0%} based on my analytical framework. "
                f"I am watching FII flow data closely as the leading indicator. "
                f"The domestic macro picture remains supportive, but global liquidity conditions "
                f"create near-term headwinds that need to be respected."
            ),
        }

        class _Msg:
            content = json.dumps(payload)
        class _Choice:
            message = _Msg()
        class _Resp:
            choices = [_Choice()]
        return _Resp()


async def _demo_with_mock():
    import types
    openai_mock = types.ModuleType("openai")
    openai_mock.AsyncOpenAI = _MockAsyncOpenAI
    sys.modules.setdefault("openai", openai_mock)

    from simulation.tiers.tier3_mesa import Tier3Simulation
    from simulation.tiers.tier2_async import Tier2Simulation, _MockAsyncOpenAI as Mock2

    event = MarketEvent(
        description    = "US CPI prints 3.4% vs 2.9% expected — hot inflation, "
                         "Fed rate cuts pushed back to Q4 2026",
        price_shock    = -0.018,
        news_sentiment = -0.65,
        uncertainty    = 0.70,
        is_systemic    = True,
        sector_impact  = "all",
    )

    print("\n" + "="*60)
    print("  Tier 1 Deep Thinkers — Demo")
    print("="*60)

    # Tier 3
    print("\n[Step 1] Running Tier 3...")
    t3 = Tier3Simulation(n_agents=10_000, n_steps=50, verbose=True)
    t3._initialise_population()
    t3_result = t3.run(event)

    # Tier 2 (mock)
    print("\n[Step 2] Running Tier 2 (mock)...")
    import openai as _oai
    _oai.AsyncOpenAI = Mock2
    t2_sim = Tier2Simulation(n_agents=200, batch_size=20, verbose=True)
    t2_result = await t2_sim.run_async(event, t3_result, api_key="mock")

    # Tier 1 (mock)
    print("\n[Step 3] Running Tier 1 deep thinkers (mock)...")
    t1_sim = Tier1Simulation(verbose=True)
    t1_result = await t1_sim.run_async(event, t3_result, t2_result, api_key="mock")

    print("\n" + "─"*60)
    print("Complete 3-tier simulation finished.")
    print(f"Total agents: {t3_result.n_agents + t2_result.n_agents + t1_result.n_agents:,}")
    print(f"\nSentiment by tier:")
    print(f"  Tier 3 (crowd):        {t3_result.final_sentiment:+.4f}")
    print(f"  Tier 2 (informed):     {t2_result.avg_sentiment:+.4f}")
    print(f"  Tier 1 (deep think):   {t1_result.avg_sentiment:+.4f}")
    print(f"\nKey insight: Tier 1 vs Tier 3 divergence = "
          f"{t1_result.avg_sentiment - t3_result.final_sentiment:+.4f} "
          f"({'contrarian setup' if abs(t1_result.avg_sentiment - t3_result.final_sentiment) > 0.2 else 'consensus'})")


if __name__ == "__main__":
    asyncio.run(_demo_with_mock())
