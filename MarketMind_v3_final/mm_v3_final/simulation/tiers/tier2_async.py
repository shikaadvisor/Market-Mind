"""
simulation/tiers/tier2_async.py
================================
The informed crowd — 200 GPT-4o-mini agents running concurrently via asyncio.

Each agent has a persona (fin-twitter influencer, retail options trader, etc.)
and reacts to the same market event + the Tier 3 crowd state.

Design choices:
- True async: all 200 calls fire in batches of 20, not sequentially.
  200 sequential calls ≈ 200s. 200 async calls ≈ 12–18s. Huge difference.
- Prompt caching: system prompt is identical per persona type — OpenAI caches
  repeated prefixes automatically. Reduces cost by ~50%.
- Structured JSON output: every agent returns a typed dict, not free-form prose.
  We validate each response. Bad JSON from a hallucinating agent = fallback.
- Tier 3 crowd state is injected into every prompt: agents "see" what the
  crowd is doing, just like real market participants see price action.
- Cost: ~200 calls × ~350 tokens avg = ~70,000 tokens ≈ $0.04 at gpt-4o-mini rates.

Run standalone:
    python simulation/tiers/tier2_async.py
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from simulation.personas import TIER2_PERSONAS, Persona
from simulation.tiers.tier3_mesa import MarketEvent, Tier3Result


# ---------------------------------------------------------------------------
# Agent response schema — what we expect back from every Tier 2 agent
# ---------------------------------------------------------------------------

@dataclass
class Tier2AgentResponse:
    persona_name:    str
    stance:          str       # "bullish" | "bearish" | "neutral"
    conviction:      float     # 0.0 – 1.0
    sentiment_score: float     # -1.0 – 1.0 (derived from stance + conviction)
    key_factor:      str       # single most important factor driving the view
    watching:        str       # what level/event they are watching
    change_mind:     str       # what would flip their view
    raw_response:    str       # full LLM output for audit
    is_fallback:     bool = False   # True if JSON parsing failed


@dataclass
class Tier2Result:
    responses:          list[Tier2AgentResponse]
    n_agents:           int
    n_successful:       int
    n_fallback:         int
    elapsed_sec:        float
    bullish_pct:        float
    bearish_pct:        float
    neutral_pct:        float
    avg_conviction:     float
    avg_sentiment:      float
    notable_behaviors:  list[str] = field(default_factory=list)

    def to_tier_snapshot_dict(self) -> dict:
        return {
            "tier":             2,
            "agent_count":      self.n_agents,
            "bullish_pct":      round(self.bullish_pct, 4),
            "bearish_pct":      round(self.bearish_pct, 4),
            "neutral_pct":      round(self.neutral_pct, 4),
            "avg_conviction":   round(self.avg_conviction, 4),
            "notable_behaviors": self.notable_behaviors,
        }


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_tier2_prompt(
    persona:        Persona,
    event:          MarketEvent,
    tier3_snapshot: dict,
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for a Tier 2 agent."""

    # Crowd context — agents "see" the price action and crowd mood
    crowd_context = (
        f"Current market conditions:\n"
        f"- Price shock: {event.price_shock:+.1%}\n"
        f"- News sentiment: {event.news_sentiment:+.2f} (-1=terrible, +1=great)\n"
        f"- Market uncertainty: {event.uncertainty:.0%}\n"
        f"- Crowd (10,000 retail agents): "
        f"{tier3_snapshot['bullish_pct']:.0%} bull / "
        f"{tier3_snapshot['bearish_pct']:.0%} bear / "
        f"{tier3_snapshot['neutral_pct']:.0%} neutral\n"
        f"- Herding index: {tier3_snapshot.get('herd_index', 0.5):.0%} "
        f"(how aligned the crowd is)\n"
        f"- Panic level: {tier3_snapshot.get('panic_pct', 0.0):.1%} of crowd in panic"
    )

    user_prompt = (
        f"Market event: {event.description}\n\n"
        f"{crowd_context}\n\n"
        f"Respond ONLY with a JSON object in this exact format — no preamble, "
        f"no markdown fences, no extra keys:\n"
        f'{{\n'
        f'  "stance": "bullish" | "bearish" | "neutral",\n'
        f'  "conviction": 0.0-1.0,\n'
        f'  "key_factor": "single most important driver of your view (max 15 words)",\n'
        f'  "watching": "the price level or event you are monitoring (max 12 words)",\n'
        f'  "change_mind": "what would flip your view (max 15 words)",\n'
        f'  "comment": "your authentic reaction in character (max 40 words)"\n'
        f'}}'
    )

    return persona.system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Single agent call (async)
# ---------------------------------------------------------------------------

async def _call_single_agent(
    client:         Any,        # openai.AsyncOpenAI
    persona:        Persona,
    event:          MarketEvent,
    tier3_snapshot: dict,
    semaphore:      asyncio.Semaphore,
    temperature:    float = 0.85,
    max_tokens:     int   = 280,
) -> Tier2AgentResponse:
    """Fire one async LLM call for a single Tier 2 agent."""

    system_prompt, user_prompt = _build_tier2_prompt(persona, event, tier3_snapshot)

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model       = "gpt-4o-mini",
                temperature = temperature,
                max_tokens  = max_tokens,
                messages    = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()

            # Parse JSON
            # Strip markdown fences if model disobeys
            clean = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)

            stance     = parsed.get("stance", "neutral").lower()
            conviction = float(parsed.get("conviction", 0.5))
            conviction = max(0.0, min(1.0, conviction))

            # Derive sentiment score
            if stance == "bullish":
                sentiment_score = conviction
            elif stance == "bearish":
                sentiment_score = -conviction
            else:
                sentiment_score = 0.0

            return Tier2AgentResponse(
                persona_name    = persona.name,
                stance          = stance,
                conviction      = conviction,
                sentiment_score = sentiment_score,
                key_factor      = str(parsed.get("key_factor", ""))[:80],
                watching        = str(parsed.get("watching", ""))[:60],
                change_mind     = str(parsed.get("change_mind", ""))[:80],
                raw_response    = raw,
                is_fallback     = False,
            )

        except Exception as e:
            # Fallback: agent is "confused" — returns neutral with low conviction
            return Tier2AgentResponse(
                persona_name    = persona.name,
                stance          = "neutral",
                conviction      = 0.30,
                sentiment_score = 0.0,
                key_factor      = f"parse_error: {str(e)[:50]}",
                watching        = "unclear",
                change_mind     = "clearer information",
                raw_response    = f"ERROR: {str(e)}",
                is_fallback     = True,
            )


# ---------------------------------------------------------------------------
# Tier 2 batch runner
# ---------------------------------------------------------------------------

class Tier2Simulation:
    """
    Runs N GPT-4o-mini agents concurrently against a market event.
    Each agent is one of the 5 Tier 2 persona types.
    """

    def __init__(
        self,
        n_agents:        int   = 200,
        batch_size:      int   = 20,     # concurrent calls per batch
        temperature:     float = 0.85,
        max_tokens:      int   = 280,
        verbose:         bool  = True,
    ):
        self.n_agents    = n_agents
        self.batch_size  = batch_size
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.verbose     = verbose

    def _build_agent_list(self, asset_class: str = "equity_index") -> list[Persona]:
        """
        Assign personas based on asset class.
        Uses get_tier2_personas_for() to select relevant personas only.
        """
        from simulation.personas import get_tier2_personas_for
        relevant_personas = get_tier2_personas_for(asset_class)
        if not relevant_personas:
            relevant_personas = TIER2_PERSONAS[:5]

        # Distribute evenly across relevant personas
        agents = []
        per_persona = self.n_agents // len(relevant_personas)
        for persona in relevant_personas:
            agents.extend([persona] * per_persona)

        # Pad to exactly n_agents
        while len(agents) < self.n_agents:
            agents.append(random.choice(relevant_personas))
        agents = agents[:self.n_agents]
        random.shuffle(agents)
        return agents

    async def run_async(
        self,
        event:          MarketEvent,
        tier3_result:   Tier3Result,
        api_key:        str,
    ) -> Tier2Result:
        """Run all agents asynchronously. Returns Tier2Result."""

        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )

        client    = AsyncOpenAI(api_key=api_key)
        semaphore = asyncio.Semaphore(self.batch_size)

        # Build tier3 snapshot dict for prompt injection
        t3_snap = tier3_result.to_tier_snapshot_dict()
        t3_snap["herd_index"] = tier3_result.final_snapshot.herd_index
        t3_snap["panic_pct"]  = tier3_result.final_snapshot.panic_agents_pct

        # Asset-class-aware agent selection
        asset_class = t3_snap.get("asset_class", "equity_index")
        agents = self._build_agent_list(asset_class)

        t0 = time.perf_counter()

        if self.verbose:
            print(f"  [Tier2] Firing {self.n_agents} async GPT-4o-mini calls "
                  f"(batch_size={self.batch_size}, asset_class={asset_class})...")

        # Fire all tasks concurrently (semaphore limits parallelism)
        tasks = [
            _call_single_agent(
                client, agent, event, t3_snap,
                semaphore, self.temperature, self.max_tokens
            )
            for agent in agents
        ]
        responses: list[Tier2AgentResponse] = await asyncio.gather(*tasks)

        elapsed = time.perf_counter() - t0

        # Aggregate
        n_bull     = sum(1 for r in responses if r.stance == "bullish")
        n_bear     = sum(1 for r in responses if r.stance == "bearish")
        n_neutral  = sum(1 for r in responses if r.stance == "neutral")
        n_fallback = sum(1 for r in responses if r.is_fallback)
        n_success  = len(responses) - n_fallback

        bull_pct    = n_bull    / len(responses)
        bear_pct    = n_bear    / len(responses)
        neutral_pct = n_neutral / len(responses)

        sentiments  = [r.sentiment_score for r in responses]
        convictions = [r.conviction      for r in responses]
        avg_sentiment  = sum(sentiments)  / len(sentiments)
        avg_conviction = sum(convictions) / len(convictions)

        # Extract notable behaviors: most common key_factors
        factors = [r.key_factor for r in responses if not r.is_fallback]
        # Simple frequency analysis on first 4 words of each factor
        factor_heads: dict[str, int] = {}
        for f in factors:
            head = " ".join(f.split()[:4]).lower()
            factor_heads[head] = factor_heads.get(head, 0) + 1
        top_factors = sorted(factor_heads.items(), key=lambda x: -x[1])[:3]
        notable = [f"{head} ({count} agents)" for head, count in top_factors]
        if n_fallback > 0:
            notable.append(f"{n_fallback} agents failed/fallback")

        result = Tier2Result(
            responses          = responses,
            n_agents           = self.n_agents,
            n_successful       = n_success,
            n_fallback         = n_fallback,
            elapsed_sec        = elapsed,
            bullish_pct        = bull_pct,
            bearish_pct        = bear_pct,
            neutral_pct        = neutral_pct,
            avg_conviction     = avg_conviction,
            avg_sentiment      = avg_sentiment,
            notable_behaviors  = notable,
        )

        if self.verbose:
            self._print_summary(result)

        return result

    def run(
        self,
        event:        MarketEvent,
        tier3_result: Tier3Result,
        api_key:      str,
    ) -> Tier2Result:
        """Sync wrapper — runs the async batch in the current event loop."""
        return asyncio.run(self.run_async(event, tier3_result, api_key))

    def _print_summary(self, r: Tier2Result) -> None:
        bull_bar = "▲" * int(r.bullish_pct  * 20)
        bear_bar = "▼" * int(r.bearish_pct  * 20)
        neut_bar = "─" * (20 - len(bull_bar) - len(bear_bar))
        bar = f"[{bull_bar}{neut_bar}{bear_bar}]"

        print(f"\n  {'─'*56}")
        print(f"  Tier 2 Complete — {r.n_agents} informed crowd agents")
        print(f"  {'─'*56}")
        print(f"  Runtime:       {r.elapsed_sec:.1f}s  "
              f"({r.elapsed_sec/r.n_agents*1000:.0f}ms/agent effective)")
        print(f"  Stance:        {bar}  "
              f"{r.bullish_pct:.0%} bull / {r.bearish_pct:.0%} bear / "
              f"{r.neutral_pct:.0%} neutral")
        print(f"  Avg sentiment: {r.avg_sentiment:+.4f}")
        print(f"  Avg conviction:{r.avg_conviction:.4f}")
        print(f"  Successful:    {r.n_successful}/{r.n_agents}")
        if r.notable_behaviors:
            print(f"  Top factors:")
            for b in r.notable_behaviors[:3]:
                print(f"    · {b}")
        print(f"  {'─'*56}\n")


# ---------------------------------------------------------------------------
# Standalone demo with mock client (no API key needed for testing)
# ---------------------------------------------------------------------------

class _MockAsyncOpenAI:
    """
    Drop-in mock for AsyncOpenAI. Returns plausible JSON responses
    without making any actual API calls. Used for local testing.
    """

    def __init__(self, **kwargs): pass

    @property
    def chat(self): return self

    @property
    def completions(self): return self

    async def create(self, model, messages, temperature, max_tokens, **kwargs):
        # Simulate variable latency
        await asyncio.sleep(random.uniform(0.01, 0.08))

        # Extract user message to infer event sentiment
        user_msg = messages[-1]["content"] if messages else ""
        is_bearish_event = "hot inflation" in user_msg.lower() or \
                           "breakdown" in user_msg.lower() or \
                           "selling" in user_msg.lower()

        # Bias stance based on event
        stances = (
            ["bearish"] * 6 + ["neutral"] * 3 + ["bullish"] * 1
            if is_bearish_event else
            ["bullish"] * 6 + ["neutral"] * 3 + ["bearish"] * 1
        )
        stance     = random.choice(stances)
        conviction = round(random.uniform(0.45, 0.88), 2)

        factors_bearish = [
            "Fed staying higher for longer kills EM flows",
            "FII selling accelerating on rate shock",
            "Technical breakdown confirms bearish structure",
            "Rising dollar headwind for Indian markets",
        ]
        factors_bullish = [
            "RBI dovish pivot supports domestic liquidity",
            "DII buying absorbing FII outflows effectively",
            "Valuation attractive at current levels",
            "Domestic growth story remains intact",
        ]
        factors = factors_bearish if is_bearish_event else factors_bullish

        payload = {
            "stance":      stance,
            "conviction":  conviction,
            "key_factor":  random.choice(factors),
            "watching":    "Nifty 24000 support" if is_bearish_event else "Nifty 24500 breakout",
            "change_mind": "Fed pivot signal" if is_bearish_event else "Inflation spike above 4%",
            "comment":     f"Market feels {'dangerous' if is_bearish_event else 'constructive'} here.",
        }

        class _Msg:
            content = json.dumps(payload)
        class _Choice:
            message = _Msg()
        class _Resp:
            choices = [_Choice()]
        return _Resp()


async def _demo_with_mock():
    """Run Tier 2 with mock client — no API key required."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    # Monkey-patch AsyncOpenAI
    import openai as _openai_module
    _openai_module.AsyncOpenAI = _MockAsyncOpenAI

    from simulation.tiers.tier3_mesa import Tier3Simulation

    event = MarketEvent(
        description    = "US CPI prints 3.4% vs 2.9% expected — hot inflation surprise",
        price_shock    = -0.018,
        news_sentiment = -0.65,
        uncertainty    = 0.70,
        is_systemic    = True,
        sector_impact  = "all",
    )

    # Run Tier 3 first
    print("\n[Running Tier 3 first to provide crowd context...]")
    t3 = Tier3Simulation(n_agents=10_000, n_steps=50, verbose=True)
    t3._initialise_population()
    t3_result = t3.run(event)

    # Run Tier 2 with mock
    print("\n[Running Tier 2 with mock LLM (no API key needed)...]")
    sim = Tier2Simulation(n_agents=200, batch_size=20, verbose=True)
    result = await sim.run_async(event, t3_result, api_key="mock")

    print(f"\nSample responses (first 5):")
    for r in result.responses[:5]:
        print(f"  [{r.persona_name:<30}] {r.stance:<8} "
              f"conviction={r.conviction:.2f}  "
              f"→ '{r.key_factor[:50]}'")


if __name__ == "__main__":
    try:
        import openai  # noqa
    except ImportError:
        print("openai not installed — using mock client")
        # Provide a minimal mock module
        import types
        openai_mock = types.ModuleType("openai")
        openai_mock.AsyncOpenAI = _MockAsyncOpenAI
        sys.modules["openai"] = openai_mock

    asyncio.run(_demo_with_mock())
