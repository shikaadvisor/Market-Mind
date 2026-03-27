"""
simulation/runner.py
====================
The simulation pipeline orchestrator. Replaces all previous stubs.

Entry point for the full Digital World simulation.
Given a MarketEvent it:
  1. Tier 3 — Mesa ABM (10,000+ agents, ~$0, ~0.1s)
  2. Tier 2 — async GPT-4o-mini (200 agents, ~$0.05, ~15s)
  3. Tier 1 — concurrent GPT-4o deep thinkers (10 agents, ~$0.12, ~10s)
  4. Coordinator aggregates all tiers → SimulationReport

Tiers 2 and 3 run concurrently via asyncio (overlap saves ~10s per run).
Total cost: ~$0.20–0.50 | Total time: ~20–30s

Usage:
    runner = SimulationRunner(openai_api_key="sk-...")
    report_dict = runner.run(event, instrument="NIFTY50")

    # Mock mode (no API key — for local testing):
    runner = SimulationRunner(mock=True)
    report_dict = runner.run(event)

Run standalone demo:
    python simulation/runner.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from simulation.tiers.tier3_mesa import Tier3Simulation, MarketEvent
from simulation.tiers.tier2_async import Tier2Simulation
from simulation.tiers.tier1_deep import Tier1Simulation
from simulation.coordinator import SimulationCoordinator


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

class SimulationRunner:
    """
    Orchestrates the full three-tier Digital World simulation.
    Tier 3 is stateful (population persists between runs for speed).
    Tiers 2 and 3 are stateless LLM calls — fresh each time.
    """

    def __init__(
        self,
        n_agents:        int   = 10_000,
        n_steps:         int   = 50,
        n_tier2_agents:  int   = 200,
        openai_api_key:  str   = "",
        mock:            bool  = False,
        verbose:         bool  = True,
    ):
        self.n_agents       = n_agents
        self.n_steps        = n_steps
        self.openai_api_key = openai_api_key
        self.mock           = mock
        self.verbose        = verbose

        # Tier 3: initialise population once, reuse across events
        self._tier3_sim = Tier3Simulation(
            n_agents = n_agents,
            n_steps  = n_steps,
            verbose  = verbose,
        )
        self._tier3_sim._initialise_population()

        # Tier 2 and 1 runners
        self._tier2_sim = Tier2Simulation(
            n_agents  = n_tier2_agents,
            batch_size = 20,
            verbose   = verbose,
        )
        self._tier1_sim = Tier1Simulation(verbose=verbose)

        # Coordinator (deterministic, no LLM)
        self._coordinator = SimulationCoordinator()

        # Install mock LLM clients if requested
        if mock:
            self._install_mocks()

    def _install_mocks(self) -> None:
        """Monkey-patch OpenAI with mock clients for local testing."""
        import types
        from simulation.tiers.tier2_async import _MockAsyncOpenAI as Mock2
        from simulation.tiers.tier1_deep  import _MockAsyncOpenAI as Mock1

        # Both mocks share the same interface; we differentiate by file
        oai_mock = types.ModuleType("openai")
        oai_mock.AsyncOpenAI = Mock2      # Tier2 uses this
        sys.modules["openai"] = oai_mock

        # Tier1 needs its own mock registered on the module
        # We handle this by patching after import
        import simulation.tiers.tier2_async as t2_mod
        import simulation.tiers.tier1_deep  as t1_mod
        t2_mod.AsyncOpenAI = Mock2        # type: ignore
        t1_mod.AsyncOpenAI = Mock1        # type: ignore

        self.openai_api_key = "mock"

    def run(self, event: MarketEvent, instrument: str = "NIFTY50",
            asset_class: str = "equity_index") -> dict:
        """
        Run the full simulation synchronously.
        Returns a dict ready for SimulationReport(**result).
        """
        return asyncio.run(self._run_async(event, instrument, asset_class))

    async def _run_async(self, event: MarketEvent, instrument: str,
                         asset_class: str = "equity_index") -> dict:
        t_total = time.perf_counter()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  MarketMind — Digital World")
            print(f"  Event: {event.description[:60]}...")
            print(f"{'='*60}")

        # ── Step 1: Tier 3 (sync, fast) ─────────────────────────────────────
        if self.verbose:
            print("\n[1/3] Tier 3 — statistical crowd (ABM)...")

        # Re-initialise population for this asset class (new composition each run)
        self._tier3_sim._initialise_population(asset_class=asset_class)
        tier3_result = self._tier3_sim.run(event)
        tier3_result._asset_class = asset_class   # propagate to Tier 1 persona selection

        # ── Steps 2+3: Tier 2 and Tier 1 run concurrently ───────────────────
        if self.verbose:
            print("\n[2+3/3] Tier 2 + Tier 1 — LLM agents (concurrent)...")

        tier2_task = asyncio.create_task(
            self._tier2_sim.run_async(event, tier3_result, self.openai_api_key)
        )
        tier1_task = asyncio.create_task(
            self._tier1_sim.run_async(event, tier3_result, None, self.openai_api_key)
        )

        tier2_result, tier1_result = await asyncio.gather(tier2_task, tier1_task)

        # ── Coordinator: aggregate all tiers ─────────────────────────────────
        report_dict = self._coordinator.aggregate(
            event      = event,
            tier3      = tier3_result,
            tier2      = tier2_result,
            tier1      = tier1_result,
            instrument = instrument,
        )

        elapsed = time.perf_counter() - t_total

        if self.verbose:
            agg = report_dict["aggregate_sentiment"]
            narrative = report_dict["dominant_narrative"]
            cascade   = report_dict["cascade_risk"]
            total     = report_dict["total_agents"]
            print(f"\n{'='*60}")
            print(f"  Simulation complete in {elapsed:.1f}s")
            print(f"  Agents: {total:,}  |  Sentiment: {agg:+.4f}")
            print(f"  Narrative: {narrative}  |  Cascade: {cascade.upper()}")
            print(f"{'='*60}\n")

        return report_dict


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  MarketMind — Full 3-Tier Digital World Demo")
    print("  (mock LLM — no API key needed)")
    print("="*60)

    scenarios = [
        MarketEvent(
            description    = "US CPI prints 3.4% vs 2.9% — hot inflation, "
                             "Fed cuts pushed back to Q4 2026",
            price_shock    = -0.018,
            news_sentiment = -0.65,
            uncertainty    = 0.70,
            is_systemic    = True,
            sector_impact  = "all",
        ),
        MarketEvent(
            description    = "RBI holds at 6.5%, tone dovish — signals cut next meeting",
            price_shock    = +0.008,
            news_sentiment = +0.45,
            uncertainty    = 0.35,
            is_systemic    = True,
            sector_impact  = "banking",
        ),
    ]

    runner = SimulationRunner(
        n_agents  = 10_000,
        n_steps   = 50,
        mock      = True,
        verbose   = True,
    )

    for i, event in enumerate(scenarios, 1):
        print(f"\n{'#'*60}")
        print(f"  SCENARIO {i}/{len(scenarios)}")
        print(f"{'#'*60}")

        report_dict = runner.run(event, instrument="NIFTY50")

        # Print the interpreter brief — this is what the desk will see
        print("\n── SENTIMENT INTERPRETER BRIEF ──")
        print(report_dict["sentiment_interpreter_brief"])
        print("\n── CONTRARIAN THESIS ──")
        print(report_dict["contrarian_thesis"])
        print()
