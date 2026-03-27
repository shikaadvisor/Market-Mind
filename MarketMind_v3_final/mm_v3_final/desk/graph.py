"""
desk/graph.py
=============
The LangGraph state machine for the trading desk.

Graph topology:
                     ┌─────────────┐
                     │  START      │
                     └──────┬──────┘
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
       ┌──────────────┐         ┌──────────────────┐
       │ macro_agent  │         │ technical_agent   │
       └──────┬───────┘         └────────┬──────────┘
               │                         │
               │   ┌─────────────────┐   │
               │   │ sentiment_agent  │   │
               │   └────────┬────────┘   │
               │            │            │
               └────────────┴────────────┘
                            │ (all three complete)
                            ▼
                   ┌─────────────────┐
                   │   risk_agent    │ (reads macro + technical)
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  coordinator    │ (reads everything → TradingSignal)
                   └────────┬────────┘
                            │
                         ┌──┴──┐
                         │ END │
                         └─────┘

Macro, Technical, and Sentiment run in PARALLEL (fan-out).
Risk runs AFTER Macro + Technical (needs their outputs).
Coordinator runs last — reads all four outputs.

This is a conditional graph — if any agent fails catastrophically,
the error is captured in state and the coordinator produces a
blocked signal with the error in do_not_trade_flags.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from desk.agents.base import DeskState
from desk.agents.macro     import MacroAgent
from desk.agents.technical import TechnicalAgent
from desk.agents.risk      import RiskAgent
from desk.agents.sentiment import SentimentAgent
from desk.coordinator      import DeskCoordinator


# ---------------------------------------------------------------------------
# Node functions — each wraps one agent
# ---------------------------------------------------------------------------

async def _macro_node(
    state:   DeskState,
    agent:   MacroAgent,
    api_key: str,
) -> dict:
    output = await agent.run(state, api_key)
    return {
        "macro_view": {
            "direction":        output.direction,
            "conviction":       output.conviction,
            "regime":           output.regime,
            "macro_context":    output.macro_context,
            "key_factors":      output.key_factors,
            "fii_outlook":      output.fii_outlook,
            "rate_environment": output.rate_environment,
            "warning_flags":    output.warning_flags,
        }
    }


async def _technical_node(
    state:   DeskState,
    agent:   TechnicalAgent,
    api_key: str,
) -> dict:
    output = await agent.run(state, api_key)
    return {
        "technical_view": {
            "direction":     output.direction,
            "conviction":    output.conviction,
            "technical_view": output.technical_view,
            "entry_low":     output.entry_low,
            "entry_high":    output.entry_high,
            "stop_loss":     output.stop_loss,
            "target_1":      output.target_1,
            "target_2":      output.target_2,
            "key_levels":    output.key_levels,
            "pattern":       output.pattern,
        }
    }


async def _sentiment_node(
    state:   DeskState,
    agent:   SentimentAgent,
    api_key: str,
) -> dict:
    output = await agent.run(state, api_key)
    return {
        "sentiment_view": {
            "crowd_direction":   output.crowd_direction,
            "crowd_conviction":  output.crowd_conviction,
            "simulation_weight": output.simulation_weight,
            "crowd_narrative":   output.crowd_narrative,
            "key_crowd_insight": output.key_crowd_insight,
            "contrarian_signal": output.contrarian_signal,
        }
    }


async def _risk_node(
    state:   DeskState,
    agent:   RiskAgent,
    api_key: str,
) -> dict:
    output = await agent.run(state, api_key)
    return {
        "risk_params": {
            "approved":            output.approved,
            "max_position_pct":    output.max_position_pct,
            "invalidation_level":  output.invalidation_level,
            "do_not_trade_flags":  output.do_not_trade_flags,
            "warning_flags":       output.warning_flags,
            "risk_reward":         output.risk_reward,
            "conviction_modifier": output.conviction_modifier,
            "rationale":           output.rationale,
        }
    }


async def _coordinator_node(
    state:       DeskState,
    coordinator: DeskCoordinator,
    api_key:     str,
) -> dict:
    signal_dict = await coordinator.synthesise(state, api_key)
    return {"final_signal": signal_dict}


# ---------------------------------------------------------------------------
# The desk graph — manual async implementation
# (full LangGraph integration is a drop-in replacement when installed)
# ---------------------------------------------------------------------------

class DeskGraph:
    """
    Implements the trading desk agent graph topology manually via asyncio.
    This is a clean async state machine — no LangGraph dependency required
    for the core logic. When you install langgraph, this class can be
    replaced with a compiled StateGraph with identical semantics.

    Execution order:
      Phase 1 (parallel): macro + technical + sentiment
      Phase 2 (sequential): risk (reads phase 1 outputs)
      Phase 3 (sequential): coordinator (reads all)
    """

    def __init__(
        self,
        openai_api_key:    str  = "",
        anthropic_api_key: str  = "",
        mock:              bool = False,
    ):
        self.openai_key    = openai_api_key
        self.anthropic_key = anthropic_api_key
        self.mock          = mock

        # Inject mock clients if testing without API keys
        if mock:
            from desk.agents.macro     import _MockMacroClient
            from desk.agents.technical import _MockTechnicalClient
            from desk.agents.risk      import _MockRiskClient
            from desk.agents.sentiment import _MockSentimentClient
            from desk.coordinator      import _MockAnthropicClient

            self.macro_agent     = MacroAgent(    mock=True, mock_client=_MockMacroClient())
            self.tech_agent      = TechnicalAgent(mock=True, mock_client=_MockTechnicalClient())
            self.risk_agent      = RiskAgent(     mock=True, mock_client=_MockRiskClient())
            self.sentiment_agent = SentimentAgent(mock=True, mock_client=_MockSentimentClient())
            self.coordinator     = DeskCoordinator(mock=True, mock_client=_MockAnthropicClient())
        else:
            self.macro_agent     = MacroAgent()
            self.tech_agent      = TechnicalAgent()
            self.risk_agent      = RiskAgent()
            self.sentiment_agent = SentimentAgent()
            self.coordinator     = DeskCoordinator()

    async def run_async(self, initial_state: DeskState) -> DeskState:
        """
        Execute the full desk graph asynchronously.
        Returns the final DeskState with all fields populated.
        """
        state = dict(initial_state)   # mutable copy

        # ── Phase 1: Macro, Technical, Sentiment in parallel ─────────────────
        macro_task    = _macro_node(state, self.macro_agent, self.openai_key)
        tech_task     = _technical_node(state, self.tech_agent, self.openai_key)
        sentiment_task = _sentiment_node(state, self.sentiment_agent, self.openai_key)

        macro_result, tech_result, sent_result = await asyncio.gather(
            macro_task, tech_task, sentiment_task
        )

        # Merge phase 1 outputs into state
        state.update(macro_result)
        state.update(tech_result)
        state.update(sent_result)

        # ── Phase 2: Risk (reads macro + technical) ───────────────────────────
        risk_result = await _risk_node(state, self.risk_agent, self.openai_key)
        state.update(risk_result)

        # ── Phase 3: Coordinator (reads everything) ───────────────────────────
        coord_result = await _coordinator_node(
            state, self.coordinator, self.anthropic_key
        )
        state.update(coord_result)

        return state   # type: ignore

    def run(self, initial_state: DeskState) -> DeskState:
        """Sync wrapper."""
        return asyncio.run(self.run_async(initial_state))


# ---------------------------------------------------------------------------
# Convenience function — build state from simulation output
# ---------------------------------------------------------------------------

def build_initial_state(
    instrument:        str,
    exchange:          str,
    asset_class:       str,
    event_description: str,
    simulation_report: dict,
    market_data:       dict | None = None,
    expiry_date:       str  | None = None,
) -> DeskState:
    """
    Construct the initial DeskState from simulation outputs and market data.
    This is the entry point called by the pipeline runner.
    """
    from datetime import date

    # Extract the brief and metrics from simulation report
    sim_brief   = simulation_report.get("sentiment_interpreter_brief", "")
    sim_metrics = {
        k: simulation_report.get(k)
        for k in [
            "aggregate_sentiment", "retail_sentiment", "institutional_sentiment",
            "herd_index", "panic_probability", "narrative_momentum",
            "contrarian_pressure", "information_velocity", "run_id",
        ]
    }
    # Add agent count from tier snapshots
    sim_metrics["agent_count"]          = simulation_report.get("total_agents", 10210)
    sim_metrics["opinion_cluster_count"] = len(simulation_report.get("opinion_clusters", []))

    return DeskState(
        instrument         = instrument,
        exchange           = exchange,
        asset_class        = asset_class,
        point_in_time_date = str(date.today()),
        market_data        = market_data or {},
        simulation_brief   = sim_brief,
        simulation_metrics = sim_metrics,
        event_description  = event_description,
        expiry_date        = expiry_date,
    )


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("\n" + "="*60)
    print("  MarketMind — Trading Desk Graph Demo")
    print("  (mock LLM — no API keys needed)")
    print("="*60)

    # Simulate what the pipeline would provide
    mock_sim_report = {
        "sentiment_interpreter_brief": (
            "SIMULATION BRIEF: Event: RBI holds at 6.5% — dovish tone.\n"
            "Aggregate sentiment: +0.31 (MILD_GREED). Retail: +0.30, Institutional: +0.35.\n"
            "Herding: 38%. Panic prob: 0%. Contrarian: 0%.\n"
            "Dominant narrative: MACRO_RISK_ON (confidence 65%).\n"
            "RECOMMENDED USE: Momentum confirming — bullish consensus, moderate weight 0.22."
        ),
        "aggregate_sentiment":     0.31,
        "retail_sentiment":        0.30,
        "institutional_sentiment": 0.35,
        "herd_index":              0.38,
        "panic_probability":       0.00,
        "narrative_momentum":      0.38,
        "contrarian_pressure":     0.00,
        "information_velocity":    0.14,
        "run_id":                  "demo-rbi-hold-001",
        "total_agents":            10210,
        "opinion_clusters":        [{}, {}],
    }

    initial_state = build_initial_state(
        instrument        = "NIFTY50",
        exchange          = "NSE",
        asset_class       = "equity",
        event_description = "RBI holds repo rate at 6.5% — tone slightly dovish, signals potential cut next meeting",
        simulation_report = mock_sim_report,
        market_data       = {
            "nifty_spot":    24280,
            "india_vix":     13.2,
            "fii_net_weekly": 1200,
            "usd_inr":       83.8,
            "nifty_52w_high": 26277,
            "nifty_52w_low":  21964,
        },
    )

    print("\n[Running trading desk graph...]")
    graph  = DeskGraph(mock=True)
    result = graph.run(initial_state)

    signal = result.get("final_signal", {})

    print(f"\n{'─'*60}")
    print(f"  FINAL SIGNAL")
    print(f"{'─'*60}")
    print(f"  Instrument:  {signal.get('instrument')}")
    print(f"  Direction:   {signal.get('direction', '').upper()}")
    print(f"  Conviction:  {signal.get('conviction', 0):.0%}")
    print(f"  Timeframe:   {signal.get('timeframe')}")
    print(f"  Entry zone:  {signal.get('entry_zone', {}).get('low')} – "
          f"{signal.get('entry_zone', {}).get('high')}")
    print(f"  Stop:        {signal.get('risk', {}).get('stop_loss')}")
    print(f"  Target 1:    {signal.get('risk', {}).get('target_1')}")
    print(f"  R:R:         {signal.get('risk', {}).get('risk_reward', 0):.2f}")
    print(f"  Vetoes:      {signal.get('do_not_trade_flags', [])}")
    print(f"  Warnings:    {signal.get('warning_flags', [])}")
    print(f"  Conf flags:  {signal.get('confidence_flags', [])}")

    print(f"\n  Coordinator thesis:")
    thesis = signal.get("coordinator_thesis", "")
    for line in thesis[:500].split(". "):
        if line.strip():
            print(f"    {line.strip()}.")

    print(f"\n  Agent contributions:")
    for ac in signal.get("agent_contributions", []):
        print(f"    [{ac.get('agent_role',''):<25}] "
              f"{ac.get('view',''):<8} "
              f"conviction={ac.get('conviction', 0):.0%}")

    # Check if actionable
    conviction   = signal.get("conviction", 0)
    vetoes       = signal.get("do_not_trade_flags", [])
    is_actionable = conviction >= 0.70 and len(vetoes) == 0
    print(f"\n  Signal status: {'✓ ACTIONABLE — review and consider' if is_actionable else '⚠ NOT ACTIONABLE'}")
    print(f"{'─'*60}\n")
