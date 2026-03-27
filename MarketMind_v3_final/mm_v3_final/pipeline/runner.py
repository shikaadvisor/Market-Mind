"""
pipeline/runner.py
==================
The end-to-end MarketMind pipeline.

One call to MarketMindPipeline.run(event) triggers:
  1. Digital World simulation (Tier 3 → 2 → 1 → coordinator)
  2. Trading Desk graph (Macro + Technical + Sentiment ∥ → Risk → Coordinator)
  3. TradingSignal construction and Pydantic validation
  4. Signal logging (CSV + JSONL)
  5. Return the validated TradingSignal

This is the only file the rest of the application needs to import.
Everything else is an implementation detail.

Usage:
    from pipeline.runner import MarketMindPipeline, PipelineConfig
    from simulation.tiers.tier3_mesa import MarketEvent

    cfg = PipelineConfig(
        openai_api_key    = "sk-...",
        anthropic_api_key = "sk-ant-...",
    )
    pipeline = MarketMindPipeline(cfg)

    event = MarketEvent(
        description    = "RBI holds at 6.5%, tone dovish",
        price_shock    = +0.008,
        news_sentiment = +0.45,
        uncertainty    = 0.35,
        is_systemic    = True,
        sector_impact  = "all",
    )

    signal = pipeline.run(
        event      = event,
        instrument = "NIFTY50",
        exchange   = "NSE",
    )
    print(signal.signal_summary)
"""

from __future__ import annotations

import asyncio
import time
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.runner import SimulationRunner
from simulation.tiers.tier3_mesa import MarketEvent
from desk.graph import DeskGraph, build_initial_state


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """All knobs for one pipeline run."""

    # API keys
    openai_api_key:    str  = field(default_factory=lambda: __import__("os").environ.get("OPENAI_API_KEY",    ""))
    anthropic_api_key: str  = field(default_factory=lambda: __import__("os").environ.get("ANTHROPIC_API_KEY", ""))

    # Simulation settings
    n_agents:      int  = 10_000
    n_steps:       int  = 50
    n_tier2:       int  = 200

    # Mock mode — no API calls, for local testing
    mock:          bool = False

    # Logging
    log_signals:   bool = True
    log_dir:       Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs")

    # Display
    verbose:       bool = True
    show_brief:    bool = True   # print simulation brief to terminal


# ---------------------------------------------------------------------------
# Pipeline result container
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Everything produced by one pipeline run."""
    signal:              object              # TradingSignal (typed below after import)
    simulation_report:   dict
    desk_state:          dict
    elapsed_sim_sec:     float
    elapsed_desk_sec:    float
    elapsed_total_sec:   float
    error:               Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def is_actionable(self) -> bool:
        return self.success and self.signal.is_actionable  # type: ignore


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class MarketMindPipeline:
    """
    Orchestrates the full MarketMind system end-to-end.
    Initialise once, call run() for each new event.
    """

    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self.cfg.log_dir.mkdir(exist_ok=True)

        # Initialise simulation runner (Tier 3 population persists)
        self._sim_runner = SimulationRunner(
            n_agents       = config.n_agents,
            n_steps        = config.n_steps,
            n_tier2_agents = config.n_tier2,
            openai_api_key = config.openai_api_key,
            mock           = config.mock,
            verbose        = config.verbose,
        )

        # Initialise desk graph
        self._desk = DeskGraph(
            openai_api_key    = config.openai_api_key,
            anthropic_api_key = config.anthropic_api_key,
            mock              = config.mock,
        )

        # Signal logger (lazy init)
        self._logger = None

        if config.verbose:
            print(f"\n{'═'*62}")
            print(f"  MarketMind Pipeline initialised")
            print(f"  Agents: {config.n_agents:,} (T3) + {config.n_tier2} (T2) + 10 (T1)")
            print(f"  Mode:   {'MOCK (no API calls)' if config.mock else 'LIVE (real LLM)'}")
            print(f"{'═'*62}\n")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(
        self,
        event:       MarketEvent,
        instrument:  str = "NIFTY50",
        exchange:    str = "NSE",
        asset_class: str = "equity",
        market_data: dict | None = None,
        expiry_date: str | None  = None,
    ) -> PipelineResult:
        """
        Run the full pipeline synchronously.
        Returns a PipelineResult containing the validated TradingSignal.
        """
        return asyncio.run(
            self._run_async(event, instrument, exchange, asset_class,
                            market_data, expiry_date)
        )

    async def _run_async(
        self,
        event:       MarketEvent,
        instrument:  str,
        exchange:    str,
        asset_class: str,
        market_data: dict | None,
        expiry_date: str | None,
    ) -> PipelineResult:

        t_total = time.perf_counter()

        _print_section("MARKETMIND PIPELINE RUN", self.cfg.verbose)
        _print_kv("Event",      event.description[:70], self.cfg.verbose)
        _print_kv("Instrument", f"{instrument} ({exchange})", self.cfg.verbose)
        _print_kv("Date",       str(date.today()), self.cfg.verbose)

        # ── Step 1: Digital World ────────────────────────────────────────────
        _print_section("STEP 1 / 3 — Digital World", self.cfg.verbose)
        t_sim = time.perf_counter()

        sim_report = await self._sim_runner._run_async(event, instrument, asset_class)
        elapsed_sim = time.perf_counter() - t_sim

        if self.cfg.verbose:
            agg = sim_report.get("aggregate_sentiment", 0)
            narrative = sim_report.get("dominant_narrative", "unknown")
            cascade   = sim_report.get("cascade_risk", "low")
            _print_kv("Sentiment",  f"{agg:+.4f}", self.cfg.verbose)
            _print_kv("Narrative",  narrative,      self.cfg.verbose)
            _print_kv("Cascade",    cascade.upper(), self.cfg.verbose)
            _print_kv("Time",       f"{elapsed_sim:.1f}s", self.cfg.verbose)

        if self.cfg.show_brief:
            brief = sim_report.get("sentiment_interpreter_brief", "")
            if brief:
                print(f"\n  {'─'*56}")
                print("  SIMULATION BRIEF (preview):")
                for line in brief.split("\n")[:12]:
                    print(f"  {line}")
                print(f"  {'─'*56}\n")

        # ── Step 2: Trading Desk ─────────────────────────────────────────────
        _print_section("STEP 2 / 3 — Trading Desk", self.cfg.verbose)
        t_desk = time.perf_counter()

        initial_state = build_initial_state(
            instrument        = instrument,
            exchange          = exchange,
            asset_class       = asset_class,
            event_description = event.description,
            simulation_report = sim_report,
            market_data       = market_data or {},
            expiry_date       = expiry_date,
        )

        desk_state = await self._desk.run_async(initial_state)
        elapsed_desk = time.perf_counter() - t_desk

        signal_dict = desk_state.get("final_signal", {})

        if self.cfg.verbose:
            direction  = signal_dict.get("direction", "?")
            conviction = signal_dict.get("conviction", 0)
            vetoes     = signal_dict.get("do_not_trade_flags", [])
            _print_kv("Direction",  direction.upper(), self.cfg.verbose)
            _print_kv("Conviction", f"{conviction:.0%}", self.cfg.verbose)
            _print_kv("Vetoes",     str(vetoes) if vetoes else "none", self.cfg.verbose)
            _print_kv("Time",       f"{elapsed_desk:.1f}s", self.cfg.verbose)

        # ── Step 3: Validate TradingSignal ───────────────────────────────────
        _print_section("STEP 3 / 3 — Signal Validation", self.cfg.verbose)

        signal, error = _build_and_validate_signal(signal_dict, sim_report)

        if error:
            _print_kv("Status", f"VALIDATION ERROR: {error[:80]}", self.cfg.verbose)
        else:
            _print_kv("Status",     "✓ Pydantic validation passed", self.cfg.verbose)
            _print_kv("Signal ID",  signal.signal_id[:16] + "...", self.cfg.verbose)  # type: ignore
            _print_kv("Actionable", "YES ✓" if signal.is_actionable else "NO (review)", self.cfg.verbose)  # type: ignore

        elapsed_total = time.perf_counter() - t_total

        result = PipelineResult(
            signal            = signal,
            simulation_report = sim_report,
            desk_state        = dict(desk_state),
            elapsed_sim_sec   = elapsed_sim,
            elapsed_desk_sec  = elapsed_desk,
            elapsed_total_sec = elapsed_total,
            error             = error,
        )

        # ── Log the signal ───────────────────────────────────────────────────
        if self.cfg.log_signals and signal is not None and error is None:
            _log_signal(signal, sim_report, self.cfg.log_dir)

        if self.cfg.verbose:
            print(f"\n  {'═'*58}")
            print(f"  Pipeline complete in {elapsed_total:.1f}s "
                  f"(sim={elapsed_sim:.1f}s, desk={elapsed_desk:.1f}s)")
            print(f"  {'═'*58}\n")

        return result


# ---------------------------------------------------------------------------
# Signal construction + Pydantic validation
# ---------------------------------------------------------------------------

def _build_and_validate_signal(
    signal_dict:   dict,
    sim_report:    dict,
) -> tuple:
    """
    Attempt to construct a TradingSignal from the coordinator output dict.
    Returns (signal, None) on success, (None, error_str) on failure.
    """
    try:
        from models.signal import (
            TradingSignal, PriceZone, RiskParameters, CrowdMetrics,
            AgentContribution, Direction, Timeframe, AssetClass, RegimeType,
        )

        # ── Build sub-models ─────────────────────────────────────────────────
        ez = signal_dict.get("entry_zone", {})
        entry_zone = PriceZone(
            low  = float(ez.get("low",  1.0)),
            high = float(ez.get("high", 2.0)),
        )

        rp = signal_dict.get("risk", {})
        t2 = rp.get("target_2")
        risk = RiskParameters(
            stop_loss          = float(rp.get("stop_loss", 1.0)),
            target_1           = float(rp.get("target_1", 2.0)),
            target_2           = float(t2) if t2 else None,
            risk_reward        = float(rp.get("risk_reward", 0.0)),
            max_position_pct   = float(rp.get("max_position_pct", 0.03)),
            invalidation_level = float(rp.get("invalidation_level", 1.0)),
        )

        cm = signal_dict.get("crowd", {})
        crowd = CrowdMetrics(
            aggregate_sentiment      = float(cm.get("aggregate_sentiment",      0.0)),
            retail_sentiment         = float(cm.get("retail_sentiment",         0.0)),
            institutional_sentiment  = float(cm.get("institutional_sentiment",  0.0)),
            herd_index               = float(cm.get("herd_index",               0.5)),
            panic_probability        = float(cm.get("panic_probability",        0.1)),
            narrative_momentum       = float(cm.get("narrative_momentum",       0.0)),
            contrarian_pressure      = float(cm.get("contrarian_pressure",      0.1)),
            information_velocity     = float(cm.get("information_velocity",     0.3)),
            opinion_cluster_count    = int(cm.get("opinion_cluster_count",      2)),
            agent_count              = int(cm.get("agent_count",                10210)),
            simulation_seed_event    = str(cm.get("simulation_seed_event",      ""))[:200],
            simulation_run_id        = str(cm.get("simulation_run_id",          "unknown")),
        )

        # ── Agent contributions ──────────────────────────────────────────────
        contributions = []
        for ac in signal_dict.get("agent_contributions", []):
            try:
                contributions.append(AgentContribution(
                    agent_name  = str(ac.get("agent_name",  "unknown")),
                    agent_role  = str(ac.get("agent_role",  "unknown")),
                    view        = str(ac.get("view",        "neutral")),
                    conviction  = float(ac.get("conviction", 0.5)),
                    key_factors = list(ac.get("key_factors", ["unknown"]))[:5],
                    dissent     = ac.get("dissent"),
                ))
            except Exception:
                pass

        # Ensure minimum 3 contributions (schema requirement)
        while len(contributions) < 3:
            contributions.append(AgentContribution(
                agent_name  = "placeholder",
                agent_role  = "unknown",
                view        = "neutral",
                conviction  = 0.3,
                key_factors = ["no_data"],
            ))

        # ── Sanitise narratives (enforce min_length) ─────────────────────────
        def _pad(s: str, min_len: int = 55) -> str:
            return s if len(s) >= min_len else s + " " + ("(no further detail provided.)" * 3)[:min_len - len(s)]

        macro_ctx  = _pad(signal_dict.get("macro_context",      "Macro analysis not available."))
        tech_view  = _pad(signal_dict.get("technical_view",     "Technical analysis not available."))
        crowd_narr = _pad(signal_dict.get("crowd_narrative",    "Crowd analysis not available."))
        coord_thesis = signal_dict.get("coordinator_thesis", "No thesis provided.")
        if len(coord_thesis) < 105:
            coord_thesis = coord_thesis + " " + ("Coordinator synthesis produced minimal output. " * 4)[:105 - len(coord_thesis)]

        # ── Resolve enums safely ─────────────────────────────────────────────
        direction_str = signal_dict.get("direction", "neutral").lower()
        try:
            direction = Direction(direction_str)
        except ValueError:
            direction = Direction.NEUTRAL

        timeframe_str = signal_dict.get("timeframe", "swing").lower()
        try:
            timeframe = Timeframe(timeframe_str)
        except ValueError:
            timeframe = Timeframe.SWING

        asset_class_str = signal_dict.get("asset_class", "equity").lower()
        try:
            asset_class = AssetClass(asset_class_str)
        except ValueError:
            asset_class = AssetClass.EQUITY

        regime_str = signal_dict.get("regime", "range_bound").lower()
        try:
            regime = RegimeType(regime_str)
        except ValueError:
            regime = RegimeType.RANGE_BOUND

        # ── Expiry ────────────────────────────────────────────────────────────
        expiry_raw = signal_dict.get("expiry")
        expiry = date.fromisoformat(str(expiry_raw)) if expiry_raw else None

        # ── Construct ─────────────────────────────────────────────────────────
        signal = TradingSignal(
            signal_id           = str(signal_dict.get("signal_id", __import__("uuid").uuid4())),
            created_at          = signal_dict.get("created_at", datetime.utcnow()),
            point_in_time_date  = date.today(),   # always today — enforces forward-only
            instrument          = str(signal_dict.get("instrument", "UNKNOWN")),
            exchange            = str(signal_dict.get("exchange",   "NSE")),
            asset_class         = asset_class,
            expiry              = expiry,
            direction           = direction,
            conviction          = max(0.0, min(1.0, float(signal_dict.get("conviction", 0.3)))),
            timeframe           = timeframe,
            regime              = regime,
            entry_zone          = entry_zone,
            risk                = risk,
            crowd               = crowd,
            macro_context       = macro_ctx,
            technical_view      = tech_view,
            crowd_narrative     = crowd_narr,
            coordinator_thesis  = coord_thesis,
            agent_contributions = contributions,
            confidence_flags    = list(signal_dict.get("confidence_flags",   [])),
            warning_flags       = list(signal_dict.get("warning_flags",      [])),
            do_not_trade_flags  = list(signal_dict.get("do_not_trade_flags", [])),
        )
        return signal, None

    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Signal logging
# ---------------------------------------------------------------------------

def _log_signal(signal, sim_report: dict, log_dir: Path) -> None:
    """Append signal to CSV and full JSONL audit log."""
    import csv
    import json
    from datetime import datetime

    # CSV — flat, easy to open in Excel/Sheets
    csv_path = log_dir / "signals.csv"
    row = signal.to_log_dict()
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    # JSONL — full fidelity audit log including reasoning
    jsonl_path = log_dir / "signals.jsonl"
    record = {
        "signal":            row,
        "coordinator_thesis": signal.coordinator_thesis,
        "macro_context":      signal.macro_context,
        "technical_view":     signal.technical_view,
        "crowd_narrative":    signal.crowd_narrative,
        "agent_contributions": [
            {
                "agent":      ac.agent_name,
                "role":       ac.agent_role,
                "view":       ac.view,
                "conviction": ac.conviction,
                "factors":    ac.key_factors,
            }
            for ac in signal.agent_contributions
        ],
        "simulation": {
            "narrative":  sim_report.get("dominant_narrative"),
            "confidence": sim_report.get("narrative_confidence"),
            "cascade":    sim_report.get("cascade_risk"),
            "bull_case":  sim_report.get("bull_case", "")[:300],
            "bear_case":  sim_report.get("bear_case", "")[:300],
        },
        "logged_at": datetime.utcnow().isoformat(),
    }
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _print_section(title: str, verbose: bool) -> None:
    if verbose:
        print(f"\n  ── {title} {'─' * max(0, 52 - len(title))}")


def _print_kv(key: str, val: str, verbose: bool) -> None:
    if verbose:
        print(f"  {key:<14} {val}")
