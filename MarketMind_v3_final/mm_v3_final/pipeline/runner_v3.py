"""
pipeline/runner_v3.py  —  MarketMind v3 (all bugs fixed)
=========================================================

Bug fixes applied in this file:
    BUG-15  Feedback context now fetched AFTER the desk runs, using the actual
            regime and direction from the macro agent's output, not hardcoded
            "range_bound" / "neutral". The second swarm weight calibration call
            also uses the real regime.

All other bugs fixed in their respective modules:
    BUG-12  → simulation/swarm_runner.py (no global persona mutation)
    BUG-13  → simulation/tiers/tier3_mesa.py (UniverseBias applied)
    BUG-14  → simulation/swarm_runner.py (ZeroDivision guard)
    Risk-1  → desk/coordinator_v3.py (devil's advocate two-pass)
    Risk-3  → utils/model_registry.py (pinned dated model versions)
    Risk-4  → utils/retry.py (exponential backoff + circuit breaker)
    Risk-9  → utils/model_registry.py (coordinator model from registry)
    Risk-10 → utils/cost_tracker.py (hard daily cap)

Architecture:
    HardenedFeed → SwarmRunner (4 universes) → DeskGraphV3 → ForwardMonitor
    Feedback loop: ForwardMonitor outcomes → PopulationFeedback → SwarmRunner weights
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger("marketmind.v3")


# ---------------------------------------------------------------------------
# Safe asyncio runner (works in Jupyter, FastAPI, scripts)
# ---------------------------------------------------------------------------

def _run_safely(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    try:
        import nest_asyncio
        nest_asyncio.apply(loop)
        return loop.run_until_complete(coro)
    except ImportError:
        pass
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfigV3:
    # API keys
    openai_api_key:    str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY",    ""))
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))

    # Swarm (per universe — total = ×4)
    n_agents_per_universe: int   = 2_500
    n_tier2_per_universe:  int   = 50
    n_steps:               int   = 50

    # Robustness
    mock:               bool  = False
    agent_timeout_sec:  int   = 90
    swarm_timeout_sec:  int   = 240
    abort_on_stale:     bool  = True
    min_conviction_log: float = 0.30

    # Cost guard (Risk 10)
    daily_cost_cap_usd: float = 20.0

    # Feedback
    enable_feedback:        bool = True
    feedback_min_outcomes:  int  = 20

    # Regression test (Risk 3)
    run_regression_on_start: bool = False

    # Logging
    log_dir:  Path = field(default_factory=lambda: Path("logs"))
    verbose:  bool = True


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResultV3:
    signal:             object
    swarm_report:       object
    desk_state:         dict
    live_signal_record: object
    elapsed_sec:        float
    cost_usd:           float = 0.0
    error:              Optional[str] = None
    data_warnings:      list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def is_actionable(self) -> bool:
        return (
            self.success
            and self.signal is not None
            and getattr(self.signal, "is_actionable", False)
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class MarketMindV3:

    def __init__(self, config: PipelineConfigV3):
        self.cfg = config
        self.cfg.log_dir.mkdir(exist_ok=True)

        # Cost guard (Risk 10)
        from utils.cost_tracker import CostTracker
        self._cost = CostTracker(
            daily_cap_usd = config.daily_cost_cap_usd,
            log_dir       = config.log_dir,
        )

        # Swarm runner
        from simulation.swarm_runner import SwarmRunner
        self._swarm = SwarmRunner(
            openai_api_key = config.openai_api_key,
            n_agents       = config.n_agents_per_universe,
            n_steps        = config.n_steps,
            n_tier2        = config.n_tier2_per_universe,
            mock           = config.mock,
            verbose        = config.verbose,
        )

        # Trading desk (v3 graph with devil's advocate + model registry)
        from desk.graph_v3 import DeskGraphV3
        self._desk = DeskGraphV3(
            openai_api_key    = config.openai_api_key,
            anthropic_api_key = config.anthropic_api_key,
            mock              = config.mock,
            agent_timeout     = config.agent_timeout_sec,
        )

        # Forward monitor
        from pipeline.forward_monitor import ForwardMonitor
        self._monitor = ForwardMonitor(log_dir=config.log_dir)

        # Feedback loop
        from feedback.population_feedback import PopulationFeedback
        self._feedback = PopulationFeedback(log_dir=config.log_dir)

        # Apply calibrated weights from previous feedback runs
        if config.enable_feedback:
            saved = self._feedback.state.global_universe_weights()
            if saved and any(v != 0.25 for v in saved.values()):
                self._swarm.apply_feedback(saved)

        if config.verbose:
            est = self._cost.estimate_run_cost()
            _v(f"MarketMind v3  ·  {config.n_agents_per_universe*4:,} T3 agents  "
               f"·  {'MOCK' if config.mock else 'LIVE'}", True)
            _v(f"Est. cost/run: ${est:.4f}  ·  Daily cap: ${config.daily_cost_cap_usd:.2f}  "
               f"·  Remaining today: ${self._cost.today_remaining:.4f}", True)
            _v(f"Active signals: {len(self._monitor.active_signals)}  "
               f"·  Feedback outcomes: {self._feedback.state.outcomes_used}", True)

        # Optional regression test on startup
        if config.run_regression_on_start and not config.mock:
            _run_safely(self._regression_check())

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(
        self,
        event,
        instrument:    str = "NIFTY50",
        exchange:      str = "NSE",
        asset_class:   str = "equity",
        market_data:   dict | None = None,
        expiry_date:   str | None = None,
        data_warnings: list[str] | None = None,
    ) -> PipelineResultV3:
        return _run_safely(self._run_async(
            event, instrument, exchange, asset_class,
            market_data, expiry_date, data_warnings or []
        ))

    async def _run_async(
        self,
        event,
        instrument:    str,
        exchange:      str,
        asset_class:   str,
        market_data:   dict | None,
        expiry_date:   str | None,
        data_warnings: list[str],
    ) -> PipelineResultV3:

        t_start = time.perf_counter()

        # ── Pre-flight cost check (Risk 10) ──────────────────────────────────
        from utils.cost_tracker import BudgetExceeded
        try:
            est = self._cost.estimate_run_cost()
            self._cost.check("tier2", 400 * self.cfg.n_tier2_per_universe * 4,
                             100 * self.cfg.n_tier2_per_universe * 4)
        except BudgetExceeded as e:
            logger.error("Budget exceeded: %s", e)
            return _err(str(e), t_start, data_warnings)

        # ── Step 1: Swarm simulation ─────────────────────────────────────────
        _v("STEP 1/3 — Multi-universe swarm", self.cfg.verbose)
        try:
            swarm = await asyncio.wait_for(
                self._swarm.run_async(event, instrument, asset_class),
                timeout=self.cfg.swarm_timeout_sec,
            )
        except asyncio.TimeoutError:
            return _err(f"Swarm timed out ({self.cfg.swarm_timeout_sec}s)", t_start, data_warnings)
        except Exception as e:
            logger.exception("Swarm failed: %s", e)
            return _err(str(e), t_start, data_warnings)

        if self.cfg.verbose:
            _v(f"Swarm: sent={swarm.ensemble_sentiment:+.4f}  "
               f"div={swarm.universe_divergence:.3f}  "
               f"contrarian={'YES' if swarm.contrarian_setup else 'no'}", True)

        # ── Step 2: Desk (no feedback context yet — BUG-15 fix deferred) ─────
        _v("STEP 2/3 — Trading desk", self.cfg.verbose)
        from desk.graph_v3 import build_initial_state_v3

        # Get calibrated universe weights for this asset class
        # We don't know the regime yet, so use asset-class-only weights
        cal_weights = self._feedback.get_universe_weights_for(asset_class, "any")
        if cal_weights:
            self._swarm.apply_feedback(cal_weights)

        initial_state = build_initial_state_v3(
            instrument        = instrument,
            exchange          = exchange,
            asset_class       = asset_class,
            event_description = event.description,
            swarm_report      = swarm,
            market_data       = market_data or {},
            expiry_date       = expiry_date,
            feedback_context  = "",  # injected after desk (BUG-15 fix below)
        )

        try:
            desk_state = await asyncio.wait_for(
                self._desk.run_async(initial_state),
                timeout=self.cfg.agent_timeout_sec * 5,
            )
        except asyncio.TimeoutError:
            return _err("Desk timed out", t_start, data_warnings)
        except Exception as e:
            logger.exception("Desk failed: %s", e)
            return _err(str(e), t_start, data_warnings)

        # ── BUG-15 FIX: feedback context with REAL regime + direction ────────
        # Now that the desk has run, we can read the actual macro regime and
        # the coordinator's direction from the produced signal.
        final_signal_dict = desk_state.get("final_signal", {})
        actual_regime = (
            desk_state.get("macro_view", {}).get("regime", "range_bound")
            or "range_bound"
        )
        actual_direction = final_signal_dict.get("direction", "neutral")

        if self.cfg.enable_feedback:
            feedback_ctx = self._feedback.get_context_for_signal(
                asset_class, actual_regime, actual_direction
            )
            if feedback_ctx:
                # Inject into coordinator_thesis as a postscript
                existing_thesis = final_signal_dict.get("coordinator_thesis", "")
                final_signal_dict["coordinator_thesis"] = (
                    existing_thesis + f"\n\n{feedback_ctx}"
                )
                # Re-apply regime-specific universe weights now we know the regime
                regime_weights = self._feedback.get_universe_weights_for(
                    asset_class, actual_regime
                )
                if regime_weights:
                    self._swarm.apply_feedback(regime_weights)

        # Propagate data warnings into signal
        if data_warnings:
            existing = list(final_signal_dict.get("warning_flags", []))
            final_signal_dict["warning_flags"] = list(set(existing + data_warnings))

        # ── Step 3: Signal validation ─────────────────────────────────────────
        _v("STEP 3/3 — Signal validation + seal", self.cfg.verbose)
        from pipeline.runner import _build_and_validate_signal
        signal, error = _build_and_validate_signal(
            final_signal_dict, swarm.metrics_dict, self.cfg
        )

        elapsed   = time.perf_counter() - t_start
        cost_usd  = self._cost.record("tier2",
                                       400 * self.cfg.n_tier2_per_universe * 4,
                                       100 * self.cfg.n_tier2_per_universe * 4)

        # ── Register with forward monitor ─────────────────────────────────────
        live_record = None
        if signal is not None and error is None:
            conv = float(getattr(signal, "conviction", 0))
            if conv >= self.cfg.min_conviction_log:
                live_record = self._monitor.register_signal(signal, swarm)
                if self.cfg.verbose:
                    from pipeline.forward_monitor import seal_signal
                    _v(f"Signal sealed {seal_signal(signal)[:12]}...  "
                       f"expires {live_record.expiry_date}  "
                       f"devil={final_signal_dict.get('devil_advocate',{}).get('strength',0):.2f}",
                       True)
            else:
                _v(f"Conviction {conv:.0%} below threshold — not tracked", self.cfg.verbose)

        if self.cfg.verbose:
            _v(f"Done in {elapsed:.1f}s  |  cost est. ${cost_usd:.4f}  "
               f"|  today total ${self._cost.today_spent:.4f}", True)

        return PipelineResultV3(
            signal             = signal,
            swarm_report       = swarm,
            desk_state         = dict(desk_state),
            live_signal_record = live_record,
            elapsed_sec        = round(elapsed, 2),
            cost_usd           = round(cost_usd, 6),
            error              = error,
            data_warnings      = data_warnings,
        )

    # -----------------------------------------------------------------------
    # Forward monitor commands
    # -----------------------------------------------------------------------

    def check_forward_signals(self, feed_kwargs: dict | None = None) -> list:
        """Poll prices for all active signals. Returns newly resolved records."""
        resolved = self._monitor.check_all(feed_kwargs)
        if self.cfg.verbose:
            for rec in resolved:
                _v(f"Resolved: {rec.instrument} {rec.direction.upper()} → "
                   f"{rec.status.upper()}  R={rec.r_multiple or 0:.2f}", True)
            _v(self._monitor.summary(), True)
        return resolved

    def suspend_signals(self, instrument: str, reason: str = "market_halt") -> int:
        """Suspend all active signals for an instrument (circuit breaker / halt)."""
        return self._monitor.suspend_by_instrument(instrument, reason)

    def run_feedback(self) -> None:
        """
        Run population feedback on all resolved outcomes.
        Requires cfg.feedback_min_outcomes resolved signals.
        Updates universe weights for future runs.
        """
        outcomes = self._monitor.get_outcomes_for_feedback()
        if len(outcomes) < self.cfg.feedback_min_outcomes:
            print(f"  Feedback: {len(outcomes)} outcomes, "
                  f"need {self.cfg.feedback_min_outcomes}. Skipping.")
            return
        state = self._feedback.run(outcomes)
        new_weights = state.global_universe_weights()
        self._swarm.apply_feedback(new_weights)
        print(self._feedback.report())

    # -----------------------------------------------------------------------
    # Regression test (Risk 3)
    # -----------------------------------------------------------------------

    async def _regression_check(self) -> None:
        from utils.model_registry import run_regression
        _v("Running model regression tests...", True)
        results = await run_regression(self.cfg.openai_api_key, self.cfg.log_dir)
        passed  = sum(1 for r in results if r.passed)
        _v(f"Regression: {passed}/{len(results)} passed", True)
        for r in results:
            if not r.passed:
                logger.warning("Regression FAIL [%s/%s]: %s", r.role, r.model, r.failures)

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def forward_monitor(self):
        return self._monitor

    @property
    def feedback(self):
        return self._feedback

    @property
    def cost(self):
        return self._cost


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _v(msg: str, verbose: bool) -> None:
    if verbose:
        print(f"  {msg}")


def _err(error: str, t_start: float, warnings: list) -> PipelineResultV3:
    return PipelineResultV3(
        signal=None, swarm_report=None, desk_state={},
        live_signal_record=None,
        elapsed_sec=round(time.perf_counter() - t_start, 2),
        error=error, data_warnings=warnings,
    )
