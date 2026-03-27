"""
workflows.py
============
Named workflow configurations for MarketMind v3.

Usage in main_v3.py:
    python main_v3.py --workflow ags_daily
    python main_v3.py --workflow ags_intraday
    python main_v3.py --workflow india_swing

Each workflow defines:
  - instruments to watch
  - polling frequency and session awareness
  - sensitivity and timeframe
  - swarm mode (full vs lite)
  - daily cost cap
  - quant model overlay (for ags - to compare against existing model)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WorkflowConfig:
    name:               str
    description:        str
    instruments:        list[str]

    # Polling
    poll_interval_sec:  int           # seconds between polls
    session_aware:      bool          # only poll during market hours
    sensitivity:        str           # "high" | "medium" | "low"

    # Signal parameters
    default_timeframe:  str           # "intraday" | "swing" | "positional"
    min_conviction:     float         # signals below this are not tracked

    # Swarm mode
    lite_mode:          bool          # True = T3 only, no T2/T1 (saves ~$0.22/run)
    n_tier2_per_universe: int         # 0 in lite mode
    n_tier1_per_universe: int         # 0 in lite mode

    # Cost control
    daily_cost_cap_usd: float

    # Quant model overlay — for comparing with an existing model
    quant_model_name:   Optional[str] = None  # e.g. "daily_momentum_v2"
    quant_model_instruments: list[str] = field(default_factory=list)

    # Special event handling
    force_run_on_calendar_event: bool = True  # always run full pipeline on WASDE days


# ---------------------------------------------------------------------------
# Ags daily — alongside existing quant model
# ---------------------------------------------------------------------------

AGS_DAILY = WorkflowConfig(
    name        = "ags_daily",
    description = "CBOT/ICE ags portfolio — daily swing signals alongside quant model",
    instruments = ["ZC", "ZS", "ZW", "SB", "CT"],

    poll_interval_sec  = 4 * 3600,   # 4 hours
    session_aware      = True,        # don't poll overnight or weekends
    sensitivity        = "medium",

    default_timeframe  = "swing",
    min_conviction     = 0.55,        # slightly higher than default — ags are noisy

    lite_mode          = False,       # full swarm for daily signals
    n_tier2_per_universe = 50,
    n_tier1_per_universe = 3,

    daily_cost_cap_usd = 8.0,

    quant_model_name   = "your_daily_quant_model",
    quant_model_instruments = ["ZC", "ZS", "ZW", "SB", "CT"],
    force_run_on_calendar_event = True,
)


# ---------------------------------------------------------------------------
# Ags intraday — new strategy development layer
# ---------------------------------------------------------------------------

AGS_INTRADAY = WorkflowConfig(
    name        = "ags_intraday",
    description = "CBOT/ICE ags — intraday session signals (lite mode)",
    instruments = ["ZC", "ZS", "ZW", "SB", "CT"],

    poll_interval_sec  = 30 * 60,    # 30 minutes
    session_aware      = True,        # CRITICAL: only poll during open session

    # Tighter sensitivity for intraday — we want real breakouts, not noise
    sensitivity        = "high",

    default_timeframe  = "intraday",
    min_conviction     = 0.65,        # higher bar — intraday is much noisier

    # Lite mode: T3 ABM only (10k agents, no LLM calls except coordinator)
    # Saves ~$0.22/run vs full mode → viable at 30min polling
    lite_mode          = True,
    n_tier2_per_universe = 0,
    n_tier1_per_universe = 0,

    daily_cost_cap_usd = 3.0,        # low cap — lite mode is cheap
    force_run_on_calendar_event = True,
)


# ---------------------------------------------------------------------------
# India swing — phase 1 test
# ---------------------------------------------------------------------------

INDIA_SWING = WorkflowConfig(
    name        = "india_swing",
    description = "India + safe-haven — swing timeframe, phase 1 test",
    instruments = ["NIFTY50", "USDINR", "GC"],

    poll_interval_sec  = 4 * 3600,
    session_aware      = True,
    sensitivity        = "low",       # conservative for phase 1

    default_timeframe  = "swing",
    min_conviction     = 0.60,

    lite_mode          = False,
    n_tier2_per_universe = 50,
    n_tier1_per_universe = 3,

    daily_cost_cap_usd = 5.0,
    force_run_on_calendar_event = True,
)


# ---------------------------------------------------------------------------
# Combined — all three running together
# ---------------------------------------------------------------------------

ALL_WORKFLOWS = {
    "ags_daily":    AGS_DAILY,
    "ags_intraday": AGS_INTRADAY,
    "india_swing":  INDIA_SWING,
}


def get_workflow(name: str) -> WorkflowConfig:
    if name not in ALL_WORKFLOWS:
        raise ValueError(f"Unknown workflow '{name}'. Valid: {list(ALL_WORKFLOWS)}")
    return ALL_WORKFLOWS[name]


def print_workflow_summary() -> None:
    print("\n  Available workflows:\n")
    for name, wf in ALL_WORKFLOWS.items():
        mode = "LITE (T3 only)" if wf.lite_mode else "FULL (T3+T2+T1)"
        print(f"  --workflow {name}")
        print(f"    {wf.description}")
        print(f"    Instruments: {', '.join(wf.instruments)}")
        print(f"    Poll: {wf.poll_interval_sec//60}min  Sensitivity: {wf.sensitivity}  Mode: {mode}")
        print(f"    Daily cap: ${wf.daily_cost_cap_usd:.0f}  Min conviction: {wf.min_conviction:.0%}")
        if wf.quant_model_name:
            print(f"    Quant overlay: {wf.quant_model_name}")
        print()
