"""
models/simulation.py
====================
The SimulationReport is the output contract of the Digital World layer.
It is the structured handoff between Layer 1 (swarm simulation) and
Layer 2 (trading desk). The Sentiment Interpreter agent on the desk
reads this object and translates it into crowd_narrative for the signal.

Key design choice: this is NOT a TradingSignal. The simulation layer
does not produce trade recommendations. It produces a description of
crowd psychology. The desk layer decides what to do with that.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class NarrativeType(str, Enum):
    """The dominant narrative running through the simulated population."""
    BULLISH_FUNDAMENTALS  = "bullish_fundamentals"
    BEARISH_FUNDAMENTALS  = "bearish_fundamentals"
    TECHNICAL_BREAKOUT    = "technical_breakout"
    TECHNICAL_BREAKDOWN   = "technical_breakdown"
    MACRO_RISK_OFF        = "macro_risk_off"
    MACRO_RISK_ON         = "macro_risk_on"
    EVENT_DRIVEN_EUPHORIA = "event_driven_euphoria"
    EVENT_DRIVEN_PANIC    = "event_driven_panic"
    SHORT_SQUEEZE         = "short_squeeze"
    CAPITULATION          = "capitulation"
    DISTRIBUTION          = "distribution"
    ACCUMULATION          = "accumulation"
    CONFUSION             = "confusion"  # no dominant narrative


class CascadeRisk(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"
    EXTREME = "extreme"


# ---------------------------------------------------------------------------
# Agent population snapshot — one per tier
# ---------------------------------------------------------------------------

class TierSnapshot(BaseModel):
    """Summary statistics from one tier of the agent population."""
    tier:           Literal[1, 2, 3]
    agent_count:    int   = Field(..., gt=0)
    bullish_pct:    float = Field(..., ge=0.0, le=1.0)
    bearish_pct:    float = Field(..., ge=0.0, le=1.0)
    neutral_pct:    float = Field(..., ge=0.0, le=1.0)
    avg_conviction: float = Field(..., ge=0.0, le=1.0)
    notable_behaviors: list[str] = Field(default_factory=list,
        description="Key behavioral patterns observed in this tier.")

    @model_validator(mode="after")
    def pcts_sum_to_one(self) -> "TierSnapshot":
        total = round(self.bullish_pct + self.bearish_pct + self.neutral_pct, 4)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Tier {self.tier} percentages sum to {total}, must be ~1.0")
        return self


# ---------------------------------------------------------------------------
# Opinion cluster — emergent group that formed during the simulation
# ---------------------------------------------------------------------------

class OpinionCluster(BaseModel):
    """
    A distinct opinion cluster that emerged during simulation.
    In real markets, these map to the different 'camps' you see on
    financial Twitter, in analyst reports, in options positioning.
    """
    cluster_id:      str
    size_pct:        float  = Field(..., ge=0.0, le=1.0,
        description="% of total population in this cluster.")
    dominant_view:   str    = Field(..., description="The narrative this cluster holds.")
    conviction:      float  = Field(..., ge=0.0, le=1.0)
    is_influential:  bool   = Field(...,
        description="True if this cluster disproportionately influences other tiers.")
    key_agents:      list[str] = Field(default_factory=list,
        description="Agent IDs of the most influential members of this cluster.")


# ---------------------------------------------------------------------------
# Cascade event — a moment of rapid opinion shift during simulation
# ---------------------------------------------------------------------------

class CascadeEvent(BaseModel):
    """
    A tipping-point moment observed during the simulation run.
    These are the most valuable outputs — they show you where the crowd
    is fragile and what could trigger a rapid move.
    """
    timestep:        int
    trigger:         str   = Field(..., description="What caused the cascade.")
    direction:       Literal["bullish_cascade", "bearish_cascade"]
    magnitude:       float = Field(..., ge=0.0, le=1.0,
        description="How severe the cascade was. 1.0 = full population shifted.")
    affected_tiers:  list[Literal[1, 2, 3]]
    recovery_steps:  int | None = Field(None,
        description="How many timesteps before the population stabilised.")


# ---------------------------------------------------------------------------
# The main simulation report
# ---------------------------------------------------------------------------

class SimulationReport(BaseModel):
    """
    The complete output of one Digital World simulation run.
    Produced by the simulation pipeline, consumed by the trading desk's
    Sentiment Interpreter agent.

    One report = one seed event = one simulation run.
    """

    # Identity
    run_id:           str      = Field(..., description="UUID. Links to full simulation logs.")
    created_at:       datetime = Field(..., description="UTC timestamp.")
    point_in_time_date: date   = Field(...,
        description="Must equal today's date. Forward inference only.")

    # Seed
    seed_event:       str      = Field(..., min_length=20,
        description="The event that was fed into the simulation population.")
    instrument:       str      = Field(..., description="Instrument being simulated.")
    simulation_steps: int      = Field(..., gt=0,
        description="Number of interaction timesteps run.")

    # Population
    total_agents:     int      = Field(..., gt=0)
    tier_snapshots:   list[TierSnapshot] = Field(..., min_length=1, max_length=3)

    # Emergent outputs — the stuff that matters
    dominant_narrative:        NarrativeType
    narrative_confidence:      float = Field(..., ge=0.0, le=1.0,
        description="How clearly the dominant narrative emerged vs. noise.")
    narrative_shift_detected:  bool  = Field(...,
        description="True if the narrative changed direction mid-simulation.")

    opinion_clusters:  list[OpinionCluster] = Field(..., min_length=1)
    cascade_events:    list[CascadeEvent]   = Field(default_factory=list)
    cascade_risk:      CascadeRisk

    # Aggregate metrics (same fields as CrowdMetrics in signal.py, computed here)
    aggregate_sentiment:     float = Field(..., ge=-1.0, le=1.0)
    retail_sentiment:        float = Field(..., ge=-1.0, le=1.0)
    institutional_sentiment: float = Field(..., ge=-1.0, le=1.0)
    herd_index:              float = Field(..., ge=0.0, le=1.0)
    panic_probability:       float = Field(..., ge=0.0, le=1.0)
    narrative_momentum:      float = Field(..., ge=-1.0, le=1.0)
    contrarian_pressure:     float = Field(..., ge=0.0, le=1.0)
    information_velocity:    float = Field(..., ge=0.0, le=1.0)

    # Qualitative synthesis (written by Tier 1 deep thinker agents)
    bull_case:        str = Field(..., min_length=50,
        description="Strongest bullish argument that emerged from simulation.")
    bear_case:        str = Field(..., min_length=50,
        description="Strongest bearish argument that emerged from simulation.")
    key_risk:         str = Field(..., min_length=30,
        description="The single biggest risk to the dominant narrative.")
    contrarian_thesis: str = Field(..., min_length=30,
        description="The most coherent contrarian view from the simulation.")

    # Desk handoff
    sentiment_interpreter_brief: str = Field(..., min_length=100,
        description=(
            "Pre-written brief for the Sentiment Interpreter agent on the trading desk. "
            "Should be direct, structured, and translate simulation output into "
            "market-relevant language. Written by the simulation coordinator."
        )
    )

    # ---------------------------------------------------------------------------
    # Validators
    # ---------------------------------------------------------------------------

    @model_validator(mode="after")
    def point_in_time_is_today(self) -> "SimulationReport":
        today = date.today()
        if self.point_in_time_date != today:
            raise ValueError(
                f"point_in_time_date is {self.point_in_time_date} but today is {today}. "
                f"Forward inference only. This system does not backtest."
            )
        return self

    @model_validator(mode="after")
    def tier_agent_counts_match_total(self) -> "SimulationReport":
        tier_sum = sum(t.agent_count for t in self.tier_snapshots)
        if abs(tier_sum - self.total_agents) > 10:  # allow small rounding
            raise ValueError(
                f"Sum of tier agent counts ({tier_sum}) does not match "
                f"total_agents ({self.total_agents})."
            )
        return self

    # ---------------------------------------------------------------------------
    # Properties
    # ---------------------------------------------------------------------------

    @property
    def has_cascade_risk(self) -> bool:
        return self.cascade_risk in (CascadeRisk.HIGH, CascadeRisk.EXTREME)

    @property
    def is_consensus_strong(self) -> bool:
        """True if the population has converged on a single narrative."""
        return self.narrative_confidence >= 0.7 and self.herd_index >= 0.6

    @property
    def report_summary(self) -> str:
        cascade = f" ⚡ CASCADE RISK: {self.cascade_risk.value.upper()}" if self.has_cascade_risk else ""
        return (
            f"[SIM {self.run_id[:8]}] {self.instrument} | "
            f"narrative={self.dominant_narrative.value} "
            f"(conf={self.narrative_confidence:.0%}) | "
            f"sentiment={self.aggregate_sentiment:+.2f} | "
            f"herd={self.herd_index:.0%} | "
            f"panic_prob={self.panic_probability:.0%}"
            f"{cascade}"
        )
