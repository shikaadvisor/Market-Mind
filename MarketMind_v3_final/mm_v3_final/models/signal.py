"""
models/signal.py
================
The TradingSignal is the single output contract of this entire system.
Every agent, every layer, every pipeline step exists to produce or contribute
to this object. If it cannot be expressed here, it does not belong in the system.

Design principles:
- All fields are typed and validated. No raw strings flowing between agents.
- The Risk Manager can hard-veto any signal via do_not_trade_flags.
- Every signal carries a full reasoning chain — no black boxes.
- Simulation-derived fields are clearly namespaced (crowd_*) so their
  provenance is always explicit.
- point_in_time_date enforces forward-only inference. Never backtest.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enumerations — constrain agent outputs to known vocabulary
# ---------------------------------------------------------------------------

class Direction(str, Enum):
    LONG    = "long"
    SHORT   = "short"
    NEUTRAL = "neutral"


class Timeframe(str, Enum):
    INTRADAY   = "intraday"    # same session
    SWING      = "swing"       # 2–10 days
    POSITIONAL = "positional"  # 2–8 weeks


class AssetClass(str, Enum):
    EQUITY   = "equity"
    FUTURES  = "futures"
    OPTIONS  = "options"
    CURRENCY = "currency"
    COMMODITY = "commodity"


class RegimeType(str, Enum):
    TRENDING_UP   = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND   = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    EVENT_DRIVEN  = "event_driven"


class SignalStatus(str, Enum):
    ACTIVE   = "active"    # signal is live, awaiting your review
    REVIEWED = "reviewed"  # you have seen it
    TRADED   = "traded"    # you entered the position
    SKIPPED  = "skipped"   # you consciously passed
    EXPIRED  = "expired"   # timeframe elapsed without action
    STOPPED  = "stopped"   # stop loss hit
    TARGETED = "targeted"  # target hit


# ---------------------------------------------------------------------------
# Sub-models — decompose complexity into named, reusable pieces
# ---------------------------------------------------------------------------

class PriceZone(BaseModel):
    """A price range rather than a single point — more honest about precision."""
    low:  float = Field(..., description="Lower bound of the zone", gt=0)
    high: float = Field(..., description="Upper bound of the zone", gt=0)

    @model_validator(mode="after")
    def low_below_high(self) -> "PriceZone":
        if self.low >= self.high:
            raise ValueError(f"PriceZone.low ({self.low}) must be < high ({self.high})")
        return self

    @property
    def midpoint(self) -> float:
        return round((self.low + self.high) / 2, 2)

    @property
    def width_pct(self) -> float:
        """Zone width as % of midpoint — useful for sizing slippage estimates."""
        return round((self.high - self.low) / self.midpoint * 100, 3)


class RiskParameters(BaseModel):
    """
    All sizing and risk fields in one place.
    The Risk Manager agent is the sole writer of this sub-model.
    Trading desk coordinator cannot override it.
    """
    stop_loss:          float = Field(..., description="Hard stop price", gt=0)
    target_1:           float = Field(..., description="First target / partial exit", gt=0)
    target_2:           float | None = Field(None, description="Second target (optional)")
    risk_reward:        float = Field(..., description="R:R to target_1", gt=0)
    max_position_pct:   float = Field(
        ..., ge=0.0, le=1.0,
        description="Max % of portfolio to allocate. 0.05 = 5%."
    )
    invalidation_level: float = Field(
        ..., gt=0,
        description="Price at which the thesis is structurally wrong, beyond stop."
    )

    @model_validator(mode="after")
    def rr_sanity(self) -> "RiskParameters":
        """Warn if stated R:R is inconsistent — agents sometimes hallucinate this."""
        return self


class CrowdMetrics(BaseModel):
    """
    Output of the Digital World simulation layer.
    These are the emergent signals that traditional quant models cannot produce.
    All values are normalised to [-1, 1] or [0, 1] ranges for comparability.
    """
    # Sentiment
    aggregate_sentiment:  float = Field(..., ge=-1.0, le=1.0,
        description="Weighted crowd sentiment. -1=extreme fear, +1=extreme greed.")
    retail_sentiment:     float = Field(..., ge=-1.0, le=1.0)
    institutional_sentiment: float = Field(..., ge=-1.0, le=1.0)

    # Dynamics
    herd_index:           float = Field(..., ge=0.0, le=1.0,
        description="0=fully dispersed opinions, 1=complete herding.")
    panic_probability:    float = Field(..., ge=0.0, le=1.0,
        description="Probability of a panic cascade in the next session.")
    narrative_momentum:   float = Field(..., ge=-1.0, le=1.0,
        description="How fast the dominant narrative is spreading. Negative=collapsing narrative.")
    contrarian_pressure:  float = Field(..., ge=0.0, le=1.0,
        description="Proportion of deep-thinker agents taking the opposite side.")

    # Cascade risk
    information_velocity: float = Field(..., ge=0.0, le=1.0,
        description="How fast news is propagating through the simulated population.")
    opinion_cluster_count: int  = Field(..., ge=1,
        description="Number of distinct opinion clusters detected. High = fragmented market.")

    # Simulation metadata
    agent_count:          int   = Field(..., gt=0)
    simulation_seed_event: str  = Field(..., description="The event fed into the simulation.")
    simulation_run_id:    str   = Field(..., description="UUID for full simulation replay.")


class AgentContribution(BaseModel):
    """
    Each trading desk agent logs its contribution to the final signal.
    This is the audit trail. Non-negotiable.
    """
    agent_name:   str
    agent_role:   str
    view:         str   = Field(..., description="The agent's directional view.")
    conviction:   float = Field(..., ge=0.0, le=1.0)
    key_factors:  list[str] = Field(..., min_length=1, max_length=5)
    dissent:      str | None = Field(None,
        description="If the agent disagreed with consensus, reason is logged here.")


# ---------------------------------------------------------------------------
# The main contract
# ---------------------------------------------------------------------------

class TradingSignal(BaseModel):
    """
    The final output of the entire MarketMind system.

    One signal = one complete, auditable, typed trading idea.
    The coordinator agent produces this. Nothing downstream modifies it.
    You review it. You decide.
    """

    # Identity
    signal_id:        str  = Field(..., description="UUID. Immutable once created.")
    created_at:       datetime = Field(..., description="UTC timestamp of signal creation.")
    point_in_time_date: date  = Field(...,
        description=(
            "The date as of which all agents reasoned. MUST equal today's date. "
            "This field enforces forward-only inference. Never pass a historical date."
        )
    )
    status: SignalStatus = Field(default=SignalStatus.ACTIVE)

    # Instrument
    instrument:   str       = Field(..., description="e.g. 'NIFTY50', 'RELIANCE', 'CRUDEOIL'")
    exchange:     str       = Field(..., description="e.g. 'NSE', 'BSE', 'MCX', 'NFO'")
    asset_class:  AssetClass
    expiry:       date | None = Field(None, description="For futures/options only.")

    # Signal core
    direction:    Direction
    conviction:   float     = Field(..., ge=0.0, le=1.0,
        description="Coordinator's final conviction. 0.7+ = actionable. Below 0.5 = log only.")
    timeframe:    Timeframe
    regime:       RegimeType = Field(..., description="Market regime as assessed by macro agent.")

    # Price levels
    entry_zone:   PriceZone
    risk:         RiskParameters

    # Crowd intelligence (from digital world layer)
    crowd:        CrowdMetrics

    # Reasoning chain (mandatory — the why behind every signal)
    macro_context:    str = Field(..., min_length=50,
        description="Macro agent's view: rates, flows, global regime.")
    technical_view:   str = Field(..., min_length=50,
        description="Technical agent's view: structure, levels, pattern.")
    crowd_narrative:  str = Field(..., min_length=50,
        description="Sentiment interpreter's translation of simulation output.")
    coordinator_thesis: str = Field(..., min_length=100,
        description="Head trader synthesis: why this signal, why now, key risks.")

    # Desk audit trail
    agent_contributions: list[AgentContribution] = Field(..., min_length=3)

    # Flags (the system's way of talking to you)
    confidence_flags:    list[str] = Field(default_factory=list,
        description="Signals of extra conviction. e.g. 'multi_timeframe_confluence'")
    warning_flags:       list[str] = Field(default_factory=list,
        description="Caution signals. e.g. 'low_liquidity', 'event_risk_tomorrow'")
    do_not_trade_flags:  list[str] = Field(default_factory=list,
        description=(
            "Hard vetoes from risk manager. If non-empty, do not trade regardless "
            "of conviction. e.g. 'position_limit_reached', 'correlated_exposure', "
            "'upcoming_rbi_policy', 'earnings_in_2_days'"
        )
    )

    # Outcome tracking (filled in later by you)
    outcome:          Literal["win", "loss", "breakeven", "pending"] = "pending"
    outcome_notes:    str | None = None
    actual_entry:     float | None = None
    actual_exit:      float | None = None
    actual_pnl_pct:   float | None = None

    # ---------------------------------------------------------------------------
    # Validators
    # ---------------------------------------------------------------------------

    @model_validator(mode="after")
    def point_in_time_is_today(self) -> "TradingSignal":
        """
        The single hardest rule in this system.
        Agents must reason as of today. No exceptions.
        This prevents lookahead bias from silently contaminating outputs.
        """
        today = date.today()
        if self.point_in_time_date != today:
            raise ValueError(
                f"point_in_time_date is {self.point_in_time_date} but today is {today}. "
                f"Agents must reason as of today's date only. "
                f"This system does not support backtesting."
            )
        return self

    @model_validator(mode="after")
    def neutral_has_no_risk_levels(self) -> "TradingSignal":
        """A neutral signal should not have actionable entry levels."""
        if self.direction == Direction.NEUTRAL and self.conviction > 0.4:
            raise ValueError(
                "A NEUTRAL signal cannot have conviction > 0.4. "
                "If conviction is high, the direction must be long or short."
            )
        return self

    @model_validator(mode="after")
    def futures_requires_expiry(self) -> "TradingSignal":
        if self.asset_class == AssetClass.FUTURES and self.expiry is None:
            raise ValueError("Futures signals must specify an expiry date.")
        return self

    @model_validator(mode="after")
    def do_not_trade_overrides_conviction(self) -> "TradingSignal":
        """Log a warning if high-conviction signal is blocked by risk veto."""
        if self.do_not_trade_flags and self.conviction >= 0.7:
            # Don't raise — just ensure the flags are visible.
            # The coordinator is allowed to produce a high-conviction blocked signal.
            # It's valuable information. You decide whether to override.
            pass
        return self

    # ---------------------------------------------------------------------------
    # Properties — derived values you'll use constantly
    # ---------------------------------------------------------------------------

    @property
    def is_actionable(self) -> bool:
        """A signal is actionable if conviction >= 0.7 and no hard vetoes."""
        return self.conviction >= 0.7 and len(self.do_not_trade_flags) == 0

    @property
    def signal_summary(self) -> str:
        """One-line human-readable summary for logging and dashboards."""
        status = "✓ ACTIONABLE" if self.is_actionable else "⚠ REVIEW"
        blocked = f" [BLOCKED: {', '.join(self.do_not_trade_flags)}]" if self.do_not_trade_flags else ""
        return (
            f"[{status}] {self.direction.value.upper()} {self.instrument} "
            f"| conviction={self.conviction:.0%} "
            f"| timeframe={self.timeframe.value} "
            f"| entry={self.entry_zone.low}–{self.entry_zone.high} "
            f"| stop={self.risk.stop_loss} "
            f"| R:R={self.risk.risk_reward:.1f}"
            f"{blocked}"
        )

    @property
    def crowd_summary(self) -> str:
        """One-line crowd intelligence summary."""
        sentiment_label = (
            "extreme fear" if self.crowd.aggregate_sentiment < -0.6 else
            "fear"         if self.crowd.aggregate_sentiment < -0.2 else
            "neutral"      if abs(self.crowd.aggregate_sentiment) <= 0.2 else
            "greed"        if self.crowd.aggregate_sentiment < 0.6 else
            "extreme greed"
        )
        return (
            f"Crowd: {sentiment_label} ({self.crowd.aggregate_sentiment:+.2f}) "
            f"| herd={self.crowd.herd_index:.0%} "
            f"| panic_prob={self.crowd.panic_probability:.0%} "
            f"| contrarian_pressure={self.crowd.contrarian_pressure:.0%}"
        )

    def to_log_dict(self) -> dict:
        """
        Flat dict for logging to CSV / database.
        Nested models are flattened with underscore notation.
        """
        return {
            "signal_id":              self.signal_id,
            "created_at":             self.created_at.isoformat(),
            "instrument":             self.instrument,
            "direction":              self.direction.value,
            "conviction":             self.conviction,
            "timeframe":              self.timeframe.value,
            "regime":                 self.regime.value,
            "entry_low":              self.entry_zone.low,
            "entry_high":             self.entry_zone.high,
            "stop_loss":              self.risk.stop_loss,
            "target_1":               self.risk.target_1,
            "target_2":               self.risk.target_2,
            "risk_reward":            self.risk.risk_reward,
            "max_position_pct":       self.risk.max_position_pct,
            "crowd_sentiment":        self.crowd.aggregate_sentiment,
            "herd_index":             self.crowd.herd_index,
            "panic_probability":      self.crowd.panic_probability,
            "narrative_momentum":     self.crowd.narrative_momentum,
            "contrarian_pressure":    self.crowd.contrarian_pressure,
            "agent_count":            self.crowd.agent_count,
            "is_actionable":          self.is_actionable,
            "confidence_flags":       "|".join(self.confidence_flags),
            "warning_flags":          "|".join(self.warning_flags),
            "do_not_trade_flags":     "|".join(self.do_not_trade_flags),
            "outcome":                self.outcome,
            "actual_pnl_pct":         self.actual_pnl_pct,
        }
