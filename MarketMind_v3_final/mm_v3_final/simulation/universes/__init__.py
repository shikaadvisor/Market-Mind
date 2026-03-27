"""
simulation/universes/
=====================
Four independent crowd universes, each with distinct psychological DNA.

The core insight behind multi-universe design:
    Markets are not populated by a single homogeneous crowd.
    They are simultaneously inhabited by momentum traders, deep-value investors,
    panic-prone retail, and structurally-driven institutional flows — all seeing
    the SAME event through completely different lenses.

    Running four separate simulations and ensembling them gives you something
    no single-universe system can: a map of HOW MUCH the crowd disagrees,
    and WHICH TYPE of player is driving the narrative.

Universe personalities:
    MOMENTUM  — trend-followers, FOMO traders, stop-hunters, chart pattern readers
    VALUE     — contrarians, fundamental PMs, mean-reversion players
    CRISIS    — tail-risk managers, panic sellers, safe-haven seekers
    STRUCTURAL — institutional flow, commodity fundamentalists, macro macro macro

Each universe has:
    - A distinct T3 population (different persona weights, bias parameters)
    - Different T2 personas (e.g. Crisis gets more risk-off hedge fund archetypes)
    - Different T1 deep thinkers (e.g. Value gets more Buffett-style analysts)
    - A base weight in the ensemble (tunable from outcome feedback)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class UniverseConfig:
    """Complete configuration for one simulation universe."""
    name:           str
    description:    str

    # Tier 3 crowd psychology — biases baked into the ABM population
    momentum_bias:      float   # 0-1: how strongly T3 agents chase price
    contrarian_bias:    float   # 0-1: fraction of population that fades moves
    panic_threshold:    float   # 0-1: how easily agents enter panic
    herding_strength:   float   # 0-1: how strongly agents copy neighbours
    news_reactivity:    float   # 0-1: speed at which news shifts sentiment
    loss_aversion:      float   # >1: Kahneman/Tversky multiplier

    # Tier 2 persona selection filter (asset class tags)
    t2_persona_tags:    list[str]

    # Tier 1 deep thinker selection filter
    t1_persona_tags:    list[str]

    # Ensemble weight — evidence-adjusted via feedback loop
    base_weight:        float   # initial weight before calibration
    calibrated_weight:  Optional[float] = None  # set by feedback after forward test

    @property
    def effective_weight(self) -> float:
        return self.calibrated_weight if self.calibrated_weight is not None else self.base_weight


# ── The four universes ──────────────────────────────────────────────────────

MOMENTUM_UNIVERSE = UniverseConfig(
    name        = "momentum",
    description = "Trend-followers, FOMO traders, stop-hunters, TA-driven retail",
    momentum_bias    = 0.85,
    contrarian_bias  = 0.08,
    panic_threshold  = 0.52,
    herding_strength = 0.72,
    news_reactivity  = 0.88,
    loss_aversion    = 3.2,
    t2_persona_tags  = ["momentum", "retail", "fomo", "options_gamma", "fin_twitter"],
    t1_persona_tags  = ["cta", "momentum_pm", "technical_strategist"],
    base_weight      = 0.30,
)

VALUE_UNIVERSE = UniverseConfig(
    name        = "value",
    description = "Contrarians, fundamental PMs, mean-reversion players, patient capital",
    momentum_bias    = 0.18,
    contrarian_bias  = 0.60,
    panic_threshold  = 0.78,
    herding_strength = 0.28,
    news_reactivity  = 0.32,
    loss_aversion    = 1.6,
    t2_persona_tags  = ["value", "contrarian", "fundamental", "long_only", "deep_value"],
    t1_persona_tags  = ["value_pm", "contrarian_macro", "fundamental_analyst"],
    base_weight      = 0.25,
)

CRISIS_UNIVERSE = UniverseConfig(
    name        = "crisis",
    description = "Tail-risk managers, panic sellers, safe-haven seekers, VIX watchers",
    momentum_bias    = 0.55,
    contrarian_bias  = 0.12,
    panic_threshold  = 0.35,  # much lower — panic easier here
    herding_strength = 0.80,  # crisis = herd behaviour maximised
    news_reactivity  = 0.95,  # hyper-reactive to news
    loss_aversion    = 4.5,
    t2_persona_tags  = ["risk_off", "tail_risk", "panic", "safe_haven", "volatility"],
    t1_persona_tags  = ["global_macro_pm", "vol_strategist", "risk_manager_pm"],
    base_weight      = 0.20,
)

STRUCTURAL_UNIVERSE = UniverseConfig(
    name        = "structural",
    description = "Institutional flows, commodity fundamentals, macro-driven positioning",
    momentum_bias    = 0.25,
    contrarian_bias  = 0.35,
    panic_threshold  = 0.82,
    herding_strength = 0.38,
    news_reactivity  = 0.45,
    loss_aversion    = 1.8,
    t2_persona_tags  = ["institutional", "commodity", "flow", "macro", "rates"],
    t1_persona_tags  = ["em_specialist", "commodity_pm", "rates_strategist", "credit_analyst"],
    base_weight      = 0.25,
)

ALL_UNIVERSES: list[UniverseConfig] = [
    MOMENTUM_UNIVERSE,
    VALUE_UNIVERSE,
    CRISIS_UNIVERSE,
    STRUCTURAL_UNIVERSE,
]


# ── Per-universe simulation result ─────────────────────────────────────────

@dataclass
class UniverseResult:
    """Output from one universe's full simulation."""
    universe_name:          str
    aggregate_sentiment:    float   # -1 to +1
    retail_sentiment:       float
    institutional_sentiment: float
    herd_index:             float
    panic_probability:      float
    narrative_momentum:     float
    contrarian_pressure:    float
    information_velocity:   float
    opinion_cluster_count:  int
    dominant_narrative:     str
    narrative_confidence:   float
    cascade_risk:           str     # "low" | "medium" | "high"
    bull_case:              str
    bear_case:              str
    sentiment_brief:        str     # prose brief for desk agents
    agent_count:            int
    run_id:                 str
    elapsed_sec:            float


# ── Ensemble aggregation ────────────────────────────────────────────────────

@dataclass
class SwarmReport:
    """
    The output of all four universes, ensembled into one coherent report.
    This replaces SimulationReport in v3.

    The desk agents receive the FULL SwarmReport, not just a single
    aggregate. This lets the Sentiment agent explicitly surface:
        "The Momentum universe is bullish 74% but the Value universe
         is only 28% bullish — extreme divergence, likely contrarian setup."
    """
    # Per-universe outputs
    universe_results:       dict[str, UniverseResult]   # keyed by universe name

    # Ensemble consensus
    ensemble_sentiment:     float   # weighted average
    ensemble_conviction:    float   # how tightly clustered the universes are (0=disagree, 1=agree)
    universe_divergence:    float   # std dev of universe sentiments — high = conflicting signals
    dominant_universe:      str     # which universe had strongest signal
    contrarian_setup:       bool    # momentum universe and value universe strongly disagree

    # Narrative synthesis (deterministic, no LLM)
    dominant_narrative:     str
    narrative_confidence:   float
    cascade_risk:           str
    bull_case:              str
    bear_case:              str
    tail_risk:              str

    # For the Sentiment desk agent's prompt
    swarm_brief:            str     # multi-paragraph summary for LLM context

    # Metrics dict (backward compat with v1 SimulationReport consumers)
    @property
    def metrics_dict(self) -> dict:
        return {
            "aggregate_sentiment":     self.ensemble_sentiment,
            "retail_sentiment":        self._universe_metric("retail_sentiment"),
            "institutional_sentiment": self._universe_metric("institutional_sentiment"),
            "herd_index":              self._universe_metric("herd_index"),
            "panic_probability":       self._universe_metric("panic_probability"),
            "narrative_momentum":      self._universe_metric("narrative_momentum"),
            "contrarian_pressure":     self._universe_metric("contrarian_pressure"),
            "information_velocity":    self._universe_metric("information_velocity"),
            "opinion_cluster_count":   int(self._universe_metric("opinion_cluster_count")),
            "agent_count":             sum(r.agent_count for r in self.universe_results.values()),
            "universe_divergence":     self.universe_divergence,
            "contrarian_setup":        self.contrarian_setup,
            "dominant_universe":       self.dominant_universe,
        }

    def _universe_metric(self, field: str) -> float:
        vals = [getattr(r, field) for r in self.universe_results.values()
                if hasattr(r, field)]
        return float(np.mean(vals)) if vals else 0.0
