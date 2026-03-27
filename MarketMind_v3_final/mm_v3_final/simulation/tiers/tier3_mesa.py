"""
simulation/tiers/tier3_mesa.py  — v3 PATCHED
=============================================
FIX-BUG13: _initialise_population() now accepts UniverseBias.
           Each universe produces a genuinely different crowd by scaling
           parameter arrays after archetype drawing — no monkey-patching.

FIX-RESET: reset_stances() added for soft reset between instrument runs.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple, Optional

import numpy as np


class Stance(IntEnum):
    STRONG_BEAR  = -2
    BEAR         = -1
    NEUTRAL      =  0
    BULL         =  1
    STRONG_BULL  =  2


PERSONA_ARCHETYPES = {
    "fomo_retail":        (0.85, 0.90, 3.2, 0.45, 0.55, 0.05),
    "momentum_chaser":    (0.90, 0.70, 2.8, 0.50, 0.60, 0.08),
    "passive_investor":   (0.20, 0.30, 1.8, 0.70, 0.80, 0.10),
    "news_trader":        (0.50, 0.95, 2.5, 0.55, 0.65, 0.07),
    "value_seeker":       (0.15, 0.40, 1.5, 0.75, 0.85, 0.40),
    "stop_loss_victim":   (0.70, 0.60, 4.0, 0.40, 0.40, 0.03),
    "options_gambler":    (0.80, 0.85, 3.5, 0.35, 0.45, 0.06),
    "systematic_retail":  (0.60, 0.50, 2.2, 0.60, 0.70, 0.12),
    "spec_long_fund":     (0.75, 0.80, 2.5, 0.55, 0.65, 0.08),
    "producer_hedger":    (0.10, 0.50, 1.2, 0.80, 0.90, 0.50),
    "commercial_buyer":   (0.15, 0.60, 1.3, 0.75, 0.88, 0.45),
    "macro_tourist":      (0.65, 0.85, 2.8, 0.50, 0.60, 0.10),
    "inventory_trader":   (0.40, 0.95, 2.2, 0.60, 0.70, 0.20),
    "spread_arb":         (0.20, 0.60, 1.5, 0.80, 0.90, 0.55),
    "carry_trader":       (0.30, 0.70, 2.0, 0.65, 0.75, 0.25),
    "fx_trend_follower":  (0.80, 0.65, 2.3, 0.50, 0.60, 0.10),
    "fx_news_trader":     (0.45, 0.98, 2.8, 0.50, 0.65, 0.05),
    "fx_retail_gambler":  (0.75, 0.85, 3.5, 0.40, 0.50, 0.05),
    "duration_manager":   (0.20, 0.85, 1.5, 0.80, 0.92, 0.35),
    "fed_watcher":        (0.35, 0.98, 1.8, 0.70, 0.85, 0.20),
    "credit_spread_pm":   (0.25, 0.75, 1.6, 0.75, 0.88, 0.30),
    "crypto_hodler":      (0.50, 0.70, 3.5, 0.40, 0.55, 0.30),
    "crypto_degen":       (0.95, 0.92, 4.5, 0.30, 0.40, 0.03),
    "crypto_whale":       (0.25, 0.55, 2.0, 0.75, 0.80, 0.45),
}

ASSET_CLASS_POPULATIONS = {
    "equity_index": {
        "fomo_retail": 0.20, "momentum_chaser": 0.18, "passive_investor": 0.12,
        "news_trader": 0.15, "value_seeker": 0.10, "stop_loss_victim": 0.10,
        "options_gambler": 0.08, "systematic_retail": 0.07,
    },
    "commodity_energy": {
        "spec_long_fund": 0.25, "producer_hedger": 0.15, "commercial_buyer": 0.15,
        "macro_tourist": 0.20, "inventory_trader": 0.15, "spread_arb": 0.10,
    },
    "commodity_metal": {
        "spec_long_fund": 0.30, "macro_tourist": 0.25, "value_seeker": 0.20,
        "producer_hedger": 0.15, "spread_arb": 0.10,
    },
    "commodity_agri": {
        "producer_hedger": 0.25, "commercial_buyer": 0.25, "spec_long_fund": 0.20,
        "inventory_trader": 0.15, "macro_tourist": 0.15,
    },
    "fx_major": {
        "carry_trader": 0.25, "fx_trend_follower": 0.25, "fx_news_trader": 0.20,
        "fx_retail_gambler": 0.15, "macro_tourist": 0.15,
    },
    "fx_em": {
        "carry_trader": 0.30, "fx_retail_gambler": 0.25, "macro_tourist": 0.25,
        "fx_news_trader": 0.20,
    },
    "fixed_income": {
        "duration_manager": 0.35, "fed_watcher": 0.30, "credit_spread_pm": 0.20,
        "macro_tourist": 0.15,
    },
    "crypto": {
        "crypto_degen": 0.35, "crypto_hodler": 0.35, "fomo_retail": 0.20,
        "crypto_whale": 0.10,
    },
}


def get_persona_population(asset_class: str) -> dict:
    for key, pop in ASSET_CLASS_POPULATIONS.items():
        if asset_class.startswith(key):
            return pop
    return ASSET_CLASS_POPULATIONS["equity_index"]


@dataclass
class MarketEvent:
    description:    str
    price_shock:    float
    news_sentiment: float
    uncertainty:    float
    is_systemic:    bool  = False
    sector_impact:  str   = "all"

    def __post_init__(self):
        self.price_shock    = max(-1.0, min(1.0, self.price_shock))
        self.news_sentiment = max(-1.0, min(1.0, self.news_sentiment))
        self.uncertainty    = max(0.0,  min(1.0, self.uncertainty))


class StepSnapshot(NamedTuple):
    step:                 int
    mean_sentiment:       float
    sentiment_std:        float
    bull_pct:             float
    bear_pct:             float
    neutral_pct:          float
    herd_index:           float
    cascade_detected:     bool
    cascade_magnitude:    float
    panic_agents_pct:     float
    information_velocity: float


@dataclass
class Tier3Result:
    n_agents:              int
    n_steps:               int
    snapshots:             list
    final_snapshot:        StepSnapshot
    aggregate_sentiment:   float
    sentiment_std:         float
    bull_pct:              float
    bear_pct:              float
    neutral_pct:           float
    herd_index:            float
    panic_probability:     float
    cascade_risk:          str
    cascade_magnitude:     float
    information_velocity:  float
    narrative_momentum:    float
    contrarian_pressure:   float
    opinion_cluster_count: int
    elapsed_sec:           float
    universe_name:         str = ""

    def to_tier_snapshot_dict(self) -> dict:
        return {
            "tier": 3, "agent_count": self.n_agents,
            "bullish_pct": round(self.bull_pct, 4),
            "bearish_pct": round(self.bear_pct, 4),
            "neutral_pct": round(self.neutral_pct, 4),
            "avg_conviction": round(abs(self.aggregate_sentiment), 4),
            "notable_behaviors": [],
            "herd_index": self.herd_index,
            "panic_pct":  self.panic_probability,
        }


@dataclass
class UniverseBias:
    """
    FIX-BUG13: Per-column multipliers applied to T3 parameter arrays.
    Values >1 amplify, <1 dampen. Applied after archetype noise addition.
    """
    momentum_sensitivity: float = 1.0
    news_reactivity:      float = 1.0
    loss_aversion:        float = 1.0
    herding_threshold:    float = 1.0
    panic_threshold:      float = 1.0
    contrarian_bias:      float = 1.0

    @classmethod
    def from_universe_config(cls, cfg) -> "UniverseBias":
        """Map 0-1 UniverseConfig values to meaningful multipliers."""
        def _m(v: float, lo: float = 0.4, hi: float = 2.0) -> float:
            return round(lo + v * (hi - lo), 3)
        return cls(
            momentum_sensitivity = _m(cfg.momentum_bias),
            news_reactivity      = _m(cfg.news_reactivity),
            loss_aversion        = _m(min(cfg.loss_aversion / 5.0, 1.0)),
            herding_threshold    = _m(1.0 - cfg.herding_strength),
            panic_threshold      = _m(cfg.panic_threshold),
            contrarian_bias      = _m(cfg.contrarian_bias),
        )

    @classmethod
    def neutral(cls) -> "UniverseBias":
        return cls()


class Tier3Simulation:

    def __init__(
        self,
        n_agents:      int             = 10_000,
        n_steps:       int             = 50,
        n_neighbours:  int             = 12,
        seed:          int             = 42,
        verbose:       bool            = True,
        universe_bias: Optional[UniverseBias] = None,
        universe_name: str             = "",
    ):
        self.n_agents     = n_agents
        self.n_steps      = n_steps
        self.n_neighbours = n_neighbours
        self.seed         = seed
        self.verbose      = verbose
        self.universe_bias = universe_bias or UniverseBias.neutral()
        self.universe_name = universe_name

        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        self._snapshots: list[StepSnapshot] = []
        self._initialised = False

    def _initialise_population(self, asset_class: str = "equity_index") -> None:
        N   = self.n_agents
        pop = get_persona_population(asset_class)
        names   = list(pop.keys())
        total   = sum(pop.values())
        weights = [v / total for v in pop.values()]

        idx        = self.rng.choice(len(names), size=N, p=weights)
        archetypes = np.array([list(PERSONA_ARCHETYPES[names[i]]) for i in idx])
        noise      = self.rng.normal(0, 0.08, size=archetypes.shape)
        params     = np.clip(archetypes + noise, 0.01, 0.99)

        # FIX-BUG13: apply per-column universe multipliers
        b = self.universe_bias
        params[:, 0] = np.clip(params[:, 0] * b.momentum_sensitivity, 0.01, 0.99)
        params[:, 1] = np.clip(params[:, 1] * b.news_reactivity,      0.01, 0.99)
        params[:, 2] = np.clip(archetypes[:, 2] * b.loss_aversion,    1.0,  6.0)
        params[:, 3] = np.clip(params[:, 3] * b.herding_threshold,    0.01, 0.99)
        params[:, 4] = np.clip(params[:, 4] * b.panic_threshold,      0.01, 0.99)
        params[:, 5] = np.clip(params[:, 5] * b.contrarian_bias,      0.01, 0.99)

        self.momentum_sensitivity = params[:, 0]
        self.news_reactivity      = params[:, 1]
        self.loss_aversion        = params[:, 2]
        self.herding_threshold    = params[:, 3]
        self.panic_threshold      = params[:, 4]
        self.contrarian_bias      = params[:, 5]
        self.persona_idx          = idx

        self.sentiment = self.rng.normal(0.0, 0.15, size=N)
        self.sentiment = np.clip(self.sentiment, -1.0, 1.0)
        self._build_social_network()
        self._initialised = True

        if self.verbose:
            b = self.universe_bias
            print(f"  [T3/{self.universe_name or 'base'}] {N:,} agents | "
                  f"mom×{b.momentum_sensitivity:.2f} panic_thr×{b.panic_threshold:.2f} "
                  f"contrarian×{b.contrarian_bias:.2f}")

    def reset_stances(self) -> None:
        """Soft reset — zero sentiment, keep social graph (fast)."""
        if self._initialised:
            self.sentiment = self.rng.normal(0.0, 0.12, size=self.n_agents)
            self.sentiment = np.clip(self.sentiment, -1.0, 1.0)

    def _build_social_network(self) -> None:
        N, K = self.n_agents, self.n_neighbours
        neighbours = np.zeros((N, K), dtype=np.int32)
        for i in range(N):
            for j in range(K):
                neighbours[i, j] = (i + j + 1) % N
        mask = self.rng.random(size=(N, K)) < 0.15
        neighbours[mask] = self.rng.integers(0, N, size=(N, K))[mask]
        self.neighbours = neighbours

    def _step(self, step, price_signal, news_signal, uncertainty) -> StepSnapshot:
        prev = self.sentiment.copy()

        nbr_mean      = self.sentiment[self.neighbours].mean(axis=1)
        herding_active = (np.abs(nbr_mean) > self.herding_threshold).astype(float)
        herding_pull   = herding_active * nbr_mean * 0.35
        news_pull      = self.news_reactivity * news_signal * 0.25
        momentum_pull  = self.momentum_sensitivity * price_signal * 0.20
        loss_effect    = self.loss_aversion * np.maximum(0, -price_signal) * 0.15
        contrarian     = -self.contrarian_bias * nbr_mean * 0.20

        clarity = 1.0 - uncertainty * 0.5
        noise   = self.rng.normal(0, 0.03 * (1 + uncertainty), size=self.n_agents)
        delta   = clarity * (herding_pull + news_pull + momentum_pull - loss_effect + contrarian) + noise
        self.sentiment = np.clip(self.sentiment + 0.18 * delta, -1.0, 1.0)

        ms  = float(self.sentiment.mean())
        std = float(self.sentiment.std())
        bp  = float((self.sentiment >  0.15).mean())
        brp = float((self.sentiment < -0.15).mean())
        hi  = float(((np.abs(self.sentiment) > 0.30) & (np.sign(self.sentiment) == np.sign(ms))).mean())
        dm  = np.abs(self.sentiment - prev)
        cp  = float((dm > 0.12).mean())
        iv  = min(1.0, float(dm.mean()) * 8.0)

        return StepSnapshot(
            step=step, mean_sentiment=ms, sentiment_std=std,
            bull_pct=bp, bear_pct=brp, neutral_pct=1-bp-brp,
            herd_index=hi, cascade_detected=(cp > 0.25),
            cascade_magnitude=min(1.0, cp * 2.5),
            panic_agents_pct=float((self.sentiment < -0.70).mean()),
            information_velocity=iv,
        )

    def _decay_signals(self, step, p0, n0, u0):
        ps = p0 * math.exp(-0.09 * step)
        ns = n0 * math.exp(-0.05 * step)
        u  = (u0 * (1 + 0.3 * (5 - step) / 5) if step < 5
              else u0 * math.exp(-0.04 * (step - 5)))
        return ps, ns, max(0.05, u)

    def run(self, event: MarketEvent) -> Tier3Result:
        if not self._initialised:
            self._initialise_population()
        t0 = time.perf_counter()
        self._snapshots = []
        for step in range(self.n_steps):
            ps, ns, u = self._decay_signals(step, event.price_shock, event.news_sentiment, event.uncertainty)
            self._snapshots.append(self._step(step, ps, ns, u))
        final   = self._snapshots[-1]
        sentv   = [s.mean_sentiment for s in self._snapshots]
        nm      = float(np.mean(np.diff(sentv[-10:]))) if len(sentv) > 10 else 0.0
        cp_arr  = float((self.contrarian_bias > 0.30).mean())
        cascade_steps = [s for s in self._snapshots if s.cascade_detected]
        cr = "high" if len(cascade_steps) > 5 else "medium" if len(cascade_steps) > 2 else "low"
        cm = float(np.max([s.cascade_magnitude for s in self._snapshots]))
        clusters = max(1, int(1 + abs(final.mean_sentiment) * 3 + final.sentiment_std * 2))
        elapsed = time.perf_counter() - t0
        if self.verbose:
            print(f"  [T3/{self.universe_name or 'base'}] "
                  f"sent={final.mean_sentiment:+.4f} herd={final.herd_index:.0%} "
                  f"cascade={cr} ({elapsed:.2f}s)")
        return Tier3Result(
            n_agents=self.n_agents, n_steps=self.n_steps,
            snapshots=self._snapshots, final_snapshot=final,
            aggregate_sentiment=round(final.mean_sentiment, 4),
            sentiment_std=round(final.sentiment_std, 4),
            bull_pct=round(final.bull_pct, 4), bear_pct=round(final.bear_pct, 4),
            neutral_pct=round(final.neutral_pct, 4),
            herd_index=round(final.herd_index, 4),
            panic_probability=round(final.panic_agents_pct, 4),
            cascade_risk=cr, cascade_magnitude=round(cm, 4),
            information_velocity=round(final.information_velocity, 4),
            narrative_momentum=round(nm, 4), contrarian_pressure=round(cp_arr, 4),
            opinion_cluster_count=clusters, elapsed_sec=round(elapsed, 3),
            universe_name=self.universe_name,
        )
