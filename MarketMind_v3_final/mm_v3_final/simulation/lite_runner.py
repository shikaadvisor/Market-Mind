"""
simulation/lite_runner.py
==========================
Lite mode swarm runner — T3 (ABM) only, no T2/T1 LLM calls.

Cost:     ~$0.04/run (coordinator only) vs ~$0.26 full mode
Latency:  ~3-5s vs ~25-35s full mode
Use case: intraday polling (every 30 min during session)

The T3 simulation still runs all 4 universes with real UniverseBias.
What's missing is the informed-crowd (T2) and deep-thinker (T1) layer.
This means the sentiment output is less nuanced but:
  - Still 10,000 agents with different psychological profiles
  - Still universe divergence and contrarian setup detection
  - Still FRED + Finnhub data enrichment
  - Still devil's advocate and full desk graph

Suitable for intraday because:
  - Intraday signals are driven by price action, not informed opinion
  - 30-min polling means you run 10-15 times per session — T2/T1 would cost $3-4/session
  - T3 crowd dynamics (panic cascades, herding, momentum) are most relevant intraday
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

import numpy as np

from simulation.universes import ALL_UNIVERSES, UniverseConfig, UniverseResult, SwarmReport
from simulation.tiers.tier3_mesa import Tier3Simulation, UniverseBias

logger = logging.getLogger("marketmind.lite")


async def _run_universe_lite(
    cfg:    UniverseConfig,
    event,
    asset_class: str,
    n_agents: int,
    n_steps:  int,
) -> UniverseResult:
    """T3-only universe run. No LLM calls."""
    t0     = time.perf_counter()
    run_id = str(uuid.uuid4())[:8]
    try:
        bias = UniverseBias.from_universe_config(cfg)
        t3   = Tier3Simulation(
            n_agents      = n_agents,
            n_steps       = n_steps,
            universe_bias = bias,
            universe_name = cfg.name,
            verbose       = False,
        )
        t3._initialise_population(asset_class)
        t3r = t3.run(event)

        sent = float(getattr(t3r, "aggregate_sentiment", 0.0))
        herd = float(getattr(t3r, "herd_index", 0.5))
        panic = float(getattr(t3r, "panic_probability", 0.1))
        cascade = "high" if panic > 0.40 and herd > 0.65 else "medium" if panic > 0.20 else "low"
        brief = (f"[{cfg.name.upper()}·lite] sent={sent:+.3f} herd={herd:.0%} "
                 f"panic={panic:.0%} cascade={cascade.upper()}\n")

        return UniverseResult(
            universe_name=cfg.name, aggregate_sentiment=round(sent, 4),
            retail_sentiment=sent, institutional_sentiment=0.0,
            herd_index=herd, panic_probability=panic,
            narrative_momentum=float(getattr(t3r, "narrative_momentum", 0.0)),
            contrarian_pressure=float(getattr(t3r, "contrarian_pressure", 0.1)),
            information_velocity=float(getattr(t3r, "information_velocity", 0.3)),
            opinion_cluster_count=int(getattr(t3r, "opinion_cluster_count", 2)),
            dominant_narrative=getattr(t3r, "cascade_risk", "unknown"),
            narrative_confidence=min(abs(sent) + 0.1, 0.75),  # capped — no T1 validation
            cascade_risk=cascade, bull_case="", bear_case="",
            sentiment_brief=brief, agent_count=t3r.n_agents,
            run_id=run_id, elapsed_sec=round(time.perf_counter() - t0, 2),
        )
    except Exception as exc:
        logger.exception("[lite/%s] failed: %s", cfg.name, exc)
        return UniverseResult(
            universe_name=cfg.name, aggregate_sentiment=0.0,
            retail_sentiment=0.0, institutional_sentiment=0.0,
            herd_index=0.5, panic_probability=0.1, narrative_momentum=0.0,
            contrarian_pressure=0.1, information_velocity=0.3,
            opinion_cluster_count=1, dominant_narrative="error",
            narrative_confidence=0.0, cascade_risk="low",
            bull_case="", bear_case="",
            sentiment_brief=f"[{cfg.name.upper()}·lite] FAILED",
            agent_count=0, run_id=uuid.uuid4().hex[:8],
            elapsed_sec=time.perf_counter() - t0,
        )


class LiteSwarmRunner:
    """
    T3-only swarm runner for intraday / high-frequency modes.
    API identical to SwarmRunner — drop-in replacement.
    """

    def __init__(
        self,
        n_agents:  int = 2_500,   # per universe
        n_steps:   int = 30,      # fewer steps — intraday dynamics settle faster
        universes: list = None,
        verbose:   bool = False,
    ):
        self.n_agents  = n_agents
        self.n_steps   = n_steps
        self.universes = universes or ALL_UNIVERSES
        self.verbose   = verbose
        # Pre-build T3 sims — reuse across polls for speed
        self._t3_sims: dict = {}

    async def run_async(self, event, instrument: str, asset_class: str) -> SwarmReport:
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *[_run_universe_lite(u, event, asset_class, self.n_agents, self.n_steps)
              for u in self.universes],
            return_exceptions=True,
        )
        clean = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                clean.append(UniverseResult(
                    universe_name=self.universes[i].name, aggregate_sentiment=0.0,
                    retail_sentiment=0.0, institutional_sentiment=0.0,
                    herd_index=0.5, panic_probability=0.1, narrative_momentum=0.0,
                    contrarian_pressure=0.1, information_velocity=0.3,
                    opinion_cluster_count=1, dominant_narrative="error",
                    narrative_confidence=0.0, cascade_risk="low",
                    bull_case="", bear_case="",
                    sentiment_brief=f"[{self.universes[i].name.upper()}·lite] CRASHED",
                    agent_count=0, run_id="crash", elapsed_sec=0.0,
                ))
            else:
                clean.append(r)

        # Ensemble (same logic as full SwarmRunner)
        by_name     = {r.universe_name: r for r in clean}
        w_sum = wt  = 0.0
        sents       = []
        dom, dom_s  = "", 0.0
        for u in self.universes:
            r = by_name.get(u.name)
            if not r: continue
            w = u.effective_weight; w_sum += w; wt += w * r.aggregate_sentiment
            sents.append(r.aggregate_sentiment)
            if abs(r.aggregate_sentiment) > abs(dom_s): dom_s = r.aggregate_sentiment; dom = u.name

        ens  = wt / max(w_sum, 1e-9)
        div  = float(np.std(sents)) if len(sents) > 1 else 0.0
        conv = max(0.0, 1.0 - div * 2.0)
        mr, vr = by_name.get("momentum"), by_name.get("value")
        contrarian = (mr and vr and abs(mr.aggregate_sentiment - vr.aggregate_sentiment) > 0.40
                      and mr.aggregate_sentiment != 0.0)
        crs = by_name.get("crisis")
        cr_risk = "high" if crs and crs.cascade_risk == "high" else "medium" if crs and crs.cascade_risk == "medium" else "low"

        brief = "[LITE MODE — T3 only]\n" + "".join(r.sentiment_brief for r in clean)
        brief += f"\nEnsemble: sent={ens:+.4f} div={div:.4f} contrarian={'YES' if contrarian else 'no'}\n"

        best = max(clean, key=lambda r: r.narrative_confidence)
        report = SwarmReport(
            universe_results=by_name,
            ensemble_sentiment=round(ens, 4), ensemble_conviction=round(conv * 0.85, 4),
            # Cap conviction at 0.85 in lite mode — T1/T2 missing means less certainty
            universe_divergence=round(div, 4), dominant_universe=dom,
            contrarian_setup=bool(contrarian),
            dominant_narrative=best.dominant_narrative,
            narrative_confidence=best.narrative_confidence * 0.80,  # lite penalty
            cascade_risk=cr_risk, bull_case="", bear_case="",
            tail_risk="Lite mode — reduced certainty, no informed-crowd validation.",
            swarm_brief=brief,
        )

        if self.verbose:
            print(f"  Lite swarm [{instrument}]: sent={ens:+.4f} div={div:.3f} "
                  f"t={time.perf_counter()-t0:.1f}s")
        return report

    def apply_feedback(self, weights: dict) -> None:
        for u in self.universes:
            if u.name in weights:
                u.calibrated_weight = weights[u.name]
