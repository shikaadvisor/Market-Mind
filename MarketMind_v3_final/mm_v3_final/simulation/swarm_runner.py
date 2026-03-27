"""
simulation/swarm_runner.py  — FIXED (Bug12 + Bug13 + Bug14)
"""
from __future__ import annotations
import asyncio, logging, time, uuid
from dataclasses import dataclass
import numpy as np
from simulation.universes import ALL_UNIVERSES, UniverseConfig, UniverseResult, SwarmReport
from simulation.tiers.tier3_mesa import Tier3Simulation, UniverseBias, MarketEvent
logger = logging.getLogger("marketmind.swarm")

def _make_t2(cfg: UniverseConfig, n: int, key: str, mock: bool):
    """FIX-BUG12: each universe gets its own T2 instance with pre-filtered personas."""
    from simulation.tiers.tier2_async import Tier2Simulation
    from simulation.personas import TIER2_PERSONAS
    filtered = [p for p in TIER2_PERSONAS
                if any(tag in (p.name+" "+" ".join(getattr(p,"key_biases",[]))).lower()
                       for tag in cfg.t2_persona_tags)] or TIER2_PERSONAS[:5]
    sim = Tier2Simulation(n_agents=n, openai_api_key=key, mock=mock, verbose=False)
    sim._persona_pool = filtered  # read by patched run_async below
    return sim

def _make_t1(cfg: UniverseConfig, key: str, mock: bool):
    """FIX-BUG12: each universe gets its own T1 instance with pre-filtered personas."""
    from simulation.tiers.tier1_deep import Tier1Simulation
    from simulation.personas import TIER1_PERSONAS
    filtered = [p for p in TIER1_PERSONAS
                if any(tag in (p.name+" "+" ".join(getattr(p,"key_biases",[]))).lower()
                       for tag in cfg.t1_persona_tags)] or TIER1_PERSONAS[:5]
    sim = Tier1Simulation(openai_api_key=key, mock=mock, verbose=False)
    sim._persona_pool = filtered
    return sim

async def _run_single_universe(cfg, event, instrument, asset_class, key, n_agents, n_steps, n_tier2, mock):
    t0 = time.perf_counter(); run_id = str(uuid.uuid4())[:8]
    try:
        bias = UniverseBias.from_universe_config(cfg)          # FIX-BUG13
        t3   = Tier3Simulation(n_agents=n_agents, n_steps=n_steps,
                               universe_bias=bias, universe_name=cfg.name, verbose=True)
        t3._initialise_population(asset_class)
        t3r  = t3.run(event)

        t2s  = _make_t2(cfg, n_tier2, key, mock)               # FIX-BUG12
        try:    t2r = await t2s.run_async(event, t3r, key)
        except Exception as e: logger.warning("[%s] T2 fail: %s", cfg.name, e); t2r = None

        t1s  = _make_t1(cfg, key, mock)                        # FIX-BUG12
        try:    t1r = await t1s.run_async(event, instrument, asset_class, t3r, t2r)
        except Exception as e: logger.warning("[%s] T1 fail: %s", cfg.name, e); t1r = None

        return _build_result(cfg, t3r, t2r, t1r, run_id, time.perf_counter()-t0)
    except Exception as exc:
        logger.exception("[%s] universe failed: %s", cfg.name, exc)
        return _neutral(cfg.name, run_id, time.perf_counter()-t0)  # FIX-BUG14: use cfg.name

def _build_result(cfg, t3, t2, t1, run_id, elapsed):
    t3s = float(getattr(t3,"aggregate_sentiment",0)) if t3 else 0
    t2s = float(getattr(t2,"avg_sentiment",0)) if t2 else 0
    t1s = float(getattr(t1,"avg_sentiment",0)) if t1 else 0
    agg = 0.45*t3s + 0.35*t2s + 0.20*t1s
    g   = lambda k,d=0.0: float(getattr(t3,k,d)) if t3 else d
    panic,herd = g("panic_probability",0.1), g("herd_index",0.5)
    cr = "high" if panic>0.40 and herd>0.65 else "medium" if panic>0.20 or herd>0.55 else "low"
    brief = (f"[{cfg.name.upper()}] sent={agg:+.3f} herd={herd:.0%} panic={panic:.0%} cascade={cr.upper()}\n"
             f"Profile: {cfg.description}\n")
    return UniverseResult(
        universe_name=cfg.name, aggregate_sentiment=round(agg,4),
        retail_sentiment=t3s, institutional_sentiment=t1s,
        herd_index=herd, panic_probability=panic,
        narrative_momentum=g("narrative_momentum"), contrarian_pressure=g("contrarian_pressure",0.1),
        information_velocity=g("information_velocity",0.3),
        opinion_cluster_count=int(g("opinion_cluster_count",2)),
        dominant_narrative=g("cascade_risk","unknown") if t3 else "unknown",
        narrative_confidence=min(abs(agg)+0.1,1.0), cascade_risk=cr,
        bull_case="", bear_case="", sentiment_brief=brief,
        agent_count=((t3.n_agents if t3 else 0)+(getattr(t2,"n_agents",0) if t2 else 0)+(getattr(t1,"n_agents",0) if t1 else 0)),
        run_id=run_id, elapsed_sec=round(elapsed,2),
    )

def _neutral(name, run_id, elapsed):
    return UniverseResult(
        universe_name=name, aggregate_sentiment=0.0,             # FIX-BUG14: name not "error_"+name
        retail_sentiment=0.0, institutional_sentiment=0.0,
        herd_index=0.5, panic_probability=0.1, narrative_momentum=0.0,
        contrarian_pressure=0.1, information_velocity=0.3, opinion_cluster_count=1,
        dominant_narrative="error", narrative_confidence=0.0, cascade_risk="low",
        bull_case="", bear_case="", sentiment_brief=f"[{name.upper()}] Failed.",
        agent_count=0, run_id=run_id, elapsed_sec=elapsed,
    )

def _ensemble(results: list[UniverseResult], universes: list[UniverseConfig]) -> SwarmReport:
    by_name = {r.universe_name: r for r in results}
    w_sum = wt_sent = 0.0; sents = []; dom = ""; dom_s = 0.0
    for u in universes:
        r = by_name.get(u.name)
        if r is None: continue
        w = u.effective_weight; w_sum += w; wt_sent += w * r.aggregate_sentiment
        sents.append(r.aggregate_sentiment)
        if abs(r.aggregate_sentiment) > abs(dom_s): dom_s = r.aggregate_sentiment; dom = u.name

    if w_sum < 1e-9:  # FIX-BUG14: guard division
        logger.error("All universes failed — neutral ensemble returned")
        return SwarmReport(
            universe_results={}, ensemble_sentiment=0.0, ensemble_conviction=0.0,
            universe_divergence=0.0, dominant_universe="none", contrarian_setup=False,
            dominant_narrative="error", narrative_confidence=0.0, cascade_risk="low",
            bull_case="", bear_case="", tail_risk="All simulations failed.",
            swarm_brief="[ERROR] All four universes failed.",
        )

    ens  = wt_sent / w_sum
    div  = float(np.std(sents)) if len(sents) > 1 else 0.0
    conv = max(0.0, 1.0 - div * 2.0)

    mr, vr = by_name.get("momentum"), by_name.get("value")
    contrarian = (mr and vr and
                  abs(mr.aggregate_sentiment - vr.aggregate_sentiment) > 0.40
                  and mr.aggregate_sentiment != 0.0)  # 0.0 = failed

    cr_r  = by_name.get("crisis")
    cr_risk = ("high" if cr_r and cr_r.cascade_risk == "high" else
               "medium" if cr_r and cr_r.cascade_risk == "medium" else "low")

    brief = "SWARM REPORT\n" + "".join(r.sentiment_brief for r in results)
    brief += (f"\nENSEMBLE\nWeighted sentiment: {ens:+.4f}\n"
              f"Divergence: {div:.4f} ({'HIGH' if div>0.30 else 'mod' if div>0.15 else 'low'})\n"
              f"Contrarian setup: {'YES' if contrarian else 'no'}\n"
              f"Dominant: {dom}  Cascade: {cr_risk.upper()}\n")

    best = max(results, key=lambda r: r.narrative_confidence)
    return SwarmReport(
        universe_results=by_name,
        ensemble_sentiment=round(ens,4), ensemble_conviction=round(conv,4),
        universe_divergence=round(div,4), dominant_universe=dom,
        contrarian_setup=bool(contrarian),
        dominant_narrative=best.dominant_narrative,
        narrative_confidence=best.narrative_confidence,
        cascade_risk=cr_risk, bull_case="", bear_case="",
        tail_risk=("Crisis universe alarmed." if cr_r and cr_r.cascade_risk=="high" else
                   "Divergence high — momentum crowd may be wrong." if div>0.30 else "Moderate."),
        swarm_brief=brief,
    )


class SwarmRunner:
    def __init__(self, openai_api_key, n_agents=2_500, n_steps=50, n_tier2=50,
                 universes=None, mock=False, verbose=True):
        self.key=openai_api_key; self.n_agents=n_agents; self.n_steps=n_steps
        self.n_tier2=n_tier2; self.universes=universes or ALL_UNIVERSES
        self.mock=mock; self.verbose=verbose
        if verbose:
            print(f"  SwarmRunner: {len(self.universes)} universes × {n_agents:,} T3 agents")
            for u in self.universes:
                b=UniverseBias.from_universe_config(u)
                print(f"    [{u.name:12s}] w={u.effective_weight:.2f}  "
                      f"mom×{b.momentum_sensitivity:.2f} contrarian×{b.contrarian_bias:.2f}")

    async def run_async(self, event, instrument, asset_class) -> SwarmReport:
        results = await asyncio.gather(
            *[_run_single_universe(u, event, instrument, asset_class,
                                   self.key, self.n_agents, self.n_steps, self.n_tier2, self.mock)
              for u in self.universes],
            return_exceptions=True,
        )
        clean = [_neutral(self.universes[i].name,"crash",0.0) if isinstance(r,Exception) else r
                 for i,r in enumerate(results)]
        report = _ensemble(clean, self.universes)
        if self.verbose:
            print(f"  Swarm: sent={report.ensemble_sentiment:+.4f}  "
                  f"div={report.universe_divergence:.4f}  "
                  f"contrarian={'YES' if report.contrarian_setup else 'no'}")
        return report

    def apply_feedback(self, weights: dict[str,float]) -> None:
        for u in self.universes:
            if u.name in weights:
                u.calibrated_weight = weights[u.name]
