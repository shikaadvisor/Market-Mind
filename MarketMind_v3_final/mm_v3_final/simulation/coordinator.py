"""
simulation/coordinator.py
==========================
The simulation coordinator — aggregates Tier 1, 2, 3 outputs into a
fully-typed SimulationReport that the trading desk can consume.

This is the "simulation head trader" — it synthesises what 10,000+
agents produced into a structured, opinionated handoff document.

Responsibilities:
- Weighted aggregation of sentiment across tiers
- Cascade risk assessment
- Narrative inference from crowd dynamics
- Bull/bear/risk synthesis
- Writing the sentiment_interpreter_brief (pre-formatted for the desk agent)
- Producing a valid SimulationReport (Pydantic-validated)

The coordinator itself does NOT call an LLM. It's deterministic logic —
fast, cheap, always consistent. The richness comes from the tier outputs
feeding into it, not from the coordinator itself adding LLM reasoning.
This is a deliberate architectural choice: LLM calls happen inside the
tiers, not in the aggregation layer.
"""

from __future__ import annotations

import uuid
from datetime import datetime, date, timezone
from typing import TYPE_CHECKING

from simulation.tiers.tier3_mesa import Tier3Result, MarketEvent
from simulation.tiers.tier2_async import Tier2Result
from simulation.tiers.tier1_deep import Tier1Result

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Aggregation weights
# ---------------------------------------------------------------------------

# How much each tier contributes to the AGGREGATE sentiment metric.
# Tier 3 carries the most weight — it's the actual crowd.
# Tier 1 has the smallest weight here but the highest desk weight
# (handled separately in the trading desk coordinator).
TIER_SENTIMENT_WEIGHTS = {
    3: 0.45,   # statistical mass
    2: 0.35,   # informed crowd
    1: 0.20,   # deep thinkers
}


# ---------------------------------------------------------------------------
# Narrative inference
# ---------------------------------------------------------------------------

def _infer_narrative(
    agg_sentiment:    float,
    herd_index:       float,
    panic_prob:       float,
    narrative_mom:    float,
    contrarian_pct:   float,
    tier1_divergence: float,    # tier1_sentiment - agg_sentiment
    event:            MarketEvent,
) -> tuple[str, float]:
    """
    Infer the dominant narrative and confidence from simulation metrics.
    Returns (narrative_name, confidence).
    """

    # Panic condition: fear + high herding + high panic probability
    if agg_sentiment < -0.50 and herd_index > 0.65 and panic_prob > 0.40:
        return "event_driven_panic", min(0.95, abs(agg_sentiment) * 0.9)

    # Euphoria: strong positive + high herding
    if agg_sentiment > 0.50 and herd_index > 0.65:
        return "event_driven_euphoria", min(0.92, agg_sentiment * 0.9)

    # Accumulation: crowd bearish but smart money (Tier 1) diverging bullish
    if agg_sentiment < -0.15 and tier1_divergence > 0.25 and contrarian_pct > 0.30:
        return "accumulation", min(0.85, contrarian_pct * 0.9 + 0.30)

    # Distribution: crowd bullish but smart money diverging bearish
    if agg_sentiment > 0.15 and tier1_divergence < -0.25 and contrarian_pct > 0.30:
        return "distribution", min(0.82, contrarian_pct * 0.9 + 0.25)

    # Macro risk-off: global systemic event driving broad selling
    if event.is_systemic and agg_sentiment < -0.25 and narrative_mom < -0.20:
        return "macro_risk_off", min(0.80, abs(agg_sentiment) * 0.75)

    # Macro risk-on: systemic positive
    if event.is_systemic and agg_sentiment > 0.25 and narrative_mom > 0.20:
        return "macro_risk_on", min(0.78, agg_sentiment * 0.75)

    # Technical momentum signals
    if narrative_mom > 0.35 and agg_sentiment > 0.10:
        return "technical_breakout", min(0.72, narrative_mom * 0.8)

    if narrative_mom < -0.35 and agg_sentiment < -0.10:
        return "technical_breakdown", min(0.72, abs(narrative_mom) * 0.8)

    # Bearish/bullish fundamentals (moderate, non-panicky)
    if agg_sentiment < -0.15:
        return "bearish_fundamentals", min(0.65, abs(agg_sentiment) * 0.70)
    if agg_sentiment > 0.15:
        return "bullish_fundamentals", min(0.65, agg_sentiment * 0.70)

    # Default: no clear narrative
    return "confusion", 0.35


# ---------------------------------------------------------------------------
# Cascade risk level
# ---------------------------------------------------------------------------

def _assess_cascade_risk(tier3: Tier3Result, panic_prob: float) -> str:
    n_cascades = len(tier3.cascade_events)
    max_mag    = max((c["magnitude"] for c in tier3.cascade_events), default=0.0)
    herd       = tier3.final_snapshot.herd_index

    if max_mag > 0.70 or (panic_prob > 0.50 and herd > 0.70):
        return "extreme"
    if max_mag > 0.45 or (n_cascades >= 2 and herd > 0.60):
        return "high"
    if max_mag > 0.25 or (n_cascades >= 1 and panic_prob > 0.25):
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Qualitative synthesis
# ---------------------------------------------------------------------------

def _build_bull_case(
    tier1:    Tier1Result,
    metrics:  dict,
    event:    MarketEvent,
) -> str:
    """Extract the strongest bull case from Tier 1 responses."""
    # Find the most bullish Tier 1 agent with highest conviction
    bullish_agents = [r for r in tier1.responses
                      if r.stance == "bullish" and not r.is_fallback]

    if bullish_agents:
        best = max(bullish_agents, key=lambda r: r.conviction)
        return (
            f"{best.persona_name.replace('_', ' ').title()} "
            f"(conviction {best.conviction:.0%}): {best.key_bull_factor}. "
            f"Leading indicator: {best.leading_indicator}. "
            f"Target: {best.three_month_target}."
        )

    # Fallback: synthesise from metrics
    cp = metrics["contrarian_pressure"]
    is_ = metrics["institutional_sentiment"]
    return (
        f"Contrarian setup: {cp:.0%} of informed agents are fading the crowd. "
        f"Institutional sentiment ({is_:+.2f}) diverging from retail. "
        f"Event: '{event.description}' may be creating a temporary dislocation."
    )


def _build_bear_case(
    tier1:    Tier1Result,
    metrics:  dict,
    tier3:    Tier3Result,
) -> str:
    bearish_agents = [r for r in tier1.responses
                      if r.stance == "bearish" and not r.is_fallback]

    if bearish_agents:
        best = max(bearish_agents, key=lambda r: r.conviction)
        return (
            f"{best.persona_name.replace('_', ' ').title()} "
            f"(conviction {best.conviction:.0%}): {best.key_bear_factor}. "
            f"Key risk: {best.tail_risk}. "
            f"Target: {best.three_month_target}."
        )

    hi = metrics["herd_index"]
    pp = metrics["panic_probability"]
    return (
        f"Crowd momentum is {tier3.dominant_sentiment_label} "
        f"with herding at {hi:.0%}. "
        f"Panic probability {pp:.0%} — cascade risk if key levels break. "
        f"Narrative momentum {metrics['narrative_momentum']:+.2f} suggests "
        f"the trend is {'accelerating' if metrics['narrative_momentum'] < -0.20 else 'stabilising'}."
    )


def _build_key_risk(tier1: Tier1Result, event: MarketEvent) -> str:
    risks = [r.tail_risk for r in tier1.responses
             if r.tail_risk and r.tail_risk not in ("unknown", "parse error")
             and not r.is_fallback]
    if risks:
        # Pick the risk mentioned by the highest-conviction agent
        best = max(
            [r for r in tier1.responses if r.tail_risk and not r.is_fallback],
            key=lambda r: r.conviction,
            default=None
        )
        if best:
            return f"[{best.persona_name}] {best.tail_risk}"
    return f"Event escalation: '{event.description}' could worsen beyond current expectations"


def _build_contrarian_thesis(tier1: Tier1Result, metrics: dict) -> str:
    contras = [r for r in tier1.responses
               if r.crowd_assessment == "contrarian_fade" and not r.is_fallback]

    if not contras:
        return (
            f"No strong contrarian thesis from deep thinkers. "
            f"Crowd and institutions are aligned "
            f"({'bullish' if metrics['aggregate_sentiment'] > 0 else 'bearish'} consensus)."
        )

    # Find the most eloquent contrarian
    best = max(contras, key=lambda r: r.conviction)
    crowd_dir = "bearish" if metrics["aggregate_sentiment"] < 0 else "bullish"
    contra_dir = "bullish" if crowd_dir == "bearish" else "bearish"

    return (
        f"{best.persona_name.replace('_', ' ').title()} is {contra_dir} "
        f"while the crowd is {crowd_dir} — "
        f"conviction {best.conviction:.0%}. Thesis: {best.key_bull_factor if contra_dir == 'bullish' else best.key_bear_factor}. "
        f"Total {len(contras)}/{len(tier1.responses)} deep thinkers fading the crowd."
    )


def _build_interpreter_brief(
    event:    MarketEvent,
    metrics:  dict,
    tier1:    Tier1Result,
    tier3:    Tier3Result,
    tier2:    Tier2Result,
    narrative: str,
    narrative_conf: float,
    cascade_risk: str,
    total_agents: int,
) -> str:
    agg  = metrics["aggregate_sentiment"]
    rs   = metrics["retail_sentiment"]
    ins  = metrics["institutional_sentiment"]
    hi   = metrics["herd_index"]
    pp   = metrics["panic_probability"]
    cp   = metrics["contrarian_pressure"]
    nm   = metrics["narrative_momentum"]

    # Determine recommended interpretation
    divergence = ins - rs
    if divergence > 0.25 and rs < -0.10:
        interpretation = (
            "CONTRARIAN BULLISH signal — retail fear with institutional bid. "
            "Weight: HIGH (0.30). Historically precedes recoveries in Indian markets."
        )
    elif divergence < -0.25 and rs > 0.10:
        interpretation = (
            "CONTRARIAN BEARISH signal — retail euphoria with institutional distribution. "
            "Weight: HIGH (0.30). Distribution phase — caution on longs."
        )
    elif abs(agg) > 0.40 and hi > 0.60:
        direction = "bullish" if agg > 0 else "bearish"
        interpretation = (
            f"MOMENTUM CONFIRMING — strong {direction} consensus across all tiers. "
            f"Weight: MODERATE (0.25). Trend likely to continue near-term."
        )
    else:
        interpretation = (
            "MIXED signal — no clear institutional vs retail divergence. "
            "Weight: LOW (0.15). Do not rely heavily on simulation this run."
        )

    # Key Tier 1 insight
    high_conv = [r for r in tier1.responses if r.conviction >= 0.75 and not r.is_fallback]
    tier1_insight = ""
    if high_conv:
        best = max(high_conv, key=lambda r: r.conviction)
        tier1_insight = (
            f"\nHIGH-CONVICTION TIER 1: {best.persona_name} is {best.stance} "
            f"at {best.conviction:.0%} — '{best.key_bull_factor if best.stance == 'bullish' else best.key_bear_factor}'"
        )

    return (
        f"SIMULATION BRIEF FOR DESK — SENTIMENT INTERPRETER AGENT\n"
        f"{'─'*50}\n"
        f"Event:         {event.description}\n"
        f"Total agents:  {total_agents:,} | Steps: {tier3.n_steps}\n"
        f"{'─'*50}\n\n"
        f"AGGREGATE METRICS:\n"
        f"  Sentiment:       {agg:+.4f}  ({tier3.dominant_sentiment_label.upper()})\n"
        f"  Retail:          {rs:+.4f}  "
        f"({'fear' if rs < -0.2 else 'greed' if rs > 0.2 else 'neutral'})\n"
        f"  Institutional:   {ins:+.4f}  "
        f"({'bullish lean' if ins > 0.1 else 'bearish lean' if ins < -0.1 else 'neutral'})\n"
        f"  Divergence:      {divergence:+.4f}  "
        f"({'institutions fading retail' if abs(divergence) > 0.20 else 'aligned'})\n"
        f"  Herd index:      {hi:.0%}  "
        f"({'HIGH — amplification risk' if hi > 0.60 else 'moderate dispersion'})\n"
        f"  Panic prob:      {pp:.0%}\n"
        f"  Contrarian pres: {cp:.0%}  "
        f"({'significant' if cp > 0.30 else 'low'})\n"
        f"  Narrative mom:   {nm:+.4f}  "
        f"({'accelerating' if abs(nm) > 0.30 else 'stabilising'})\n\n"
        f"DOMINANT NARRATIVE: {narrative.upper()} (confidence: {narrative_conf:.0%})\n"
        f"CASCADE RISK:       {cascade_risk.upper()}\n"
        f"{'⚠ CASCADE DETECTED — elevated volatility risk' if cascade_risk in ('high', 'extreme') else ''}\n"
        f"\nTIER BREAKDOWN:\n"
        f"  Tier 3 (10k retail):   "
        f"{tier3.final_snapshot.bull_pct:.0%} bull / {tier3.final_snapshot.bear_pct:.0%} bear\n"
        f"  Tier 2 (200 informed): "
        f"{tier2.bullish_pct:.0%} bull / {tier2.bearish_pct:.0%} bear\n"
        f"  Tier 1 (10 deep):      "
        f"{tier1.bullish_pct:.0%} bull / {tier1.bearish_pct:.0%} bear  "
        f"({tier1.contrarian_pct:.0%} contrarian)\n"
        f"{tier1_insight}\n\n"
        f"RECOMMENDED USE:\n"
        f"  {interpretation}"
    )


# ---------------------------------------------------------------------------
# Opinion cluster builder
# ---------------------------------------------------------------------------

def _build_opinion_clusters(
    tier3:   Tier3Result,
    tier2:   Tier2Result,
    tier1:   Tier1Result,
    metrics: dict,
) -> list[dict]:
    """Build the opinion_clusters list for SimulationReport."""
    clusters = []

    # Cluster 1: Retail crowd (Tier 3 dominant stance)
    final = tier3.final_snapshot
    dominant_retail = (
        "bullish" if final.bull_pct > final.bear_pct else "bearish"
    )
    clusters.append({
        "cluster_id":    "retail_crowd",
        "size_pct":      round(0.75, 4),   # Tier 3 is 75% of agents by count
        "dominant_view": (
            f"Retail crowd: {tier3.dominant_sentiment_label} "
            f"({final.bull_pct:.0%} bull / {final.bear_pct:.0%} bear)"
        ),
        "conviction":    round(final.herd_index, 4),
        "is_influential": True,
        "key_agents":    [],
    })

    # Cluster 2: Informed crowd (Tier 2)
    dominant_informed = (
        "bullish" if tier2.bullish_pct > tier2.bearish_pct else "bearish"
    )
    clusters.append({
        "cluster_id":    "informed_crowd",
        "size_pct":      round(tier2.n_agents / (tier3.n_agents + tier2.n_agents + tier1.n_agents), 4),
        "dominant_view": (
            f"Informed crowd: {dominant_informed} "
            f"({tier2.bullish_pct:.0%} bull / {tier2.bearish_pct:.0%} bear)"
        ),
        "conviction":    round(tier2.avg_conviction, 4),
        "is_influential": True,
        "key_agents":    [],
    })

    # Cluster 3: Institutional deep thinkers (Tier 1)
    dominant_inst = (
        "bullish" if tier1.bullish_pct > tier1.bearish_pct else "bearish"
    )
    t1_key_agents = [
        r.persona_name for r in tier1.responses
        if r.conviction >= 0.70 and not r.is_fallback
    ][:3]
    clusters.append({
        "cluster_id":    "institutional_deep",
        "size_pct":      round(tier1.n_agents / (tier3.n_agents + tier2.n_agents + tier1.n_agents), 4),
        "dominant_view": (
            f"Institutional deep thinkers: {dominant_inst} "
            f"({tier1.bullish_pct:.0%} bull / {tier1.bearish_pct:.0%} bear, "
            f"{tier1.contrarian_pct:.0%} contrarian)"
        ),
        "conviction":    round(tier1.avg_conviction, 4),
        "is_influential": True,
        "key_agents":    t1_key_agents,
    })

    return clusters


# ---------------------------------------------------------------------------
# Main coordinator
# ---------------------------------------------------------------------------

class SimulationCoordinator:
    """
    Aggregates all three tier outputs into a SimulationReport dict.
    Deterministic — no LLM calls.
    """

    def aggregate(
        self,
        event:    MarketEvent,
        tier3:    Tier3Result,
        tier2:    Tier2Result,
        tier1:    Tier1Result,
        instrument: str = "NIFTY50",
    ) -> dict:
        """
        Produce a dict ready for SimulationReport(**result).
        All fields match the Pydantic schema in models/simulation.py.
        """
        run_id = str(uuid.uuid4())
        total_agents = tier3.n_agents + tier2.n_agents + tier1.n_agents

        # ── Weighted sentiment aggregation ───────────────────────────────────
        w3 = TIER_SENTIMENT_WEIGHTS[3]
        w2 = TIER_SENTIMENT_WEIGHTS[2]
        w1 = TIER_SENTIMENT_WEIGHTS[1]

        agg_sentiment = (
            w3 * tier3.final_sentiment +
            w2 * tier2.avg_sentiment   +
            w1 * tier1.avg_sentiment
        )
        agg_sentiment = max(-1.0, min(1.0, agg_sentiment))

        # Retail = Tier 3 + Tier 2 (both are "retail/semi-retail")
        retail_sentiment = (
            0.60 * tier3.final_sentiment + 0.40 * tier2.avg_sentiment
        )
        # Institutional = Tier 1
        institutional_sentiment = tier1.avg_sentiment

        t3_snap = tier3.final_snapshot

        # Panic probability: scale up from Tier 3 panic agent %
        panic_probability = min(1.0, t3_snap.panic_agents_pct * 2.0 +
                                0.05 * len(tier3.cascade_events))

        metrics = {
            "aggregate_sentiment":     round(float(agg_sentiment), 4),
            "retail_sentiment":        round(float(max(-1, min(1, retail_sentiment))), 4),
            "institutional_sentiment": round(float(max(-1, min(1, institutional_sentiment))), 4),
            "herd_index":              round(float(t3_snap.herd_index), 4),
            "panic_probability":       round(float(min(1.0, panic_probability)), 4),
            "narrative_momentum":      round(float(tier3.narrative_momentum), 4),
            "contrarian_pressure":     round(float(tier1.contrarian_pct), 4),
            "information_velocity":    round(float(t3_snap.information_velocity), 4),
        }

        # ── Narrative ────────────────────────────────────────────────────────
        tier1_divergence = institutional_sentiment - agg_sentiment
        narrative, narrative_conf = _infer_narrative(
            agg_sentiment      = metrics["aggregate_sentiment"],
            herd_index         = metrics["herd_index"],
            panic_prob         = metrics["panic_probability"],
            narrative_mom      = metrics["narrative_momentum"],
            contrarian_pct     = metrics["contrarian_pressure"],
            tier1_divergence   = tier1_divergence,
            event              = event,
        )

        # ── Cascade risk ─────────────────────────────────────────────────────
        cascade_risk = _assess_cascade_risk(tier3, metrics["panic_probability"])

        # ── Cascade events ───────────────────────────────────────────────────
        cascade_events = [
            {
                "timestep":      c["step"],
                "trigger":       f"Sentiment cascade at simulation step {c['step']}",
                "direction":     f"{c['direction']}_cascade",
                "magnitude":     round(c["magnitude"], 4),
                "affected_tiers": [3],
                "recovery_steps": None,
            }
            for c in tier3.cascade_events
        ]

        # ── Qualitative synthesis ─────────────────────────────────────────────
        bull_case   = _build_bull_case(tier1, metrics, event)
        bear_case   = _build_bear_case(tier1, metrics, tier3)
        key_risk    = _build_key_risk(tier1, event)
        contra      = _build_contrarian_thesis(tier1, metrics)
        brief       = _build_interpreter_brief(
            event, metrics, tier1, tier3, tier2,
            narrative, narrative_conf, cascade_risk, total_agents
        )

        # ── Opinion clusters ──────────────────────────────────────────────────
        clusters = _build_opinion_clusters(tier3, tier2, tier1, metrics)

        # ── Tier snapshots ────────────────────────────────────────────────────
        tier_snapshots = [
            tier3.to_tier_snapshot_dict(),
            tier2.to_tier_snapshot_dict(),
            tier1.to_tier_snapshot_dict(),
        ]

        return {
            "run_id":                      run_id,
            "created_at":                  datetime.now(timezone.utc),
            "point_in_time_date":          date.today(),
            "seed_event":                  event.description,
            "instrument":                  instrument,
            "simulation_steps":            tier3.n_steps,
            "total_agents":                total_agents,
            "tier_snapshots":              tier_snapshots,
            "dominant_narrative":          narrative,
            "narrative_confidence":        round(narrative_conf, 4),
            "narrative_shift_detected":    tier3.narrative_momentum < -0.30,
            "opinion_clusters":            clusters,
            "cascade_events":              cascade_events,
            "cascade_risk":                cascade_risk,
            "bull_case":                   bull_case,
            "bear_case":                   bear_case,
            "key_risk":                    key_risk,
            "contrarian_thesis":           contra,
            "sentiment_interpreter_brief": brief,
            **metrics,
        }
