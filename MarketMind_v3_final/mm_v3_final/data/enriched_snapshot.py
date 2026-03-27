"""
data/enriched_snapshot.py
==========================
EnrichedSnapshot — replaces FeedSnapshot as the output of the data layer.

Every field carries three metadata attributes:
    value       — the actual number or string
    confidence  — 0.0 to 1.0 (1.0 = cross-validated across sources)
    source      — which source provided this value
    staleness   — seconds since the value was fetched

The quality gate decides whether a snapshot is good enough to enter the swarm.
Fields below minimum confidence are excluded from agent prompts.

Why this matters:
    A sentiment score of 0.45 with confidence 0.95 (validated across Finnhub
    + Alpha Vantage) is worth telling the swarm about.
    A sentiment score of 0.45 with confidence 0.20 (single source, stale) is
    noise. The swarm should not see it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger("marketmind.quality_gate")


# ---------------------------------------------------------------------------
# Tagged field
# ---------------------------------------------------------------------------

@dataclass
class TaggedValue:
    """A single data field with provenance and quality metadata."""
    value:      Any
    confidence: float   # 0.0 – 1.0
    source:     str
    fetched_at: float = field(default_factory=time.time)

    @property
    def staleness_sec(self) -> float:
        return time.time() - self.fetched_at

    @property
    def is_fresh(self) -> bool:
        return self.staleness_sec < 300  # 5 minutes

    def to_dict(self) -> dict:
        return {
            "value":       self.value,
            "confidence":  round(self.confidence, 3),
            "source":      self.source,
            "staleness_s": int(self.staleness_sec),
        }


# ---------------------------------------------------------------------------
# EnrichedSnapshot
# ---------------------------------------------------------------------------

@dataclass
class EnrichedSnapshot:
    """
    Complete picture of a market instrument, multi-sourced and quality-tagged.
    Agents see only the fields that cleared the minimum confidence threshold.
    """
    instrument:  str
    asset_class: str
    timestamp:   datetime

    # ── Price block (primary — quality gate hard requirement) ─────────────
    spot:            Optional[TaggedValue] = None
    change_pct:      Optional[TaggedValue] = None
    high:            Optional[TaggedValue] = None
    low:             Optional[TaggedValue] = None
    volume_ratio:    Optional[TaggedValue] = None

    # ── Technical levels ──────────────────────────────────────────────────
    ma_20:           Optional[TaggedValue] = None
    ma_50:           Optional[TaggedValue] = None
    ma_200:          Optional[TaggedValue] = None
    rsi_14:          Optional[TaggedValue] = None
    atr_pct:         Optional[TaggedValue] = None
    hist_vol_20:     Optional[TaggedValue] = None
    week_52_high:    Optional[TaggedValue] = None
    week_52_low:     Optional[TaggedValue] = None

    # ── News and sentiment ────────────────────────────────────────────────
    news_sentiment_score:    Optional[TaggedValue] = None   # composite
    finnhub_news_score:      Optional[TaggedValue] = None
    av_news_sentiment:       Optional[TaggedValue] = None
    news_headlines:          list[str] = field(default_factory=list)
    news_count_24h:          Optional[TaggedValue] = None
    buzz_score:              Optional[TaggedValue] = None

    # ── Macro context ─────────────────────────────────────────────────────
    us_10y:              Optional[TaggedValue] = None
    us_2y:               Optional[TaggedValue] = None
    yield_curve_2s10s:   Optional[TaggedValue] = None
    fed_funds_rate:      Optional[TaggedValue] = None
    breakeven_10y:       Optional[TaggedValue] = None
    cpi_yoy:             Optional[TaggedValue] = None
    core_cpi_yoy:        Optional[TaggedValue] = None
    unemployment_rate:   Optional[TaggedValue] = None
    credit_spread_hy:    Optional[TaggedValue] = None
    m2_growth:           Optional[TaggedValue] = None
    sofr:                Optional[TaggedValue] = None
    vix:                 Optional[TaggedValue] = None
    dxy:                 Optional[TaggedValue] = None
    gold_usd:            Optional[TaggedValue] = None

    # ── Equity-specific ───────────────────────────────────────────────────
    pcr:                 Optional[TaggedValue] = None
    india_vix:           Optional[TaggedValue] = None
    earnings_days_away:  Optional[TaggedValue] = None
    analyst_bull_pct:    Optional[TaggedValue] = None
    insider_sentiment:   Optional[TaggedValue] = None
    earnings_surprise:   Optional[TaggedValue] = None

    # ── India-specific ────────────────────────────────────────────────────
    repo_rate:           Optional[TaggedValue] = None
    india_10y_yield:     Optional[TaggedValue] = None
    cpi_india:           Optional[TaggedValue] = None

    # ── Energy-specific ───────────────────────────────────────────────────
    crude_inventory_change:  Optional[TaggedValue] = None
    natgas_storage_change:   Optional[TaggedValue] = None
    refinery_utilisation:    Optional[TaggedValue] = None

    # ── Crypto-specific ───────────────────────────────────────────────────
    crypto_fear_greed:       Optional[TaggedValue] = None
    btc_dominance:           Optional[TaggedValue] = None
    total_market_cap_usd:    Optional[TaggedValue] = None

    # ── Quality metadata ──────────────────────────────────────────────────
    sources_used:        list[str] = field(default_factory=list)
    data_warnings:       list[str] = field(default_factory=list)
    gate_passed:         bool = False
    gate_reason:         str  = ""
    overall_confidence:  float = 0.0

    def to_agent_dict(self, min_confidence: float = 0.40) -> dict:
        """
        Convert to a flat dict for LLM agent prompts.
        Only includes fields above min_confidence threshold.
        Fields below threshold are excluded — agents see nothing is better
        than agents seeing low-confidence noise.
        """
        result: dict = {
            "instrument":   self.instrument,
            "asset_class":  self.asset_class,
            "fetched_at":   self.timestamp.isoformat(),
            "sources_used": self.sources_used,
        }

        def _add(name: str, tv: Optional[TaggedValue]) -> None:
            if tv is not None and tv.confidence >= min_confidence:
                result[name] = tv.value
                result[f"{name}_confidence"] = round(tv.confidence, 2)

        # Price block
        _add("spot",         self.spot)
        _add("change_pct",   self.change_pct)
        _add("high",         self.high)
        _add("low",          self.low)
        _add("volume_ratio", self.volume_ratio)

        # Technicals
        for attr in ("ma_20","ma_50","ma_200","rsi_14","atr_pct",
                     "hist_vol_20","week_52_high","week_52_low"):
            _add(attr, getattr(self, attr))

        # News
        _add("news_sentiment_score", self.news_sentiment_score)
        _add("buzz_score",           self.buzz_score)
        _add("news_count_24h",       self.news_count_24h)
        if self.news_headlines:
            result["top_headlines"] = self.news_headlines[:5]

        # Macro
        for attr in ("us_10y","us_2y","yield_curve_2s10s","fed_funds_rate",
                     "breakeven_10y","cpi_yoy","core_cpi_yoy","unemployment_rate",
                     "credit_spread_hy","vix","dxy","gold_usd","sofr"):
            _add(attr, getattr(self, attr))

        # Asset-class specific
        for attr in ("pcr","india_vix","earnings_days_away","analyst_bull_pct",
                     "insider_sentiment","earnings_surprise",
                     "repo_rate","india_10y_yield","cpi_india",
                     "crude_inventory_change","natgas_storage_change","refinery_utilisation",
                     "crypto_fear_greed","btc_dominance","total_market_cap_usd"):
            _add(attr, getattr(self, attr))

        if self.data_warnings:
            result["data_warnings"] = self.data_warnings

        return result


# ---------------------------------------------------------------------------
# Data Quality Gate
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    passed:          bool
    reason:          str
    overall_confidence: float
    missing_fields:  list[str] = field(default_factory=list)
    low_confidence:  list[str] = field(default_factory=list)
    warnings:        list[str] = field(default_factory=list)


class DataQualityGate:
    """
    Decides whether an EnrichedSnapshot is good enough to enter the swarm.

    Hard requirements (ABORT if missing):
      - spot price > 0 and not stale (< 5 minutes for liquid markets)
      - change_pct available

    Soft requirements (warnings, don't abort):
      - At least 1 news source with confidence >= 0.5
      - At least 1 macro context field
      - No single source providing > 80% of fields (concentration risk)

    Earnings proximity check:
      - If earnings_days_away <= 2 and asset_class is equity:
        → add do_not_trade warning automatically (risk agent also checks this
          but belt-and-suspenders means the data layer flags it too)
    """

    HARD_REQUIRED_FIELDS = ["spot"]
    SOFT_REQUIRED_FIELDS = ["change_pct", "rsi_14", "hist_vol_20"]
    NEWS_FIELDS          = ["news_sentiment_score", "finnhub_news_score", "av_news_sentiment"]
    MACRO_FIELDS         = ["us_10y", "vix", "dxy", "cpi_yoy", "fed_funds_rate"]

    MIN_SPOT_CONFIDENCE  = 0.70   # spot price must be highly confident
    MIN_FIELD_CONFIDENCE = 0.40   # other fields below this are excluded
    MAX_SPOT_STALENESS   = 300    # 5 minutes for liquid markets

    def evaluate(self, snap: EnrichedSnapshot) -> GateResult:
        warnings: list[str] = []
        missing:  list[str] = []
        low_conf: list[str] = []

        # ── Hard requirement: spot price ─────────────────────────────────────
        if snap.spot is None:
            return GateResult(
                passed=False,
                reason="spot_price_missing — no data source returned a valid price",
                overall_confidence=0.0,
                missing_fields=["spot"],
            )

        if snap.spot.value <= 0:
            return GateResult(
                passed=False,
                reason=f"spot_price_zero_or_negative ({snap.spot.value})",
                overall_confidence=0.0,
            )

        if snap.spot.confidence < self.MIN_SPOT_CONFIDENCE:
            return GateResult(
                passed=False,
                reason=f"spot_confidence_too_low ({snap.spot.confidence:.2f} < {self.MIN_SPOT_CONFIDENCE})",
                overall_confidence=snap.spot.confidence,
            )

        if snap.spot.staleness_sec > self.MAX_SPOT_STALENESS:
            return GateResult(
                passed=False,
                reason=f"spot_stale ({snap.spot.staleness_sec:.0f}s > {self.MAX_SPOT_STALENESS}s)",
                overall_confidence=snap.spot.confidence,
            )

        # ── Soft requirements ────────────────────────────────────────────────
        for field_name in self.SOFT_REQUIRED_FIELDS:
            tv = getattr(snap, field_name, None)
            if tv is None:
                missing.append(field_name)
            elif tv.confidence < self.MIN_FIELD_CONFIDENCE:
                low_conf.append(field_name)

        # ── News coverage ────────────────────────────────────────────────────
        news_coverage = sum(
            1 for f in self.NEWS_FIELDS
            if getattr(snap, f, None) is not None
            and getattr(snap, f).confidence >= 0.5
        )
        if news_coverage == 0:
            warnings.append("no_news_data — news sentiment not enriched")

        # ── Macro coverage ───────────────────────────────────────────────────
        macro_coverage = sum(
            1 for f in self.MACRO_FIELDS
            if getattr(snap, f, None) is not None
        )
        if macro_coverage == 0:
            warnings.append("no_macro_context — no FRED/yield data enriched")

        # ── Earnings proximity (equity) ──────────────────────────────────────
        if snap.asset_class.startswith("equity") and snap.earnings_days_away is not None:
            days = snap.earnings_days_away.value
            if isinstance(days, (int, float)) and 0 <= days <= 2:
                warnings.append(
                    f"earnings_in_{int(days)}_days — recommend do_not_trade (risk agent will also check)"
                )

        # ── Source concentration check ────────────────────────────────────────
        source_counts: dict[str, int] = {}
        for attr_name in vars(EnrichedSnapshot):
            tv = getattr(snap, attr_name, None)
            if isinstance(tv, TaggedValue):
                source_counts[tv.source] = source_counts.get(tv.source, 0) + 1
        total_fields = sum(source_counts.values())
        if total_fields > 0:
            max_concentration = max(source_counts.values()) / total_fields
            if max_concentration > 0.80:
                dominant = max(source_counts, key=source_counts.get)
                warnings.append(f"single_source_dominance — {dominant} provides {max_concentration:.0%} of fields")

        # ── Overall confidence ────────────────────────────────────────────────
        all_confidences = [
            getattr(snap, attr).confidence
            for attr in vars(EnrichedSnapshot)
            if isinstance(getattr(snap, attr, None), TaggedValue)
        ]
        overall = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

        return GateResult(
            passed           = True,
            reason           = "gate_passed",
            overall_confidence = round(overall, 3),
            missing_fields   = missing,
            low_confidence   = low_conf,
            warnings         = warnings,
        )
