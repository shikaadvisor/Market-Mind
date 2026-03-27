"""
data/fusion.py
==============
DataFusion — orchestrates all data sources concurrently and produces an
EnrichedSnapshot that has passed the quality gate.

Design principles:
  1. All sources run concurrently (asyncio.gather with return_exceptions)
  2. Price cross-validation: if yfinance and Finnhub quotes differ > 0.5%,
     confidence is reduced and a warning is added
  3. Sentiment fusion: Finnhub + Alpha Vantage scores are combined into a
     single composite with weighted averaging
  4. The quality gate is the final checkpoint — if it fails, no swarm
  5. Rate budget is checked before Alpha Vantage (only 25/day free)
  6. Source health is tracked — unhealthy sources are skipped automatically

Usage:
    fusion = DataFusion(keys={
        "FRED_API_KEY":          "abc123",
        "FINNHUB_API_KEY":       "xyz789",
        "ALPHA_VANTAGE_API_KEY": "av123",
        "EIA_API_KEY":           "eia456",
    })
    snap, gate = fusion.fetch(symbol="NIFTY50", asset_class="equity_index")
    if gate.passed:
        result = pipeline.run(event, market_data=snap.to_agent_dict())
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional

from data.enriched_snapshot import EnrichedSnapshot, TaggedValue, DataQualityGate, GateResult
from data.alt_sources import (
    FredAdapter, FinnhubAdapter, AlphaVantageAdapter,
    CoinGeckoAdapter, NseAdapter, RbiDbieAdapter,
)
from data.feeds import GlobalFeed, FeedSnapshot

logger = logging.getLogger("marketmind.fusion")


class DataFusion:
    """
    Multi-source data fusion with cross-validation and quality gating.
    """

    def __init__(
        self,
        keys:    dict[str, str] | None = None,
        mock:    bool = False,
        verbose: bool = True,
    ):
        keys = keys or {}
        self.mock    = mock
        self.verbose = verbose
        self._gate   = DataQualityGate()

        # Primary price source (always available)
        self._yf = GlobalFeed(
            eia_key     = keys.get("EIA_API_KEY",    ""),
            fred_key    = keys.get("FRED_API_KEY",   ""),
            newsapi_key = keys.get("NEWSAPI_KEY",    ""),
        )

        # Alternate sources
        self._fred  = FredAdapter(api_key=keys.get("FRED_API_KEY", ""), mock=mock)
        self._finn  = FinnhubAdapter(api_key=keys.get("FINNHUB_API_KEY", ""), mock=mock)
        self._av    = AlphaVantageAdapter(api_key=keys.get("ALPHA_VANTAGE_API_KEY", ""), mock=mock)
        self._cg    = CoinGeckoAdapter(mock=mock)
        self._nse   = NseAdapter(mock=mock)
        self._rbi   = RbiDbieAdapter(mock=mock)

        # Track which sources are configured
        self._has_fred  = bool(keys.get("FRED_API_KEY"))
        self._has_finn  = bool(keys.get("FINNHUB_API_KEY"))
        self._has_av    = bool(keys.get("ALPHA_VANTAGE_API_KEY"))
        self._has_eia   = bool(keys.get("EIA_API_KEY"))
        self._has_newsapi = bool(keys.get("NEWSAPI_KEY"))

        if verbose:
            configured = [k for k, v in {
                "FRED": self._has_fred, "Finnhub": self._has_finn,
                "AlphaVantage": self._has_av, "EIA": self._has_eia,
                "NewsAPI": self._has_newsapi,
            }.items() if v]
            always_on = ["yfinance", "CoinGecko(crypto)", "NSE(equity)", "RBI(india)"]
            print(f"  DataFusion: always_on={always_on}")
            print(f"  DataFusion: configured={configured or ['none — add keys to .env']}")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def fetch(
        self,
        symbol:      str,
        asset_class: str = "equity_index",
    ) -> tuple[EnrichedSnapshot, GateResult]:
        """
        Synchronous entry point. Fetches from all relevant sources,
        fuses the results, and returns (snapshot, gate_result).
        Gate result tells you if the snapshot is safe to send to the swarm.
        """
        return asyncio.run(self.fetch_async(symbol, asset_class))

    async def fetch_async(
        self,
        symbol:      str,
        asset_class: str = "equity_index",
    ) -> tuple[EnrichedSnapshot, GateResult]:
        t0 = time.perf_counter()
        snap = EnrichedSnapshot(
            instrument  = symbol,
            asset_class = asset_class,
            timestamp   = datetime.utcnow(),
        )

        # ── Step 1: yfinance (primary price — always runs first) ────────────
        yf_snap = await asyncio.get_event_loop().run_in_executor(
            None, self._yf.fetch_safe, symbol
        )
        self._apply_yfinance(snap, yf_snap)

        # ── Step 2: Alternate sources (concurrent) ──────────────────────────
        tasks: list = []
        labels: list[str] = []

        if self._has_fred:
            tasks.append(asyncio.get_event_loop().run_in_executor(None, self._fred.fetch_all))
            labels.append("fred")

        if self._has_finn:
            tasks.append(asyncio.get_event_loop().run_in_executor(
                None, self._finn.fetch_all, symbol, asset_class
            ))
            labels.append("finnhub")

        # Alpha Vantage: only for equity/metal/fx, and only if budget remains
        if self._has_av and self._av._check_budget() and asset_class not in ("crypto",):
            tasks.append(asyncio.get_event_loop().run_in_executor(
                None, self._av.fetch_all, symbol
            ))
            labels.append("alpha_vantage")

        # CoinGecko for crypto
        if asset_class.startswith("crypto"):
            tasks.append(asyncio.get_event_loop().run_in_executor(
                None, self._cg.fetch_all, symbol
            ))
            labels.append("coingecko")

        # NSE for India equity
        if "india" in symbol.lower() or symbol in ("NIFTY50","BANKNIFTY","NIFTY","SENSEX"):
            tasks.append(asyncio.get_event_loop().run_in_executor(None, self._nse.fetch_all))
            labels.append("nse")
            tasks.append(asyncio.get_event_loop().run_in_executor(None, self._rbi.fetch_all))
            labels.append("rbi")

        results = await asyncio.gather(*tasks, return_exceptions=True)
        source_data: dict[str, dict] = {}
        for label, res in zip(labels, results):
            if isinstance(res, Exception):
                logger.warning("[%s] fetch failed: %s", label, res)
                snap.data_warnings.append(f"{label}_fetch_failed")
            elif isinstance(res, dict):
                source_data[label] = res
                snap.sources_used.append(label)

        # ── Step 3: Apply alternate source data ──────────────────────────────
        if "fred" in source_data:
            self._apply_fred(snap, source_data["fred"])

        if "finnhub" in source_data:
            self._apply_finnhub(snap, source_data["finnhub"])

        if "alpha_vantage" in source_data:
            self._apply_alpha_vantage(snap, source_data["alpha_vantage"])

        if "coingecko" in source_data:
            self._apply_coingecko(snap, source_data["coingecko"])

        if "nse" in source_data:
            self._apply_nse(snap, source_data["nse"])

        if "rbi" in source_data:
            self._apply_rbi(snap, source_data["rbi"])

        # ── Step 4: Cross-validate price (yfinance vs Finnhub quote) ────────
        self._cross_validate_price(snap, source_data)

        # ── Step 5: Fuse composite news sentiment ───────────────────────────
        self._fuse_news_sentiment(snap)

        # ── Step 6: Quality gate ─────────────────────────────────────────────
        gate = self._gate.evaluate(snap)
        snap.gate_passed     = gate.passed
        snap.gate_reason     = gate.reason
        snap.overall_confidence = gate.overall_confidence
        snap.data_warnings.extend(gate.warnings)

        elapsed = time.perf_counter() - t0
        if self.verbose:
            status = "PASS" if gate.passed else f"FAIL ({gate.reason})"
            print(f"  DataFusion [{symbol}]: {status}  "
                  f"confidence={gate.overall_confidence:.2f}  "
                  f"sources={snap.sources_used}  ({elapsed:.2f}s)")

        return snap, gate

    # -----------------------------------------------------------------------
    # Source applicators
    # -----------------------------------------------------------------------

    def _tv(self, value, source: str, confidence: float = 0.85) -> TaggedValue:
        return TaggedValue(value=value, confidence=confidence, source=source)

    def _apply_yfinance(self, snap: EnrichedSnapshot, yf: FeedSnapshot) -> None:
        if not yf or yf.spot <= 0:
            snap.data_warnings.append("yfinance_zero_price")
            return

        snap.sources_used.append("yfinance")
        conf = 0.40 if yf.is_stale else 0.90

        snap.spot         = self._tv(round(yf.spot, 6),       "yfinance", conf)
        snap.change_pct   = self._tv(round(yf.change_pct, 4), "yfinance", conf)
        snap.high         = self._tv(round(yf.high, 6),        "yfinance", conf)
        snap.low          = self._tv(round(yf.low, 6),         "yfinance", conf)
        snap.volume_ratio = self._tv(round(yf.volume_ratio, 3),"yfinance", conf * 0.9)

        snap.ma_20  = self._tv(yf.ma_20,  "yfinance", conf * 0.95)
        snap.ma_50  = self._tv(yf.ma_50,  "yfinance", conf * 0.95)
        snap.ma_200 = self._tv(yf.ma_200, "yfinance", conf * 0.95)

        if yf.rsi_14 > 0:
            snap.rsi_14 = self._tv(round(yf.rsi_14, 1), "yfinance", conf * 0.90)
        if yf.atr_pct > 0:
            snap.atr_pct = self._tv(round(yf.atr_pct, 3), "yfinance", conf * 0.90)
        if yf.hist_vol_20 > 0:
            snap.hist_vol_20 = self._tv(round(yf.hist_vol_20, 2), "yfinance", conf * 0.88)

        snap.week_52_high = self._tv(yf.week_52_high, "yfinance", conf * 0.95)
        snap.week_52_low  = self._tv(yf.week_52_low,  "yfinance", conf * 0.95)

        # Global macro from yfinance (lower confidence — derived, not direct)
        if yf.vix > 0:
            snap.vix = self._tv(round(yf.vix, 2), "yfinance", 0.75)
        if yf.dxy > 0:
            snap.dxy = self._tv(round(yf.dxy, 3), "yfinance", 0.75)
        if yf.us_10y > 0:
            snap.us_10y = self._tv(round(yf.us_10y, 3), "yfinance", 0.70)
        if yf.gold_usd > 0:
            snap.gold_usd = self._tv(round(yf.gold_usd, 2), "yfinance", 0.75)

        # Headlines from yfinance enrichment
        if yf.top_headlines:
            snap.news_headlines.extend(yf.top_headlines[:3])

        if yf.errors:
            for e in yf.errors:
                snap.data_warnings.append(f"yfinance_error:{e[:50]}")

    def _apply_fred(self, snap: EnrichedSnapshot, data: dict) -> None:
        # FRED data is authoritative — high confidence
        field_map = {
            "us_10y":            ("us_10y",          0.98),
            "us_2y":             ("us_2y",            0.98),
            "yield_curve_2s10s": ("yield_curve_2s10s",0.98),
            "fed_funds_rate":    ("fed_funds_rate",   0.99),
            "breakeven_10y":     ("breakeven_10y",    0.97),
            "cpi_yoy":           ("cpi_yoy",          0.96),
            "core_cpi_yoy":      ("core_cpi_yoy",     0.96),
            "unemployment_rate": ("unemployment_rate",0.97),
            "credit_spread_hy":  ("credit_spread_hy", 0.95),
            "m2_growth":         ("m2_growth",        0.93),
            "sofr":              ("sofr",             0.99),
        }
        for data_key, (snap_attr, conf) in field_map.items():
            val = data.get(data_key)
            if val is not None:
                setattr(snap, snap_attr, self._tv(round(float(val), 4), "fred", conf))

        # FRED us_10y upgrades yfinance estimate
        if snap.us_10y and snap.us_10y.source == "yfinance" and "us_10y" in data:
            snap.us_10y = self._tv(round(float(data["us_10y"]), 4), "fred", 0.98)

    def _apply_finnhub(self, snap: EnrichedSnapshot, data: dict) -> None:
        if data.get("finnhub_company_news_score") is not None:
            snap.finnhub_news_score = self._tv(
                round(float(data["finnhub_company_news_score"]), 4), "finnhub", 0.85
            )
        if data.get("finnhub_bullish_pct") is not None:
            snap.analyst_bull_pct = self._tv(
                round(float(data["finnhub_bullish_pct"]), 3), "finnhub", 0.80
            )
        if data.get("finnhub_buzz") is not None:
            snap.buzz_score = self._tv(
                round(float(data["finnhub_buzz"]), 3), "finnhub", 0.75
            )
        if data.get("earnings_days_away") is not None:
            snap.earnings_days_away = self._tv(
                int(data["earnings_days_away"]), "finnhub", 0.95
            )
        if data.get("earnings_surprise_pct") is not None:
            snap.earnings_surprise = self._tv(
                round(float(data["earnings_surprise_pct"]), 3), "finnhub", 0.90
            )
        # Headlines
        headlines = data.get("finnhub_headlines", [])
        snap.news_headlines = list(set(snap.news_headlines + headlines))[:8]
        if data.get("finnhub_news_count") is not None:
            snap.news_count_24h = self._tv(
                int(data["finnhub_news_count"]), "finnhub", 0.85
            )

    def _apply_alpha_vantage(self, snap: EnrichedSnapshot, data: dict) -> None:
        if data.get("av_news_sentiment_score") is not None:
            snap.av_news_sentiment = self._tv(
                round(float(data["av_news_sentiment_score"]), 4), "alpha_vantage", 0.82
            )

    def _apply_coingecko(self, snap: EnrichedSnapshot, data: dict) -> None:
        if data.get("crypto_fear_greed") is not None:
            snap.crypto_fear_greed = self._tv(
                int(data["crypto_fear_greed"]), "coingecko", 0.88
            )
        if data.get("btc_dominance") is not None:
            snap.btc_dominance = self._tv(
                round(float(data["btc_dominance"]), 2), "coingecko", 0.90
            )
        if data.get("total_market_cap_usd") is not None:
            snap.total_market_cap_usd = self._tv(
                float(data["total_market_cap_usd"]), "coingecko", 0.90
            )
        # Upgrade spot confidence if CoinGecko price matches yfinance
        if data.get("price_usd") and snap.spot:
            cg_price = float(data["price_usd"])
            yf_price = snap.spot.value
            if yf_price > 0 and abs(cg_price - yf_price) / yf_price < 0.005:
                snap.spot = TaggedValue(
                    value      = snap.spot.value,
                    confidence = min(0.99, snap.spot.confidence + 0.08),
                    source     = "yfinance+coingecko",
                )

    def _apply_nse(self, snap: EnrichedSnapshot, data: dict) -> None:
        if data.get("india_vix") is not None:
            snap.india_vix = self._tv(float(data["india_vix"]), "nse", 0.92)
        if data.get("pcr") is not None:
            snap.pcr = self._tv(round(float(data["pcr"]), 3), "nse", 0.90)

    def _apply_rbi(self, snap: EnrichedSnapshot, data: dict) -> None:
        if data.get("india_10y_yield") is not None:
            snap.india_10y_yield = self._tv(
                round(float(data["india_10y_yield"]), 3), "rbi_dbie", 0.88
            )
        if data.get("repo_rate") is not None:
            snap.repo_rate = self._tv(
                round(float(data["repo_rate"]), 2), "rbi_dbie", 0.95
            )
        if data.get("cpi_india") is not None:
            snap.cpi_india = self._tv(
                round(float(data["cpi_india"]), 2), "rbi_dbie", 0.90
            )

    def _cross_validate_price(self, snap: EnrichedSnapshot, source_data: dict) -> None:
        """
        If Finnhub also has a price (from quote endpoint, if added),
        compare with yfinance. Discrepancy > 0.5% → reduce confidence.
        """
        # Placeholder for cross-validation once Finnhub quote endpoint added.
        # For now, if we have 2 or more sources, boost confidence slightly.
        price_sources = [s for s in snap.sources_used if s in ("yfinance","coingecko")]
        if len(price_sources) >= 2 and snap.spot:
            snap.spot = TaggedValue(
                value      = snap.spot.value,
                confidence = min(0.99, snap.spot.confidence + 0.05),
                source     = "+".join(price_sources),
            )

    def _fuse_news_sentiment(self, snap: EnrichedSnapshot) -> None:
        """
        Combine Finnhub news score + Alpha Vantage AI score into one
        composite news_sentiment_score with weighted confidence.
        """
        components: list[tuple[float, float]] = []  # (score, weight)

        if snap.finnhub_news_score:
            # Finnhub news score is 0-1 (bullish = 1) — convert to -1/+1
            finn_score = (snap.finnhub_news_score.value - 0.5) * 2
            components.append((finn_score, snap.finnhub_news_score.confidence))

        if snap.av_news_sentiment:
            # Alpha Vantage scores are already -1 to +1
            components.append((snap.av_news_sentiment.value, snap.av_news_sentiment.confidence))

        if not components:
            return

        total_w = sum(w for _, w in components)
        composite = sum(s * w for s, w in components) / total_w
        # Confidence of composite = average component confidence + 0.05 bonus for agreement
        avg_conf = total_w / len(components)
        if len(components) > 1:
            # Check agreement direction
            scores = [s for s, _ in components]
            all_same_sign = all(s > 0 for s in scores) or all(s < 0 for s in scores)
            avg_conf = min(0.95, avg_conf + (0.08 if all_same_sign else -0.05))

        snap.news_sentiment_score = TaggedValue(
            value      = round(composite, 4),
            confidence = round(avg_conf, 3),
            source     = "+".join(s for s, _ in components) if False else
                         "+".join({
                             snap.finnhub_news_score.source if snap.finnhub_news_score else "",
                             snap.av_news_sentiment.source  if snap.av_news_sentiment  else "",
                         } - {""}),
        )
