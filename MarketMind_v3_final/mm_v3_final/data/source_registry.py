"""
data/source_registry.py
========================
Central registry of every data source MarketMind uses.

One place to:
  - See every source, its rate limits, cost, and what it provides
  - Track live health (last_success, consecutive_failures)
  - Decide which sources to try for a given instrument/asset_class
  - Budget API calls against rate limits before firing

Registration → get API keys at:
  FRED:          https://fred.stlouisfed.org/docs/api/api_key.html   (free, instant)
  Finnhub:       https://finnhub.io/register                         (free, instant, 60/min)
  Alpha Vantage: https://www.alphavantage.co/support/#api-key        (free, instant, 25/day)
  EIA:           https://www.eia.gov/opendata/register.php           (free, instant)
  CoinGecko:     no key needed for public endpoints
  RBI DBIE:      https://dbie.rbi.org.in                             (no key, public)
  NSE:           no key — public CSV/JSON endpoints
  NewsAPI:       https://newsapi.org/register                        (free 100/day dev tier)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SourceTier(str, Enum):
    FREE_NO_KEY  = "free_no_key"    # yfinance, CoinGecko, RBI DBIE, NSE
    FREE_KEY     = "free_key"       # FRED, Finnhub, Alpha Vantage, EIA, NewsAPI
    PAID         = "paid"           # future: Polygon, Twelve Data, etc.


@dataclass
class RateLimit:
    calls_per_minute:  Optional[int] = None
    calls_per_day:     Optional[int] = None
    calls_per_second:  Optional[float] = None

    def min_interval_sec(self) -> float:
        """Minimum seconds to wait between calls to respect rate limits."""
        intervals = []
        if self.calls_per_minute:
            intervals.append(60.0 / self.calls_per_minute)
        if self.calls_per_second:
            intervals.append(1.0 / self.calls_per_second)
        return max(intervals) if intervals else 0.0


@dataclass
class SourceSpec:
    name:          str
    tier:          SourceTier
    rate_limit:    RateLimit
    env_key:       Optional[str]         # env var name for the API key
    register_url:  str
    description:   str
    asset_classes: list[str]             # which asset classes this helps
    provides:      list[str]             # data fields it contributes
    priority:      int = 50             # higher = tried first for overlapping fields

    # Runtime health tracking (not persisted — reset each process)
    _last_success:         float = field(default=0.0, init=False, repr=False)
    _consecutive_failures: int   = field(default=0,   init=False, repr=False)
    _daily_calls:          int   = field(default=0,   init=False, repr=False)
    _daily_reset_at:       float = field(default=0.0, init=False, repr=False)
    _last_call_at:         float = field(default=0.0, init=False, repr=False)

    @property
    def is_healthy(self) -> bool:
        return self._consecutive_failures < 5

    @property
    def daily_budget_ok(self) -> bool:
        now = time.time()
        if now - self._daily_reset_at > 86400:
            self._daily_calls    = 0
            self._daily_reset_at = now
        if self.rate_limit.calls_per_day:
            return self._daily_calls < self.rate_limit.calls_per_day * 0.90  # 10% safety margin
        return True

    def wait_if_needed(self) -> None:
        """Sleep if calling too fast."""
        interval = self.rate_limit.min_interval_sec()
        if interval > 0:
            elapsed = time.time() - self._last_call_at
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def record_call(self, success: bool) -> None:
        self._last_call_at = time.time()
        self._daily_calls += 1
        if success:
            self._last_success         = time.time()
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1


# ---------------------------------------------------------------------------
# Source definitions
# ---------------------------------------------------------------------------

SOURCES: dict[str, SourceSpec] = {

    # ── Tier 1: No key ────────────────────────────────────────────────────

    "yfinance": SourceSpec(
        name          = "yfinance",
        tier          = SourceTier.FREE_NO_KEY,
        rate_limit    = RateLimit(calls_per_minute=30),
        env_key       = None,
        register_url  = "https://pypi.org/project/yfinance/",
        description   = "Yahoo Finance — prices, OHLCV, technicals, OI for all asset classes",
        asset_classes = ["all"],
        provides      = ["spot","open","high","low","volume","ma_20","ma_50","ma_200",
                         "rsi_14","atr_14","atr_pct","hist_vol_20","week_52_high","week_52_low",
                         "change_pct","week_change_pct","month_change_pct","ytd_change_pct"],
        priority      = 100,
    ),

    "coingecko": SourceSpec(
        name          = "coingecko",
        tier          = SourceTier.FREE_NO_KEY,
        rate_limit    = RateLimit(calls_per_minute=30),
        env_key       = None,
        register_url  = "https://www.coingecko.com/en/api",
        description   = "CoinGecko — crypto prices, market cap, volume, fear/greed index",
        asset_classes = ["crypto"],
        provides      = ["spot","market_cap","volume_24h","change_pct_24h","fear_greed_index",
                         "btc_dominance","total_market_cap"],
        priority      = 90,
    ),

    "rbi_dbie": SourceSpec(
        name          = "rbi_dbie",
        tier          = SourceTier.FREE_NO_KEY,
        rate_limit    = RateLimit(calls_per_minute=10),
        env_key       = None,
        register_url  = "https://dbie.rbi.org.in/DBIE/dbie.rbi?site=statistics",
        description   = "RBI Database on Indian Economy — repo rate, CPI India, INR, FII flows",
        asset_classes = ["equity_index", "fx_em", "fixed_income"],
        provides      = ["repo_rate","cpi_india","india_gdp_growth","fii_equity_net",
                         "dii_equity_net","india_10y_yield"],
        priority      = 80,
    ),

    "nse_public": SourceSpec(
        name          = "nse_public",
        tier          = SourceTier.FREE_NO_KEY,
        rate_limit    = RateLimit(calls_per_minute=10),
        env_key       = None,
        register_url  = "https://www.nseindia.com/api",
        description   = "NSE India public API — PCR, OI, delivery %, FII/DII daily data",
        asset_classes = ["equity_index", "equity_stock"],
        provides      = ["pcr","total_call_oi","total_put_oi","delivery_pct",
                         "fii_index_fut_net","dii_cash_net","india_vix"],
        priority      = 85,
    ),

    # ── Tier 2: Free API key ───────────────────────────────────────────────

    "fred": SourceSpec(
        name          = "fred",
        tier          = SourceTier.FREE_KEY,
        rate_limit    = RateLimit(calls_per_minute=120),   # very generous
        env_key       = "FRED_API_KEY",
        register_url  = "https://fred.stlouisfed.org/docs/api/api_key.html",
        description   = "Federal Reserve Economic Data — macro indicators, yields, CPI, GDP",
        asset_classes = ["all"],
        provides      = ["us_10y","us_2y","us_30y","fed_funds_rate","cpi_yoy","core_cpi_yoy",
                         "pce_yoy","real_gdp_growth","unemployment_rate","m2_growth",
                         "breakeven_10y","credit_spread_hy","dxy","sofr","yield_curve_2s10s"],
        priority      = 95,
    ),

    "finnhub": SourceSpec(
        name          = "finnhub",
        tier          = SourceTier.FREE_KEY,
        rate_limit    = RateLimit(calls_per_minute=60),
        env_key       = "FINNHUB_API_KEY",
        register_url  = "https://finnhub.io/register",
        description   = "Finnhub — news, sentiment, earnings calendar, insider transactions, fundamentals",
        asset_classes = ["equity_index", "equity_stock", "crypto", "fx_major"],
        provides      = ["news_headlines","news_sentiment","earnings_date","earnings_surprise",
                         "recommendation_trend","insider_sentiment","price_target",
                         "economic_calendar_events","analyst_rating"],
        priority      = 90,
    ),

    "alpha_vantage": SourceSpec(
        name          = "alpha_vantage",
        tier          = SourceTier.FREE_KEY,
        rate_limit    = RateLimit(calls_per_minute=5, calls_per_day=25),
        env_key       = "ALPHA_VANTAGE_API_KEY",
        register_url  = "https://www.alphavantage.co/support/#api-key",
        description   = "Alpha Vantage — AI news sentiment scores, economic indicators (25/day free)",
        asset_classes = ["equity_index", "equity_stock", "fx_major", "commodity_metal"],
        provides      = ["news_sentiment_score","news_relevance_score","sector_sentiment",
                         "economic_indicators_us","real_gdp","treasury_yield"],
        priority      = 60,   # lower — daily limit means we use sparingly
    ),

    "eia": SourceSpec(
        name          = "eia",
        tier          = SourceTier.FREE_KEY,
        rate_limit    = RateLimit(calls_per_minute=500),   # very generous
        env_key       = "EIA_API_KEY",
        register_url  = "https://www.eia.gov/opendata/register.php",
        description   = "EIA — crude/nat gas inventory, refinery utilisation, production data",
        asset_classes = ["commodity_energy"],
        provides      = ["crude_inventory_change","natgas_storage_change","refinery_utilisation",
                         "us_crude_production","crude_imports","strategic_reserve"],
        priority      = 95,
    ),

    "newsapi": SourceSpec(
        name          = "newsapi",
        tier          = SourceTier.FREE_KEY,
        rate_limit    = RateLimit(calls_per_day=100),
        env_key       = "NEWSAPI_KEY",
        register_url  = "https://newsapi.org/register",
        description   = "NewsAPI — global headlines, source-filtered news (100/day free dev tier)",
        asset_classes = ["all"],
        provides      = ["top_headlines","news_count_24h"],
        priority      = 70,
    ),
}


# ---------------------------------------------------------------------------
# Registry interface
# ---------------------------------------------------------------------------

def get_source(name: str) -> SourceSpec:
    if name not in SOURCES:
        raise KeyError(f"Unknown source '{name}'. Valid: {list(SOURCES)}")
    return SOURCES[name]


def available_sources(asset_class: str, env_vars: dict | None = None) -> list[SourceSpec]:
    """
    Return sources that are relevant for this asset_class AND have their
    API key available (if required). Sorted by priority descending.
    """
    import os
    env = env_vars or os.environ

    result = []
    for spec in SOURCES.values():
        # Check asset class relevance
        if "all" not in spec.asset_classes:
            if not any(asset_class.startswith(ac) for ac in spec.asset_classes):
                continue
        # Check API key availability
        if spec.env_key and not env.get(spec.env_key):
            continue
        # Check health
        if not spec.is_healthy:
            continue
        result.append(spec)

    return sorted(result, key=lambda s: s.priority, reverse=True)


def print_source_table() -> None:
    """Print a human-readable table of all sources and their status."""
    import os
    print(f"\n  {'Source':<18} {'Tier':<14} {'Limit':<20} {'Key set':<8} {'Register at'}")
    print(f"  {'─'*80}")
    for name, spec in SOURCES.items():
        key_set = "✓" if (not spec.env_key or os.environ.get(spec.env_key)) else "✗"
        if spec.rate_limit.calls_per_day:
            limit = f"{spec.rate_limit.calls_per_day}/day"
        elif spec.rate_limit.calls_per_minute:
            limit = f"{spec.rate_limit.calls_per_minute}/min"
        else:
            limit = "unlimited"
        print(f"  {name:<18} {spec.tier.value:<14} {limit:<20} {key_set:<8} {spec.register_url}")
    print()
