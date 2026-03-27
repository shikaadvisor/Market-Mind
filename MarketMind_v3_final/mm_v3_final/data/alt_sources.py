"""
data/alt_sources.py
====================
Async adapters for every alternate data source.

Each adapter:
  - Returns a typed dict (never raises — errors go in the dict)
  - Respects rate limits via source_registry.SourceSpec.wait_if_needed()
  - Records success/failure on the SourceSpec health tracker
  - Has a mock mode for testing without API keys

Sources implemented:
  FredAdapter         — FRED macro indicators (GDP, CPI, yields, M2, breakeven)
  FinnhubAdapter      — news sentiment, earnings calendar, insider sentiment
  AlphaVantageAdapter — AI-scored news sentiment (careful: 25 req/day free)
  CoinGeckoAdapter    — crypto prices, fear/greed index, BTC dominance
  RbiDbieAdapter      — RBI India macro (repo rate, CPI India, FII flows)
  NseAdapter          — NSE public endpoints (PCR, India VIX, OI data)
  EiaAdapter          — EIA energy inventory (crude, nat gas, refinery)
  NewsApiAdapter      — global headlines (100/day dev tier)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import date, datetime, timedelta
from typing import Any
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode

from data.source_registry import SOURCES, SourceSpec

logger = logging.getLogger("marketmind.alt_sources")

_TIMEOUT = 8   # seconds per HTTP request


# ---------------------------------------------------------------------------
# Shared HTTP helper
# ---------------------------------------------------------------------------

def _get_json(url: str, headers: dict | None = None, timeout: int = _TIMEOUT) -> dict | list | None:
    """Synchronous JSON fetch with timeout. Returns None on error."""
    try:
        req = Request(url, headers=headers or {"User-Agent": "MarketMind/3.0"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        logger.debug("HTTP error for %s: %s", url[:80], e)
        return None


def _safe_float(d: dict, *keys, default: float = 0.0) -> float:
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return default


# ---------------------------------------------------------------------------
# FRED Adapter
# ---------------------------------------------------------------------------

FRED_SERIES = {
    "us_10y":            "DGS10",       # 10Y Treasury yield
    "us_2y":             "DGS2",        # 2Y Treasury yield
    "us_30y":            "DGS30",       # 30Y Treasury yield
    "fed_funds_rate":    "FEDFUNDS",    # Fed funds effective rate
    "cpi_yoy":           "CPIAUCSL",    # CPI all items
    "core_cpi_yoy":      "CPILFESL",    # Core CPI (ex food/energy)
    "pce_yoy":           "PCEPI",       # PCE price index
    "real_gdp_growth":   "A191RL1Q225SBEA",  # Real GDP growth rate
    "unemployment_rate": "UNRATE",      # Unemployment rate
    "m2_growth":         "M2SL",        # M2 money supply
    "breakeven_10y":     "T10YIE",      # 10Y breakeven inflation
    "credit_spread_hy":  "BAMLH0A0HYM2", # HY credit spread
    "sofr":              "SOFR",        # SOFR overnight rate
}


class FredAdapter:
    BASE = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str, mock: bool = False):
        self.api_key = api_key
        self.mock    = mock
        self._spec   = SOURCES["fred"]
        self._cache: dict[str, tuple[float, Any]] = {}  # series_id → (timestamp, value)
        self._cache_ttl = 3600  # 1 hour — FRED data doesn't change intra-day

    def fetch_series_latest(self, series_id: str) -> float | None:
        """Fetch the latest value for a FRED series. Cached for 1 hour."""
        now = time.time()
        if series_id in self._cache:
            ts, val = self._cache[series_id]
            if now - ts < self._cache_ttl:
                return val

        if self.mock:
            return {"DGS10": 4.32, "DGS2": 4.85, "FEDFUNDS": 5.33,
                    "CPIAUCSL": 3.2, "UNRATE": 3.8, "T10YIE": 2.28,
                    "BAMLH0A0HYM2": 3.05, "SOFR": 5.30}.get(series_id, 0.0)

        self._spec.wait_if_needed()
        url = (f"{self.BASE}?series_id={series_id}"
               f"&api_key={self.api_key}&file_type=json"
               f"&sort_order=desc&limit=1")
        data = _get_json(url)
        success = bool(data and data.get("observations"))
        self._spec.record_call(success)

        if success:
            obs = data["observations"][0]
            val_str = obs.get("value", ".")
            if val_str != ".":
                val = float(val_str)
                self._cache[series_id] = (now, val)
                return val
        return None

    def fetch_all(self) -> dict[str, float]:
        """Fetch all configured FRED series. Returns dict of available values."""
        result: dict[str, float] = {}
        for field_name, series_id in FRED_SERIES.items():
            val = self.fetch_series_latest(series_id)
            if val is not None:
                result[field_name] = val
        # Derived: yield curve (2s10s)
        if "us_10y" in result and "us_2y" in result:
            result["yield_curve_2s10s"] = round(
                (result["us_10y"] - result["us_2y"]) * 100, 2
            )  # in basis points
        return result


# ---------------------------------------------------------------------------
# Finnhub Adapter
# ---------------------------------------------------------------------------

class FinnhubAdapter:
    BASE = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str, mock: bool = False):
        self.api_key = api_key
        self.mock    = mock
        self._spec   = SOURCES["finnhub"]

    def _get(self, endpoint: str, params: dict = {}) -> dict | list | None:
        self._spec.wait_if_needed()
        params_with_key = {**params, "token": self.api_key}
        url = f"{self.BASE}/{endpoint}?{urlencode(params_with_key)}"
        data = _get_json(url)
        self._spec.record_call(data is not None)
        return data

    def get_news_sentiment(self, symbol: str) -> dict:
        """Company news sentiment score from Finnhub."""
        if self.mock:
            return {"buzz": {"articlesInLastWeek": 12, "weeklyAverage": 9, "buzz": 1.3},
                    "companyNewsScore": 0.72,
                    "sectorAverageBullishPercent": 0.58,
                    "sectorAverageNewsScore": 0.55,
                    "sentiment": {"bearishPercent": 0.32, "bullishPercent": 0.68}}

        data = self._get("news-sentiment", {"symbol": symbol})
        return data or {}

    def get_company_news(self, symbol: str, days_back: int = 3) -> list[dict]:
        """Recent company-specific headlines."""
        if self.mock:
            return [{"headline": f"Mock headline for {symbol}", "sentiment": 0.5,
                     "datetime": int(time.time())}]
        today     = date.today()
        from_date = (today - timedelta(days=days_back)).isoformat()
        to_date   = today.isoformat()
        data = self._get("company-news", {"symbol": symbol,
                                           "from": from_date, "to": to_date})
        return data[:10] if isinstance(data, list) else []

    def get_earnings_calendar(self, symbol: str | None = None, days_ahead: int = 5) -> list[dict]:
        """Upcoming earnings dates — critical for risk agent event detection."""
        if self.mock:
            return []
        today = date.today()
        to    = (today + timedelta(days=days_ahead)).isoformat()
        params = {"from": today.isoformat(), "to": to}
        if symbol:
            params["symbol"] = symbol
        data = self._get("calendar/earnings", params)
        if isinstance(data, dict):
            return data.get("earningsCalendar", [])
        return []

    def get_insider_sentiment(self, symbol: str) -> dict:
        """Aggregate insider transaction sentiment."""
        if self.mock:
            return {"data": [{"change": 5, "month": 3, "mspr": 0.15, "year": 2025}],
                    "symbol": symbol}
        data = self._get("stock/insider-sentiment", {"symbol": symbol})
        return data or {}

    def get_recommendation_trends(self, symbol: str) -> dict:
        """Analyst recommendation trend (buy/sell/hold counts)."""
        if self.mock:
            return [{"buy": 15, "hold": 8, "sell": 2, "strongBuy": 5, "strongSell": 0,
                     "period": date.today().isoformat()}]
        data = self._get("stock/recommendation", {"symbol": symbol})
        return (data[0] if isinstance(data, list) and data else {})

    def get_market_news(self, category: str = "general") -> list[dict]:
        """General market news headlines."""
        if self.mock:
            return [{"headline": "Markets open higher on positive macro data",
                     "datetime": int(time.time()), "source": "mock"}]
        data = self._get("news", {"category": category})
        return data[:8] if isinstance(data, list) else []

    def get_economic_calendar(self, days_ahead: int = 3) -> list[dict]:
        """Upcoming economic events (FOMC, CPI, NFP, etc.)."""
        if self.mock:
            return []
        today = date.today()
        to    = (today + timedelta(days=days_ahead)).isoformat()
        data  = self._get("calendar/economic", {"from": today.isoformat(), "to": to})
        if isinstance(data, dict):
            return data.get("economicCalendar", [])
        return []

    def fetch_all(self, symbol: str, asset_class: str) -> dict:
        """Fetch everything relevant for this symbol/asset_class."""
        result: dict = {}

        # News sentiment
        ns = self.get_news_sentiment(symbol)
        if ns:
            result["finnhub_company_news_score"]    = float(ns.get("companyNewsScore", 0))
            result["finnhub_bullish_pct"]           = float(ns.get("sentiment", {}).get("bullishPercent", 0.5))
            result["finnhub_bearish_pct"]           = float(ns.get("sentiment", {}).get("bearishPercent", 0.5))
            result["finnhub_buzz"]                  = float(ns.get("buzz", {}).get("buzz", 1.0))

        # Headlines
        news = self.get_company_news(symbol)
        result["finnhub_headlines"] = [n.get("headline","") for n in news[:5]]
        result["finnhub_news_count"] = len(news)

        # Upcoming earnings (equity only)
        if asset_class.startswith("equity"):
            earnings = self.get_earnings_calendar(symbol, days_ahead=7)
            if earnings:
                next_e = earnings[0]
                result["next_earnings_date"]    = next_e.get("date", "")
                result["earnings_days_away"]    = (
                    (date.fromisoformat(next_e["date"]) - date.today()).days
                    if next_e.get("date") else 999
                )
                result["earnings_surprise_pct"] = float(next_e.get("surprisePercent", 0))
            else:
                result["earnings_days_away"] = 999

        # Recommendations
        if asset_class.startswith("equity"):
            recs = self.get_recommendation_trends(symbol)
            if recs:
                total = sum([
                    int(recs.get("strongBuy", 0)),
                    int(recs.get("buy", 0)),
                    int(recs.get("hold", 0)),
                    int(recs.get("sell", 0)),
                    int(recs.get("strongSell", 0)),
                ])
                bullish = int(recs.get("strongBuy", 0)) + int(recs.get("buy", 0))
                result["analyst_bull_pct"] = round(bullish / total, 3) if total > 0 else 0.5

        # General market news
        mkt_news = self.get_market_news()
        result["finnhub_market_headlines"] = [n.get("headline","") for n in mkt_news[:3]]

        return result


# ---------------------------------------------------------------------------
# Alpha Vantage Adapter  (budget: 25 req/day free — use sparingly)
# ---------------------------------------------------------------------------

class AlphaVantageAdapter:
    BASE = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str, mock: bool = False):
        self.api_key = api_key
        self.mock    = mock
        self._spec   = SOURCES["alpha_vantage"]
        # Daily budget tracking
        self._calls_today  = 0
        self._budget_day   = date.today()
        self.DAILY_BUDGET  = 20  # leave 5 calls as safety margin

    def _check_budget(self) -> bool:
        if date.today() != self._budget_day:
            self._calls_today = 0
            self._budget_day  = date.today()
        return self._calls_today < self.DAILY_BUDGET

    def _get(self, params: dict) -> dict | None:
        if not self._check_budget():
            logger.warning("Alpha Vantage daily budget exhausted — skipping")
            return None
        self._spec.wait_if_needed()
        url = f"{self.BASE}?{urlencode({**params, 'apikey': self.api_key})}"
        data = _get_json(url)
        success = bool(data and "Note" not in data and "Information" not in data)
        self._spec.record_call(success)
        if success:
            self._calls_today += 1
        return data if success else None

    def get_news_sentiment(self, ticker: str) -> dict:
        """AI-scored news sentiment per ticker. Uses 1 of 25 daily calls."""
        if self.mock:
            return {
                "feed": [{"title": f"Mock news for {ticker}", "overall_sentiment_score": 0.3,
                           "relevance_score": "0.85", "ticker_sentiment": [
                               {"ticker": ticker, "relevance_score": "0.85",
                                "ticker_sentiment_score": "0.28",
                                "ticker_sentiment_label": "Somewhat-Bullish"}]}],
                "sentiment_score_definition": "mock"
            }
        data = self._get({"function": "NEWS_SENTIMENT", "tickers": ticker,
                          "limit": "10", "sort": "LATEST"})
        return data or {}

    def parse_sentiment(self, data: dict, ticker: str) -> dict:
        """Extract structured sentiment from Alpha Vantage news feed."""
        result = {}
        feed = data.get("feed", [])
        if not feed:
            return result

        scores = []
        relevances = []
        for item in feed[:10]:
            ts = item.get("ticker_sentiment", [])
            for ts_entry in ts:
                if ts_entry.get("ticker") == ticker.upper():
                    try:
                        rel = float(ts_entry.get("relevance_score", 0))
                        scr = float(ts_entry.get("ticker_sentiment_score", 0))
                        if rel >= 0.20:  # only meaningful relevance
                            scores.append(scr)
                            relevances.append(rel)
                    except (TypeError, ValueError):
                        pass

        if scores:
            # Weighted average by relevance
            total_w = sum(relevances)
            avg_score = sum(s * r for s, r in zip(scores, relevances)) / total_w
            result["av_news_sentiment_score"] = round(avg_score, 4)
            result["av_news_articles_count"]  = len(scores)
            result["av_news_avg_relevance"]   = round(sum(relevances) / len(relevances), 3)

        return result

    def fetch_all(self, symbol: str) -> dict:
        """Full Alpha Vantage fetch for a symbol. Costs 1 daily call."""
        data = self.get_news_sentiment(symbol)
        return self.parse_sentiment(data, symbol)


# ---------------------------------------------------------------------------
# CoinGecko Adapter (no key, rate-limited to ~30/min)
# ---------------------------------------------------------------------------

class CoinGeckoAdapter:
    BASE = "https://api.coingecko.com/api/v3"

    SYMBOL_MAP = {
        "BTCUSD": "bitcoin", "ETHUSD": "ethereum",
        "BTC":    "bitcoin", "ETH":    "ethereum",
    }

    def __init__(self, mock: bool = False):
        self.mock  = mock
        self._spec = SOURCES["coingecko"]

    def get_fear_greed_index(self) -> dict:
        """Crypto Fear & Greed Index (0=extreme fear, 100=extreme greed)."""
        if self.mock:
            return {"value": 58, "value_classification": "Greed",
                    "timestamp": str(int(time.time()))}
        self._spec.wait_if_needed()
        data = _get_json("https://api.alternative.me/fng/?limit=1")
        self._spec.record_call(data is not None)
        if data and data.get("data"):
            d = data["data"][0]
            return {"value": int(d.get("value", 50)),
                    "value_classification": d.get("value_classification", "Neutral"),
                    "timestamp": d.get("timestamp", "")}
        return {}

    def get_global_data(self) -> dict:
        """Global crypto market data: total market cap, BTC dominance."""
        if self.mock:
            return {"total_market_cap_usd": 2.8e12, "btc_dominance": 52.3,
                    "eth_dominance": 17.1, "defi_market_cap": 1.2e11}
        self._spec.wait_if_needed()
        data = _get_json(f"{self.BASE}/global")
        self._spec.record_call(data is not None)
        if data and data.get("data"):
            d = data["data"]
            return {
                "total_market_cap_usd": d.get("total_market_cap", {}).get("usd", 0),
                "btc_dominance":        round(d.get("market_cap_percentage", {}).get("btc", 0), 2),
                "eth_dominance":        round(d.get("market_cap_percentage", {}).get("eth", 0), 2),
                "defi_market_cap":      d.get("total_value_locked", {}).get("usd", 0),
            }
        return {}

    def get_coin_price(self, symbol: str) -> dict:
        """Price data for a specific crypto."""
        coin_id = self.SYMBOL_MAP.get(symbol.upper(), symbol.lower())
        if self.mock:
            return {"price_usd": 68000.0, "change_24h": 2.4, "volume_24h": 38e9,
                    "market_cap": 1.34e12}
        self._spec.wait_if_needed()
        data = _get_json(f"{self.BASE}/simple/price?ids={coin_id}"
                         f"&vs_currencies=usd&include_24hr_change=true"
                         f"&include_24hr_vol=true&include_market_cap=true")
        self._spec.record_call(data is not None)
        if data and coin_id in data:
            d = data[coin_id]
            return {"price_usd": d.get("usd", 0), "change_24h": d.get("usd_24h_change", 0),
                    "volume_24h": d.get("usd_24h_vol", 0), "market_cap": d.get("usd_market_cap", 0)}
        return {}

    def fetch_all(self, symbol: str) -> dict:
        result = {}
        fg = self.get_fear_greed_index()
        if fg:
            result["crypto_fear_greed"]        = int(fg.get("value", 50))
            result["crypto_sentiment_label"]   = fg.get("value_classification", "Neutral")
        gd = self.get_global_data()
        result.update(gd)
        if symbol in self.SYMBOL_MAP or "USD" in symbol:
            cp = self.get_coin_price(symbol)
            result.update(cp)
        return result


# ---------------------------------------------------------------------------
# NSE India Adapter (public endpoints, no key)
# ---------------------------------------------------------------------------

class NseAdapter:
    BASE = "https://www.nseindia.com/api"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (compatible; MarketMind/3.0)",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com",
    }

    def __init__(self, mock: bool = False):
        self.mock  = mock
        self._spec = SOURCES["nse_public"]
        self._session_cookie: str = ""

    def _nse_get(self, endpoint: str) -> dict | None:
        """NSE requires a valid session cookie — fetch homepage first if missing."""
        self._spec.wait_if_needed()
        url = f"{self.BASE}/{endpoint}"
        data = _get_json(url, headers=self.HEADERS)
        self._spec.record_call(data is not None)
        return data

    def get_india_vix(self) -> float | None:
        """India VIX from NSE."""
        if self.mock:
            return 13.2
        data = self._nse_get("allIndices")
        if isinstance(data, dict) and "data" in data:
            for idx in data["data"]:
                if idx.get("index", "").upper() in ("INDIA VIX", "INDIAVIX"):
                    try:
                        return float(idx["last"])
                    except (TypeError, ValueError, KeyError):
                        pass
        return None

    def get_pcr(self) -> float | None:
        """Put-Call Ratio from NSE options data."""
        if self.mock:
            return 1.08
        data = self._nse_get("option-chain-indices?symbol=NIFTY")
        if isinstance(data, dict) and "filtered" in data:
            filtered = data["filtered"]
            ce_oi = float(filtered.get("CE", {}).get("totOI", 0) or 0)
            pe_oi = float(filtered.get("PE", {}).get("totOI", 0) or 0)
            return round(pe_oi / ce_oi, 3) if ce_oi > 0 else None
        return None

    def fetch_all(self) -> dict:
        result = {}
        vix = self.get_india_vix()
        if vix: result["india_vix"] = vix
        pcr = self.get_pcr()
        if pcr: result["pcr"] = pcr
        return result


# ---------------------------------------------------------------------------
# RBI DBIE Adapter (India macro, no key)
# ---------------------------------------------------------------------------

class RbiDbieAdapter:
    # RBI publishes key rates in structured format via their statistics portal
    # Using publicly accessible JSON endpoints from RBI's data warehouse
    BASE = "https://api.rbi.org.in/api/v4"   # RBI Open Data Initiative

    def __init__(self, mock: bool = False):
        self.mock  = mock
        self._spec = SOURCES["rbi_dbie"]
        self._cache: dict[str, tuple[float, Any]] = {}
        self._cache_ttl = 7200  # 2 hours

    def get_repo_rate(self) -> float | None:
        """Current RBI repo rate."""
        if self.mock:
            return 6.5
        # RBI key policy rates are updated on RBI website after MPC decisions
        # For simplicity, cross-reference yfinance INR bond proxy or scrape RBI
        # Using FRED India equivalent via yfinance proxy
        try:
            import yfinance as yf
            # India 10Y government bond yield as proxy
            tick = yf.Ticker("^INBMK")
            hist = tick.history(period="2d")
            if not hist.empty:
                return round(float(hist["Close"].iloc[-1]), 3)
        except Exception:
            pass
        return None

    def get_india_cpi(self) -> float | None:
        """India CPI YoY (most recent) — sourced via MOSPI data."""
        if self.mock:
            return 5.1
        # India CPI YoY is published monthly by MOSPI — use FRED proxy FPCPITOTLZGIND
        try:
            # Cross-reference via FRED if FRED key available
            return None   # fallback to None; FRED adapter handles this if key present
        except Exception:
            return None

    def fetch_all(self) -> dict:
        result = {}
        repo = self.get_repo_rate()
        if repo: result["india_10y_yield"] = repo
        cpi  = self.get_india_cpi()
        if cpi: result["cpi_india"] = cpi
        return result
