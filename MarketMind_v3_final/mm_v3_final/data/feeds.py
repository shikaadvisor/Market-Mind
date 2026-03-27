"""
data/feeds.py
=============
Global market data feed — universe-aware, asset-class-smart.

Single YFinanceFeed handles every instrument in the universe.
yfinance covers equities, indices, commodities, FX, crypto, bonds
across all major exchanges worldwide — all through one interface.

FeedSnapshot is now fully global:
  - No INR-hardcoded fields. All flows are normalised to USD.
  - asset_class field drives which enrichments are applied.
  - Commodity-specific fields (inventory, production data)
  - FX-specific fields (carry, interest rate differential)
  - Bond-specific fields (yield, duration proxy)
  - Volatility surface fields (implied vol, term structure proxy)

Enrichment sources (all free, no auth):
  - yfinance: price, volume, technicals for everything
  - EIA API: crude/gas inventory (free, needs EIA_API_KEY)
  - FRED API: macro indicators (free, needs FRED_API_KEY)
  - NewsAPI: headlines (free tier, needs NEWSAPI_KEY)

All API keys are optional — system degrades gracefully without them.
"""

from __future__ import annotations

import json
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Universal FeedSnapshot
# ---------------------------------------------------------------------------

@dataclass
class FeedSnapshot:
    """
    Normalised market data snapshot — works for ALL asset classes.
    Fields not applicable to an asset class default to 0.0 / empty.
    Check snap.asset_class to know which fields are populated.
    """

    # Identity
    instrument:     str
    exchange:       str
    asset_class:    str            # from universe.AssetClass
    currency:       str            # price denomination
    timestamp:      datetime

    # ── Universal price fields (always populated) ─────────────────────────
    spot:           float = 0.0
    prev_close:     float = 0.0
    open_price:     float = 0.0
    high:           float = 0.0
    low:            float = 0.0
    change_pct:     float = 0.0    # % change from prev close
    week_change_pct:  float = 0.0
    month_change_pct: float = 0.0
    ytd_change_pct:   float = 0.0

    # ── Volume and breadth ────────────────────────────────────────────────
    volume:         float = 0.0
    volume_ratio:   float = 1.0    # today / 20-day avg
    open_interest:  float = 0.0    # futures OI

    # ── Technical levels ──────────────────────────────────────────────────
    ma_20:          float = 0.0
    ma_50:          float = 0.0
    ma_200:         float = 0.0
    week_52_high:   float = 0.0
    week_52_low:    float = 0.0
    rsi_14:         float = 50.0
    atr_14:         float = 0.0    # Average True Range (absolute)
    atr_pct:        float = 0.0    # ATR as % of price — vol normalised

    # ── Volatility surface ────────────────────────────────────────────────
    implied_vol:    float = 0.0    # generic implied vol if available
    hist_vol_20:    float = 0.0    # 20-day realised vol (annualised %)
    vol_regime:     str   = ""     # "low" | "medium" | "high" — derived

    # ── Equity-specific ───────────────────────────────────────────────────
    equity_vix:     float = 0.0    # local fear index (VIX, INDIAVIX, VSTOXX)
    pcr:            float = 1.0    # put-call ratio (index options)
    advance_decline: float = 1.0   # breadth ratio
    pe_ratio:       float = 0.0

    # ── Commodity-specific ────────────────────────────────────────────────
    inventory_change: float = 0.0  # weekly change (barrels, bu, etc.)
    inventory_vs_avg: float = 0.0  # % vs 5-year seasonal avg
    seasonal_factor:  float = 0.0  # seasonal index (1.0 = neutral)
    basis:            float = 0.0  # spot - nearest future

    # ── FX-specific ───────────────────────────────────────────────────────
    rate_differential: float = 0.0  # interest rate spread (% p.a.)
    carry_return_ann:  float = 0.0  # annualised carry return
    real_rate:         float = 0.0  # nominal yield - inflation

    # ── Bond/rates-specific ───────────────────────────────────────────────
    yield_value:    float = 0.0    # yield in % (for bond instruments)
    yield_2y:       float = 0.0    # 2Y yield (for curve context)
    yield_10y:      float = 0.0    # 10Y yield
    yield_30y:      float = 0.0    # 30Y yield
    curve_2s10s:    float = 0.0    # 2s10s spread (bp)

    # ── Global macro context (always enriched) ────────────────────────────
    dxy:            float = 104.0  # US Dollar Index
    us_10y:         float = 4.5    # US 10-year yield
    vix:            float = 18.0   # CBOE VIX (global risk barometer)
    gold_usd:       float = 2000.0 # Gold price (risk-off indicator)
    crude_wti:      float = 75.0   # WTI crude (global growth proxy)

    # ── Regional flows (FX-normalised to USD) ─────────────────────────────
    # Positive = inflows, negative = outflows
    institutional_flow_usd: float = 0.0   # latest institutional net flow
    retail_flow_proxy:       float = 0.0   # estimated retail positioning

    # ── News and sentiment ────────────────────────────────────────────────
    top_headlines:            list[str] = field(default_factory=list)
    news_sentiment_score:     float = 0.0   # -1 to +1
    social_sentiment_score:   float = 0.0   # -1 to +1 (if available)

    # ── Metadata ─────────────────────────────────────────────────────────
    source:         str  = "yfinance"
    is_stale:       bool = False
    errors:         list[str] = field(default_factory=list)

    # ── Derived properties ────────────────────────────────────────────────

    @property
    def is_above_200ma(self) -> bool:
        return self.spot > self.ma_200 > 0

    @property
    def distance_from_52w_high_pct(self) -> float:
        if self.week_52_high <= 0:
            return 0.0
        return round((self.spot - self.week_52_high) / self.week_52_high * 100, 2)

    @property
    def distance_from_52w_low_pct(self) -> float:
        if self.week_52_low <= 0:
            return 0.0
        return round((self.spot - self.week_52_low) / self.week_52_low * 100, 2)

    def to_market_data_dict(self) -> dict:
        """
        Convert to the market_data dict the desk agents receive.
        Asset-class-aware: includes relevant fields, excludes noise.
        """
        base = {
            "instrument":       self.instrument,
            "asset_class":      self.asset_class,
            "currency":         self.currency,
            "spot":             self.spot,
            "change_pct":       self.change_pct,
            "week_change_pct":  self.week_change_pct,
            "month_change_pct": self.month_change_pct,
            "ytd_change_pct":   self.ytd_change_pct,
            "volume_ratio":     self.volume_ratio,
            "ma_20":            self.ma_20,
            "ma_50":            self.ma_50,
            "ma_200":           self.ma_200,
            "week_52_high":     self.week_52_high,
            "week_52_low":      self.week_52_low,
            "rsi_14":           self.rsi_14,
            "atr_pct":          self.atr_pct,
            "hist_vol_20":      self.hist_vol_20,
            "dist_from_52w_high": self.distance_from_52w_high_pct,
            "dist_from_52w_low":  self.distance_from_52w_low_pct,
            # Global macro always
            "dxy":              self.dxy,
            "us_10y":           self.us_10y,
            "vix":              self.vix,
            "gold_usd":         self.gold_usd,
            "crude_wti":        self.crude_wti,
        }

        ac = self.asset_class
        if ac.startswith("equity"):
            base.update({
                "equity_vix":    self.equity_vix,
                "pcr":           self.pcr,
                "advance_decline": self.advance_decline,
                "institutional_flow": self.institutional_flow_usd,
                "pe_ratio":      self.pe_ratio,
            })
        elif ac.startswith("commodity"):
            base.update({
                "open_interest": self.open_interest,
                "inventory_change": self.inventory_change,
                "inventory_vs_avg": self.inventory_vs_avg,
                "seasonal_factor":  self.seasonal_factor,
                "basis":         self.basis,
            })
        elif ac.startswith("fx"):
            base.update({
                "rate_differential": self.rate_differential,
                "carry_return_ann":  self.carry_return_ann,
                "real_rate":         self.real_rate,
            })
        elif ac == "fixed_income":
            base.update({
                "yield_value":   self.yield_value,
                "yield_2y":      self.yield_2y,
                "yield_10y":     self.yield_10y,
                "curve_2s10s":   self.curve_2s10s,
            })

        return base


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class MarketDataFeed(ABC):
    @abstractmethod
    def fetch(self, symbol: str) -> FeedSnapshot: ...

    def fetch_safe(self, symbol: str) -> FeedSnapshot:
        try:
            return self.fetch(symbol)
        except Exception as e:
            from data.universe import get_instrument
            try:
                spec = get_instrument(symbol)
                ac, ccy, exch = spec.asset_class, spec.currency, spec.exchange
            except Exception:
                ac, ccy, exch = "equity_index", "USD", "UNKNOWN"
            return FeedSnapshot(
                instrument=symbol, exchange=exch,
                asset_class=ac, currency=ccy,
                timestamp=datetime.now(), is_stale=True,
                errors=[str(e)], source=self.__class__.__name__,
            )


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _http_get(url: str, headers: dict | None = None, timeout: int = 12) -> dict | list | None:
    try:
        req = Request(url, headers=headers or {"User-Agent": "MarketMind/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# RSI and volatility helpers
# ---------------------------------------------------------------------------

def _rsi(closes, period=14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains  = [max(0, d) for d in deltas[-period:]]
    losses = [-min(0, d) for d in deltas[-period:]]
    avg_g  = sum(gains) / period
    avg_l  = sum(losses) / period
    if avg_l == 0:
        return 100.0
    return round(100 - 100 / (1 + avg_g / avg_l), 2)


def _hist_vol(closes, period=20) -> float:
    """Annualised historical volatility (%)."""
    import math
    if len(closes) < period + 1:
        return 0.0
    returns = [math.log(closes[i] / closes[i-1])
               for i in range(len(closes)-period, len(closes))
               if closes[i-1] > 0]
    if not returns:
        return 0.0
    mean   = sum(returns) / len(returns)
    var    = sum((r - mean) ** 2 for r in returns) / len(returns)
    return round(math.sqrt(var) * math.sqrt(252) * 100, 2)


def _atr(highs, lows, closes, period=14) -> float:
    """Average True Range over period."""
    if len(closes) < period + 1:
        return 0.0
    trs = []
    for i in range(1, min(period + 1, len(closes))):
        tr = max(
            highs[-i] - lows[-i],
            abs(highs[-i] - closes[-(i+1)]),
            abs(lows[-i]  - closes[-(i+1)]),
        )
        trs.append(tr)
    return round(sum(trs) / len(trs), 4) if trs else 0.0


# ---------------------------------------------------------------------------
# YFinance Feed — handles the entire universe
# ---------------------------------------------------------------------------

class YFinanceFeed(MarketDataFeed):
    """
    Universal feed using yfinance.
    Covers: equity indices, stocks, commodities (futures), FX, crypto, bonds.
    All through one interface, all for free.
    """

    def __init__(self):
        self._yf = None

    def _yf_lib(self):
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
            except ImportError:
                raise ImportError("pip install yfinance")
        return self._yf

    def fetch(self, symbol: str) -> FeedSnapshot:
        from data.universe import get_instrument, UNIVERSE

        # Resolve symbol to spec
        sym_upper = symbol.upper()
        spec = get_instrument(sym_upper) if sym_upper in UNIVERSE else None
        yf_sym = spec.yf_symbol if spec else f"{symbol}.NS"

        yf = self._yf_lib()
        ticker = yf.Ticker(yf_sym)

        # Download 1 year of daily data for full technical picture
        hist = ticker.history(period="1y", interval="1d", auto_adjust=True)

        if hist.empty:
            raise ValueError(f"No data returned for {yf_sym}")

        closes  = hist["Close"].dropna().values
        highs   = hist["High"].dropna().values
        lows    = hist["Low"].dropna().values
        volumes = hist["Volume"].dropna().values if "Volume" in hist.columns else []

        spot       = float(closes[-1])
        prev_close = float(closes[-2]) if len(closes) >= 2 else spot
        open_p     = float(hist["Open"].iloc[-1])
        high_p     = float(hist["High"].iloc[-1])
        low_p      = float(hist["Low"].iloc[-1])

        chg_pct  = (spot - prev_close) / prev_close * 100 if prev_close else 0.0
        wk_chg   = (spot - closes[-6])  / closes[-6]  * 100 if len(closes) > 5  else 0.0
        mo_chg   = (spot - closes[-22]) / closes[-22] * 100 if len(closes) > 21 else 0.0

        # Year start for YTD
        ytd_start = closes[0] if len(closes) > 0 else spot
        ytd_chg   = (spot - ytd_start) / ytd_start * 100 if ytd_start else 0.0

        # MAs
        ma20  = float(closes[-20:].mean())  if len(closes) >= 20  else float(closes.mean())
        ma50  = float(closes[-50:].mean())  if len(closes) >= 50  else float(closes.mean())
        ma200 = float(closes[-200:].mean()) if len(closes) >= 200 else float(closes.mean())

        # 52-week range
        w52_high = float(closes[-252:].max()) if len(closes) >= 252 else float(closes.max())
        w52_low  = float(closes[-252:].min()) if len(closes) >= 252 else float(closes.min())

        # Technicals
        rsi  = _rsi(list(closes), 14)
        hvol = _hist_vol(list(closes), 20)
        atr  = _atr(list(highs), list(lows), list(closes), 14)
        atr_pct = round(atr / spot * 100, 3) if spot > 0 else 0.0

        # Volume ratio
        if len(volumes) > 20:
            vol_today = float(volumes[-1])
            vol_avg20 = float(volumes[-21:-1].mean()) if hasattr(volumes, 'mean') else sum(volumes[-21:-1]) / 20
            vol_ratio = vol_today / vol_avg20 if vol_avg20 > 0 else 1.0
        else:
            vol_today = float(volumes[-1]) if len(volumes) > 0 else 0.0
            vol_ratio = 1.0

        # Vol regime classification
        if hvol < 10:
            vol_regime_str = "low"
        elif hvol < 25:
            vol_regime_str = "medium"
        else:
            vol_regime_str = "high"

        # Build snapshot
        snap = FeedSnapshot(
            instrument       = sym_upper,
            exchange         = spec.exchange if spec else "UNKNOWN",
            asset_class      = spec.asset_class if spec else "equity_index",
            currency         = spec.currency if spec else "USD",
            timestamp        = datetime.now(),
            spot             = round(spot, 6),
            prev_close       = round(prev_close, 6),
            open_price       = round(open_p, 6),
            high             = round(high_p, 6),
            low              = round(low_p, 6),
            change_pct       = round(chg_pct, 3),
            week_change_pct  = round(wk_chg, 3),
            month_change_pct = round(mo_chg, 3),
            ytd_change_pct   = round(ytd_chg, 3),
            volume           = round(vol_today, 0),
            volume_ratio     = round(vol_ratio, 2),
            ma_20            = round(ma20, 6),
            ma_50            = round(ma50, 6),
            ma_200           = round(ma200, 6),
            week_52_high     = round(w52_high, 6),
            week_52_low      = round(w52_low, 6),
            rsi_14           = rsi,
            atr_14           = round(atr, 6),
            atr_pct          = atr_pct,
            hist_vol_20      = hvol,
            vol_regime       = vol_regime_str,
            source           = "yfinance",
        )

        # Asset-class-specific enrichment
        if spec:
            self._enrich(snap, spec, ticker)

        return snap

    def _enrich(self, snap: FeedSnapshot, spec, ticker) -> None:
        """Add asset-class-specific fields."""
        ac = spec.asset_class

        if ac == "fixed_income":
            # Yield instruments — spot IS the yield
            snap.yield_value = snap.spot

        elif ac == "volatility":
            # Vol instruments — spot IS the vol
            snap.implied_vol = snap.spot


    def fetch_macro_context(self) -> dict:
        """
        Fetch the 5 global macro anchors that every signal needs.
        DXY, US10Y, VIX, Gold, WTI Crude.
        """
        symbols = {
            "dxy":      "DX-Y.NYB",
            "us_10y":   "^TNX",
            "vix":      "^VIX",
            "gold_usd": "GC=F",
            "crude_wti": "CL=F",
        }
        result = {}
        yf = self._yf_lib()
        for key, sym in symbols.items():
            try:
                t = yf.Ticker(sym)
                h = t.history(period="2d", interval="1d")
                result[key] = round(float(h["Close"].iloc[-1]), 4) if not h.empty else 0.0
            except Exception:
                defaults = {"dxy": 104.0, "us_10y": 4.5, "vix": 18.0,
                            "gold_usd": 2000.0, "crude_wti": 75.0}
                result[key] = defaults[key]
        return result

    def enrich_with_macro(self, snap: FeedSnapshot) -> FeedSnapshot:
        """Inject global macro context into any snapshot."""
        macro = self.fetch_macro_context()
        snap.dxy       = macro.get("dxy",       snap.dxy)
        snap.us_10y    = macro.get("us_10y",    snap.us_10y)
        snap.vix       = macro.get("vix",       snap.vix)
        snap.gold_usd  = macro.get("gold_usd",  snap.gold_usd)
        snap.crude_wti = macro.get("crude_wti", snap.crude_wti)
        return snap


# ---------------------------------------------------------------------------
# EIA Feed (crude/gas inventory — free with key)
# ---------------------------------------------------------------------------

class EIAFeed:
    """
    US Energy Information Administration API.
    Provides weekly crude oil and natural gas inventory data.
    Free API key at: https://www.eia.gov/opendata/register.php
    """

    BASE = "https://api.eia.gov/v2"

    def __init__(self, api_key: str = ""):
        import os
        self.api_key = api_key or os.environ.get("EIA_API_KEY", "")

    def fetch_crude_inventory(self) -> dict:
        """Returns {'change_mmbbl': X, 'total_mmbbl': Y, 'vs_5yr_avg': Z%}"""
        if not self.api_key:
            return {}
        url = (f"{self.BASE}/petroleum/stoc/wstk/data/?frequency=weekly"
               f"&data[0]=value&sort[0][column]=period&sort[0][direction]=desc"
               f"&offset=0&length=2&api_key={self.api_key}")
        data = _http_get(url)
        if not data:
            return {}
        try:
            rows   = data.get("response", {}).get("data", [])
            latest = float(rows[0]["value"]) if rows else 0.0
            prev   = float(rows[1]["value"]) if len(rows) > 1 else latest
            return {
                "total_mmbbl":  round(latest, 1),
                "change_mmbbl": round(latest - prev, 1),
            }
        except Exception:
            return {}

    def fetch_gas_storage(self) -> dict:
        """Returns {'change_bcf': X, 'total_bcf': Y}"""
        if not self.api_key:
            return {}
        url = (f"{self.BASE}/natural-gas/stor/wkly/data/?frequency=weekly"
               f"&data[0]=value&sort[0][column]=period&sort[0][direction]=desc"
               f"&offset=0&length=2&api_key={self.api_key}")
        data = _http_get(url)
        if not data:
            return {}
        try:
            rows   = data.get("response", {}).get("data", [])
            latest = float(rows[0]["value"]) if rows else 0.0
            prev   = float(rows[1]["value"]) if len(rows) > 1 else latest
            return {
                "total_bcf":  round(latest, 1),
                "change_bcf": round(latest - prev, 1),
            }
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# FRED Feed (macro indicators — free with key)
# ---------------------------------------------------------------------------

class FREDFeed:
    """
    St. Louis Federal Reserve FRED API.
    Macro indicators: CPI, PCE, NFP, unemployment, yield curve, etc.
    Free API key at: https://fred.stlouisfed.org/docs/api/api_key.html
    """

    BASE = "https://api.stlouisfed.org/fred/series/observations"

    SERIES = {
        "us_cpi_yoy":    "CPIAUCSL",      # US CPI YoY
        "us_pce":        "PCEPI",          # PCE deflator
        "us_unemployment": "UNRATE",       # US unemployment rate
        "us_10y_yield":  "DGS10",          # 10Y Treasury yield
        "us_2y_yield":   "DGS2",           # 2Y Treasury yield
        "us_hys_spread": "BAMLH0A0HYM2",   # HY credit spread
        "em_spread":     "BAMLH0A0HYM2EY", # EM spread proxy
        "dxy":           "DTWEXBGS",        # Trade-weighted dollar
    }

    def __init__(self, api_key: str = ""):
        import os
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")

    def fetch(self, series_id: str, n: int = 2) -> list[float]:
        if not self.api_key:
            return []
        url = (f"{self.BASE}?series_id={series_id}"
               f"&api_key={self.api_key}&file_type=json"
               f"&sort_order=desc&limit={n}")
        data = _http_get(url)
        if not data:
            return []
        try:
            obs = data.get("observations", [])
            return [float(o["value"]) for o in obs if o["value"] != "."]
        except Exception:
            return []

    def fetch_yield_curve(self) -> dict:
        y2  = self.fetch("DGS2",  1)
        y10 = self.fetch("DGS10", 1)
        y30 = self.fetch("DGS30", 1)
        result = {
            "yield_2y":    y2[0]  if y2  else 0.0,
            "yield_10y":   y10[0] if y10 else 0.0,
            "yield_30y":   y30[0] if y30 else 0.0,
        }
        if result["yield_2y"] and result["yield_10y"]:
            result["curve_2s10s"] = round(
                (result["yield_10y"] - result["yield_2y"]) * 100, 1
            )
        return result


# ---------------------------------------------------------------------------
# NewsAPI Feed
# ---------------------------------------------------------------------------

class NewsApiFeed:
    """
    NewsAPI.org headline sentiment.
    Free tier: 100 req/day. Needs NEWSAPI_KEY.
    """

    BASE = "https://newsapi.org/v2"

    BEARISH = [
        "crash", "fall", "drop", "decline", "sell", "loss", "fear", "panic",
        "recession", "default", "crisis", "plunge", "tumble", "slump",
        "inflation", "rate hike", "hawkish", "outflow", "selling", "weak",
        "disappoints", "misses", "concern", "risk", "war", "sanctions",
    ]
    BULLISH = [
        "rally", "gain", "rise", "surge", "buy", "profit", "confidence",
        "growth", "recovery", "strong", "beat", "above expectations",
        "inflow", "upgrade", "dovish", "rate cut", "stimulus", "bullish",
        "record", "high", "optimism", "deal", "ceasefire", "supply cut",
    ]

    def __init__(self, api_key: str = ""):
        import os
        self.api_key = api_key or os.environ.get("NEWSAPI_KEY", "")

    def fetch(self, query: str, n: int = 20) -> tuple[list[str], float]:
        if not self.api_key:
            return [], 0.0
        from_dt = (datetime.utcnow() - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S")
        params  = urlencode({
            "q": query, "from": from_dt, "sortBy": "publishedAt",
            "language": "en", "pageSize": n, "apiKey": self.api_key,
        })
        data = _http_get(f"{self.BASE}/everything?{params}")
        if not data or data.get("status") != "ok":
            return [], 0.0
        headlines = [
            a["title"] for a in data.get("articles", [])[:n]
            if a.get("title") and "[Removed]" not in a.get("title", "")
        ]
        return headlines, self._score(headlines)

    def _score(self, headlines: list[str]) -> float:
        if not headlines:
            return 0.0
        total = sum(
            sum(1 for w in self.BULLISH if w in h.lower()) -
            sum(1 for w in self.BEARISH if w in h.lower())
            for h in headlines
        )
        return round(max(-1.0, min(1.0, total / (len(headlines) * 3))), 3)


# ---------------------------------------------------------------------------
# Composite global feed
# ---------------------------------------------------------------------------

class GlobalFeed(MarketDataFeed):
    """
    Main feed class. Combines yfinance + optional enrichment sources.

    Usage:
        feed = GlobalFeed(
            eia_key     = "...",   # optional
            fred_key    = "...",   # optional
            newsapi_key = "...",   # optional
        )
        snap = feed.fetch("CL")      # WTI Crude
        snap = feed.fetch("EURUSD")  # FX pair
        snap = feed.fetch("GC")      # Gold futures
        snap = feed.fetch("SPX")     # S&P 500
    """

    def __init__(
        self,
        eia_key:     str = "",
        fred_key:    str = "",
        newsapi_key: str = "",
        enrich_macro: bool = True,
    ):
        self.yf      = YFinanceFeed()
        self.eia     = EIAFeed(api_key=eia_key)
        self.fred    = FREDFeed(api_key=fred_key)
        self.news    = NewsApiFeed(api_key=newsapi_key)
        self.enrich_macro = enrich_macro
        self._macro_cache: dict = {}
        self._macro_cache_ts: float = 0.0

    def fetch(self, symbol: str) -> FeedSnapshot:
        from data.universe import get_instrument, UNIVERSE

        snap = self.yf.fetch_safe(symbol)

        # Global macro context (cached for 5 minutes)
        if self.enrich_macro:
            now = time.time()
            if now - self._macro_cache_ts > 300:
                try:
                    self._macro_cache = self.yf.fetch_macro_context()
                    self._macro_cache_ts = now
                except Exception:
                    pass
            if self._macro_cache:
                snap.dxy       = self._macro_cache.get("dxy",       snap.dxy)
                snap.us_10y    = self._macro_cache.get("us_10y",    snap.us_10y)
                snap.vix       = self._macro_cache.get("vix",       snap.vix)
                snap.gold_usd  = self._macro_cache.get("gold_usd",  snap.gold_usd)
                snap.crude_wti = self._macro_cache.get("crude_wti", snap.crude_wti)

        # Asset-class enrichment
        ac = snap.asset_class
        try:
            if ac.startswith("commodity_energy"):
                self._enrich_energy(snap, symbol)
            elif ac == "fixed_income":
                self._enrich_rates(snap)
            elif ac.startswith("fx"):
                self._enrich_fx(snap, symbol)
        except Exception as e:
            snap.errors.append(f"Enrichment error ({ac}): {e}")

        # News headlines for all instruments
        try:
            from data.universe import get_instrument
            spec   = get_instrument(symbol.upper()) if symbol.upper() in UNIVERSE else None
            query  = f"{snap.instrument} {spec.name if spec else symbol} market"
            headlines, score = self.news.fetch(query=query, n=15)
            snap.top_headlines        = headlines
            snap.news_sentiment_score = score
        except Exception as e:
            snap.errors.append(f"News: {e}")

        return snap

    def _enrich_energy(self, snap: FeedSnapshot, symbol: str) -> None:
        if symbol.upper() in ("CL", "BZ"):
            data = self.eia.fetch_crude_inventory()
            if data:
                snap.inventory_change = data.get("change_mmbbl", 0.0)
        elif symbol.upper() == "NG":
            data = self.eia.fetch_gas_storage()
            if data:
                snap.inventory_change = data.get("change_bcf", 0.0)

    def _enrich_rates(self, snap: FeedSnapshot) -> None:
        curve = self.fred.fetch_yield_curve()
        if curve:
            snap.yield_2y    = curve.get("yield_2y",    snap.yield_2y)
            snap.yield_10y   = curve.get("yield_10y",   snap.yield_10y)
            snap.yield_30y   = curve.get("yield_30y",   snap.yield_30y)
            snap.curve_2s10s = curve.get("curve_2s10s", snap.curve_2s10s)

    def _enrich_fx(self, snap: FeedSnapshot, symbol: str) -> None:
        # Approximate rate differential using short rates
        if "USD" in symbol.upper():
            rates = self.fred.fetch("DGS2", 1)
            if rates:
                snap.rate_differential = round(rates[0] - 1.5, 2)

    def is_market_open(self, symbol: str = "SPX") -> bool:
        from data.universe import is_instrument_tradeable
        try:
            return is_instrument_tradeable(symbol)
        except Exception:
            return True


# ---------------------------------------------------------------------------
# Broker feed stub (Zerodha / IBKR)
# ---------------------------------------------------------------------------

class BrokerFeed(MarketDataFeed):
    """
    Stub for broker API feeds (Zerodha Kite / IBKR TWS).
    Falls back to GlobalFeed until wired.

    To implement Zerodha:
        pip install kiteconnect
        Override connect() and fetch()

    To implement IBKR:
        pip install ib_insync
        Override connect() and fetch()
    """

    def __init__(self, broker: str = "zerodha", **kwargs):
        self.broker    = broker
        self._connected = False
        self._fallback = GlobalFeed(**kwargs)

    def connect(self, **credentials) -> bool:
        print(f"  [BrokerFeed] {self.broker} not yet wired. Using GlobalFeed fallback.")
        return False

    def fetch(self, symbol: str) -> FeedSnapshot:
        if not self._connected:
            return self._fallback.fetch(symbol)
        raise NotImplementedError("Implement fetch() with your broker API")


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    print("\n=== GlobalFeed Test ===\n")

    feed = GlobalFeed(
        eia_key     = os.environ.get("EIA_API_KEY", ""),
        fred_key    = os.environ.get("FRED_API_KEY", ""),
        newsapi_key = os.environ.get("NEWSAPI_KEY", ""),
    )

    test_symbols = ["CL", "GC", "EURUSD", "SPX", "NIFTY50", "ZW"]

    for sym in test_symbols:
        print(f"  [{sym}] fetching...")
        try:
            snap = feed.fetch_safe(sym)
            print(f"    Spot: {snap.spot:>12,.4f} {snap.currency}")
            print(f"    Chg:  {snap.change_pct:>+8.3f}%  "
                  f"ATR%: {snap.atr_pct:.2f}%  "
                  f"HVol: {snap.hist_vol_20:.1f}%  "
                  f"RSI: {snap.rsi_14:.0f}")
            print(f"    MA200:{snap.ma_200:>12,.4f}  "
                  f"Above: {'YES' if snap.is_above_200ma else 'NO'}")
            if snap.errors:
                print(f"    Errors: {snap.errors}")
        except Exception as e:
            print(f"    ERROR: {e}")
        print()
