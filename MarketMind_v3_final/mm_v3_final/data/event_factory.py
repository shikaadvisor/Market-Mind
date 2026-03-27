"""
data/event_factory.py
=====================
Global, asset-class-aware event factory.

The core insight for a commodities quant:
  - A 2% move in WTI Crude is noise. A 2% move in the Bund is a crisis.
  - A VIX spike to 25 is different from an OVX spike to 40.
  - Gold's triggers are geopolitical. Wheat's are weather and WASDE.
  - EURUSD triggers are ECB/Fed. USDBRL triggers are political/EM risk.

This factory knows all of that. Every threshold is scaled by the
instrument's vol_regime (from universe.py). Every trigger type is
asset-class specific. Every event description is instrument-aware.

Detection hierarchy:
  1. Calendar events (always fire regardless of price)
  2. Extreme price action (asset-class scaled)
  3. Technical structure breaks
  4. Vol regime change
  5. Inventory/flow/positioning signals
  6. News sentiment shift
  7. Cross-asset divergence (when context available)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.feeds import FeedSnapshot
from simulation.tiers.tier3_mesa import MarketEvent


# ---------------------------------------------------------------------------
# Global calendar events — asset-class tagged
# ---------------------------------------------------------------------------

CALENDAR_2026 = {
    # ── US macro ──────────────────────────────────────────────────────────
    date(2026, 1, 15):  ("us_cpi",   ["equity_index", "fx_major", "fixed_income", "commodity_metal"]),
    date(2026, 2, 12):  ("us_cpi",   ["equity_index", "fx_major", "fixed_income", "commodity_metal"]),
    date(2026, 3, 12):  ("us_cpi",   ["equity_index", "fx_major", "fixed_income", "commodity_metal"]),
    date(2026, 4,  9):  ("us_cpi",   ["equity_index", "fx_major", "fixed_income", "commodity_metal"]),
    date(2026, 5, 14):  ("us_cpi",   ["equity_index", "fx_major", "fixed_income", "commodity_metal"]),
    date(2026, 6, 11):  ("us_cpi",   ["equity_index", "fx_major", "fixed_income", "commodity_metal"]),

    date(2026, 1, 29):  ("us_fomc",  ["equity_index", "fx_major", "fixed_income", "commodity_metal", "commodity_energy"]),
    date(2026, 3, 19):  ("us_fomc",  ["equity_index", "fx_major", "fixed_income", "commodity_metal", "commodity_energy"]),
    date(2026, 5,  7):  ("us_fomc",  ["equity_index", "fx_major", "fixed_income", "commodity_metal", "commodity_energy"]),
    date(2026, 6, 18):  ("us_fomc",  ["equity_index", "fx_major", "fixed_income", "commodity_metal", "commodity_energy"]),
    date(2026, 7, 30):  ("us_fomc",  ["equity_index", "fx_major", "fixed_income", "commodity_metal", "commodity_energy"]),
    date(2026, 9, 17):  ("us_fomc",  ["equity_index", "fx_major", "fixed_income", "commodity_metal", "commodity_energy"]),

    date(2026, 1,  2):  ("us_nfp",   ["equity_index", "fx_major", "fixed_income"]),
    date(2026, 2,  6):  ("us_nfp",   ["equity_index", "fx_major", "fixed_income"]),
    date(2026, 3,  6):  ("us_nfp",   ["equity_index", "fx_major", "fixed_income"]),
    date(2026, 4,  3):  ("us_nfp",   ["equity_index", "fx_major", "fixed_income"]),
    date(2026, 5,  8):  ("us_nfp",   ["equity_index", "fx_major", "fixed_income"]),
    date(2026, 6,  5):  ("us_nfp",   ["equity_index", "fx_major", "fixed_income"]),

    # ── ECB ───────────────────────────────────────────────────────────────
    date(2026, 1, 30):  ("ecb_meeting", ["equity_index", "fx_major", "fixed_income"]),
    date(2026, 3, 12):  ("ecb_meeting", ["equity_index", "fx_major", "fixed_income"]),
    date(2026, 4, 30):  ("ecb_meeting", ["equity_index", "fx_major", "fixed_income"]),
    date(2026, 6, 11):  ("ecb_meeting", ["equity_index", "fx_major", "fixed_income"]),

    # ── RBI India ─────────────────────────────────────────────────────────
    date(2026, 2,  7):  ("rbi_policy", ["equity_index", "fx_em", "fixed_income"]),
    date(2026, 4,  9):  ("rbi_policy", ["equity_index", "fx_em", "fixed_income"]),
    date(2026, 6,  6):  ("rbi_policy", ["equity_index", "fx_em", "fixed_income"]),
    date(2026, 8,  8):  ("rbi_policy", ["equity_index", "fx_em", "fixed_income"]),
    date(2026, 10, 8):  ("rbi_policy", ["equity_index", "fx_em", "fixed_income"]),
    date(2026, 12, 5):  ("rbi_policy", ["equity_index", "fx_em", "fixed_income"]),

    # ── BOJ ───────────────────────────────────────────────────────────────
    date(2026, 1, 24):  ("boj_meeting", ["equity_index", "fx_major"]),
    date(2026, 3, 19):  ("boj_meeting", ["equity_index", "fx_major"]),
    date(2026, 5,  1):  ("boj_meeting", ["equity_index", "fx_major"]),

    # ── BOE ───────────────────────────────────────────────────────────────
    date(2026, 2,  6):  ("boe_meeting", ["equity_index", "fx_major"]),
    date(2026, 3, 20):  ("boe_meeting", ["equity_index", "fx_major"]),
    date(2026, 5,  8):  ("boe_meeting", ["equity_index", "fx_major"]),

    # ── Energy ────────────────────────────────────────────────────────────
    date(2026, 2,  4):  ("opec_meeting", ["commodity_energy"]),
    date(2026, 6,  3):  ("opec_meeting", ["commodity_energy"]),

    # ── Agriculture ───────────────────────────────────────────────────────
    date(2026, 1, 10):  ("usda_wasde", ["commodity_agri", "commodity_soft"]),
    date(2026, 2, 11):  ("usda_wasde", ["commodity_agri", "commodity_soft"]),
    date(2026, 3, 11):  ("usda_wasde", ["commodity_agri", "commodity_soft"]),
    date(2026, 4,  9):  ("usda_wasde", ["commodity_agri", "commodity_soft"]),
    date(2026, 5, 12):  ("usda_wasde", ["commodity_agri", "commodity_soft"]),
    date(2026, 6,  5):  ("usda_wasde", ["commodity_agri", "commodity_soft"]),
    date(2026, 7, 11):  ("usda_wasde", ["commodity_agri", "commodity_soft"]),
    date(2026, 8, 12):  ("usda_wasde", ["commodity_agri", "commodity_soft"]),
    date(2026, 9, 11):  ("usda_wasde", ["commodity_agri", "commodity_soft"]),
    date(2026, 10, 9):  ("usda_wasde", ["commodity_agri", "commodity_soft"]),
    date(2026, 11,10):  ("usda_wasde", ["commodity_agri", "commodity_soft"]),
    date(2026, 12, 9):  ("usda_wasde", ["commodity_agri", "commodity_soft"]),
    # USDA Crop Production (Jan, Aug — major)
    date(2026, 1, 12):  ("usda_crop_production", ["commodity_agri"]),
    date(2026, 8, 12):  ("usda_crop_production", ["commodity_agri"]),
    # USDA Planting Intentions + Grain Stocks (quarter-end)
    date(2026, 3, 31):  ("usda_planting_and_stocks", ["commodity_agri"]),
    date(2026, 6, 30):  ("usda_grain_stocks", ["commodity_agri"]),
    date(2026, 9, 30):  ("usda_grain_stocks", ["commodity_agri"]),
    date(2026, 12, 1):  ("usda_grain_stocks", ["commodity_agri"]),
}

# Weekly EIA inventory release (Wednesdays)
def _is_eia_day(d: date) -> bool:
    return d.weekday() == 2   # Wednesday

# USDA Crop Progress — every Monday Apr 13 through Nov 16
def _is_crop_progress_day(d: date) -> bool:
    if d.weekday() != 0:  # Monday
        return False
    from datetime import date as _date
    return _date(d.year, 4, 13) <= d <= _date(d.year, 11, 16)

# USDA Export Inspections — every Monday morning (released 11am ET)
def _is_export_inspection_day(d: date) -> bool:
    return d.weekday() == 0  # Monday

# NSE monthly expiry (last Thursday) — fixed version
def _is_nse_expiry(d: date) -> bool:
    import calendar as _cal
    if d.weekday() != 3: return False
    last_day = _cal.monthrange(d.year, d.month)[1]
    last_date = date(d.year, d.month, last_day)
    days_back = (last_date.weekday() - 3) % 7
    last_thursday = date(d.year, d.month, last_day - days_back)
    return d == last_thursday


# ---------------------------------------------------------------------------
# Base thresholds (medium sensitivity)
# These are scaled by instrument.threshold_scale
# ---------------------------------------------------------------------------

BASE_THRESHOLDS = {
    "gap_large":         0.012,   # 1.2% daily move threshold
    "gap_extreme":       0.025,   # 2.5% = extreme move
    "vix_spike":         0.20,    # 20% spike in vol index
    "volume_surge":      2.5,     # 2.5x average volume
    "ma200_proximity":   0.005,   # within 0.5% of 200MA
    "w52_proximity":     0.005,   # within 0.5% of 52w high/low
    "rsi_oversold":      35,
    "rsi_overbought":    70,
    "news_shift":        0.25,
    "inventory_draw":    3.0,     # mmbbl draw (energy)
    "inventory_build":   3.0,     # mmbbl build (energy)
    "atr_multiplier":    2.0,     # move > 2x ATR = significant
}

SENSITIVITY_SCALE = {
    "high":   0.65,
    "medium": 1.00,
    "low":    1.50,
}


@dataclass
class EventDetection:
    triggered:        bool
    trigger_type:     str
    triggers_fired:   list[str]
    significance:     float
    description:      str
    price_shock:      float
    news_sentiment:   float
    uncertainty:      float
    is_systemic:      bool
    sector_impact:    str
    asset_class:      str
    rationale:        str
    calendar_event:   Optional[str] = None


# ---------------------------------------------------------------------------
# The global event factory
# ---------------------------------------------------------------------------

class EventFactory:
    """
    Asset-class-aware, universe-scaled event factory.
    Instantiate once, call detect() for each new snapshot.
    """

    def __init__(self, sensitivity: str = "medium"):
        assert sensitivity in SENSITIVITY_SCALE
        self.sensitivity   = sensitivity
        self._sens_scale   = SENSITIVITY_SCALE[sensitivity]

        # State tracking per instrument
        self._prev: dict[str, dict] = {}

    def _get_prev(self, symbol: str) -> dict:
        return self._prev.get(symbol, {})

    def _set_prev(self, symbol: str, snap: FeedSnapshot) -> None:
        self._prev[symbol] = {
            "vix":     snap.equity_vix or snap.vix,
            "news":    snap.news_sentiment_score,
            "close":   snap.spot,
            "vol":     snap.hist_vol_20,
        }

    def _thresh(self, base: float, spec) -> float:
        """Scale threshold by instrument vol regime and sensitivity."""
        scale = spec.threshold_scale if spec else 1.0
        return base * scale * self._sens_scale

    def detect(self, snap: FeedSnapshot) -> EventDetection:
        """
        Detect events in a FeedSnapshot.
        Returns EventDetection whether or not triggered.
        """
        from data.universe import UNIVERSE, get_instrument

        today  = date.today()
        sym    = snap.instrument.upper()
        spec   = UNIVERSE.get(sym)
        ac     = snap.asset_class
        prev   = self._get_prev(sym)

        triggers    : list[str] = []
        price_shocks: list[float] = []
        uncertainty = 0.30
        calendar_ev = None

        chg = snap.change_pct / 100.0

        # ── 1. Calendar events ────────────────────────────────────────────
        if today in CALENDAR_2026:
            ev_name, ev_asset_classes = CALENDAR_2026[today]
            if any(ac.startswith(eac) for eac in ev_asset_classes):
                triggers.append(f"calendar_{ev_name}")
                uncertainty += 0.35
                calendar_ev = ev_name

        if _is_eia_day(today) and ac.startswith("commodity_energy"):
            triggers.append("calendar_eia_inventory")
            uncertainty += 0.20

        if _is_crop_progress_day(today) and spec and spec.asset_class in ("commodity_agri",):
            triggers.append("calendar_crop_progress")
            uncertainty += 0.10
        if _is_export_inspection_day(today) and spec and spec.asset_class in ("commodity_agri", "commodity_soft"):
            triggers.append("calendar_export_inspections")
            uncertainty += 0.05

        if _is_nse_expiry(today) and snap.exchange == "NSE":
            triggers.append("calendar_nse_expiry")
            uncertainty += 0.10

        # ── 2. Price action — scaled by ATR ──────────────────────────────
        gap_thresh    = self._thresh(BASE_THRESHOLDS["gap_large"],   spec)
        extreme_thresh = self._thresh(BASE_THRESHOLDS["gap_extreme"], spec)

        # Use ATR-normalised move when ATR is available
        if snap.atr_pct > 0:
            atr_mult = abs(chg) / (snap.atr_pct / 100.0) if snap.atr_pct > 0 else 0
            if atr_mult >= BASE_THRESHOLDS["atr_multiplier"] * self._sens_scale:
                dir_tag = "surge" if chg > 0 else "breakdown"
                triggers.append(f"atr_move_{dir_tag}")
                price_shocks.append(chg)
        else:
            if abs(chg) >= extreme_thresh:
                dir_tag = "extreme_gap_up" if chg > 0 else "extreme_gap_down"
                triggers.append(dir_tag)
                price_shocks.append(chg)
                uncertainty += 0.20
            elif abs(chg) >= gap_thresh:
                dir_tag = "gap_up" if chg > 0 else "gap_down"
                triggers.append(dir_tag)
                price_shocks.append(chg)

        # ── 3. Technical structure ────────────────────────────────────────
        prev_close = prev.get("close", 0)
        if snap.ma_200 > 0 and prev_close > 0:
            proximity = self._thresh(BASE_THRESHOLDS["ma200_proximity"], spec)
            if snap.spot > snap.ma_200 and prev_close < snap.ma_200 * (1 + proximity):
                triggers.append("ma200_cross_up")
                price_shocks.append(0.010)
            elif snap.spot < snap.ma_200 and prev_close > snap.ma_200 * (1 - proximity):
                triggers.append("ma200_cross_down")
                price_shocks.append(-0.015)
                uncertainty += 0.15

        w52_prox = self._thresh(BASE_THRESHOLDS["w52_proximity"], spec)
        if snap.week_52_high > 0 and snap.spot >= snap.week_52_high * (1 - w52_prox):
            triggers.append("w52_high_breach")
            price_shocks.append(0.008)
        if snap.week_52_low > 0 and snap.spot <= snap.week_52_low * (1 + w52_prox):
            triggers.append("w52_low_breach")
            price_shocks.append(-0.015)
            uncertainty += 0.15

        # RSI extremes — NEVER scaled by threshold_scale (RSI is 0-100, scaling makes no sense)
        # For high-vol commodities (threshold_scale > 1), use tighter RSI thresholds
        # because they move faster: oversold = 25, overbought = 75 for HIGH vol
        if spec and spec.vol_regime == "high":
            rsi_os, rsi_ob = 25, 75
        else:
            rsi_os = BASE_THRESHOLDS["rsi_oversold"]   # 35
            rsi_ob = BASE_THRESHOLDS["rsi_overbought"] # 70
        if snap.rsi_14 > 0:
            if snap.rsi_14 <= rsi_os and chg > 0:
                triggers.append("rsi_oversold_bounce")
            elif snap.rsi_14 >= rsi_ob and chg < 0:
                triggers.append("rsi_overbought_reversal")

        # ── 4. Volatility regime change ───────────────────────────────────
        prev_vix  = prev.get("vix", 0)
        curr_vol  = snap.equity_vix or snap.vix

        if prev_vix > 0 and curr_vol > 0:
            vix_chg = (curr_vol - prev_vix) / prev_vix
            vix_thr = self._thresh(BASE_THRESHOLDS["vix_spike"], spec)
            if vix_chg >= vix_thr:
                triggers.append("vol_spike")
                uncertainty += 0.25
            elif vix_chg <= -vix_thr:
                triggers.append("vol_collapse")

        # Realised vol regime shift
        prev_hvol = prev.get("vol", 0)
        if prev_hvol > 5 and snap.hist_vol_20 > prev_hvol * 1.5:
            triggers.append("realised_vol_expansion")
            uncertainty += 0.15

        # ── 5. Volume surge ───────────────────────────────────────────────
        vol_thr = BASE_THRESHOLDS["volume_surge"] * self._sens_scale
        if snap.volume_ratio >= vol_thr:
            triggers.append("volume_surge")

        # ── 6. Commodity-specific ─────────────────────────────────────────
        if ac.startswith("commodity_energy"):
            inv_draw  = BASE_THRESHOLDS["inventory_draw"]  * self._sens_scale
            inv_build = BASE_THRESHOLDS["inventory_build"] * self._sens_scale
            if snap.inventory_change <= -inv_draw:
                triggers.append("inventory_large_draw")
                price_shocks.append(0.012)
            elif snap.inventory_change >= inv_build:
                triggers.append("inventory_large_build")
                price_shocks.append(-0.010)

        if ac.startswith("commodity_metal") and ac != "commodity_metal":
            # Precious metals react to USD and real rates
            if snap.dxy > 0 and abs(snap.change_pct) > 0.5:
                triggers.append("precious_metal_dxy_move")

        # ── 7. FX-specific ────────────────────────────────────────────────
        if ac.startswith("fx"):
            if abs(snap.rate_differential) > 3.0:
                triggers.append("carry_extreme")

        # ── 8. Rates-specific ─────────────────────────────────────────────
        if ac == "fixed_income" and snap.curve_2s10s != 0:
            if snap.curve_2s10s < -50:
                triggers.append("yield_curve_deeply_inverted")
                uncertainty += 0.15
            elif snap.curve_2s10s > 100:
                triggers.append("yield_curve_steep")

        # ── 9. News sentiment shift ───────────────────────────────────────
        prev_news = prev.get("news", 0)
        if prev_news != 0.0:
            shift = abs(snap.news_sentiment_score - prev_news)
            news_thr = self._thresh(BASE_THRESHOLDS["news_shift"], spec)
            if shift >= news_thr:
                triggers.append("news_sentiment_shift")

        # ── Update state ──────────────────────────────────────────────────
        self._set_prev(sym, snap)

        # ── Aggregate ─────────────────────────────────────────────────────
        if not triggers:
            return EventDetection(
                triggered=False, trigger_type="none", triggers_fired=[],
                significance=0.0,
                description=f"No event: {sym} {chg:+.2%} (below all thresholds)",
                price_shock=chg, news_sentiment=snap.news_sentiment_score,
                uncertainty=uncertainty, is_systemic=False,
                sector_impact=spec.sector if spec else "all",
                asset_class=ac, rationale="Below all thresholds",
                calendar_event=None,
            )

        # Combined price shock
        combined_shock = sum(price_shocks) / max(len(price_shocks), 1) if price_shocks else chg * 0.5
        combined_shock = max(-0.15, min(0.15, combined_shock))

        # Significance scoring — asset-class weighted
        base_sig = min(1.0,
            0.30 * min(1.0, len(triggers) / 3.0) +
            0.30 * min(1.0, abs(combined_shock) / 0.03) +
            0.20 * (0.8 if calendar_ev else 0.0) +
            0.20 * abs(snap.news_sentiment_score)
        )

        # Vol instruments get higher significance on moves (that's their job)
        if ac == "volatility":
            base_sig = min(1.0, base_sig * 1.3)

        trigger_type = "multi_signal" if len(triggers) >= 3 else triggers[0]
        is_systemic  = _is_systemic(triggers, ac, snap)
        sector       = spec.sector if spec else "all"
        description  = _build_description(sym, triggers, snap, combined_shock,
                                           today, calendar_ev, spec)

        uncertainty = min(0.95, uncertainty)
        news_sent   = snap.news_sentiment_score or (chg * 5.0)

        return EventDetection(
            triggered      = True,
            trigger_type   = trigger_type,
            triggers_fired = triggers,
            significance   = round(base_sig, 3),
            description    = description,
            price_shock    = round(combined_shock, 4),
            news_sentiment = round(max(-1, min(1, news_sent)), 3),
            uncertainty    = round(uncertainty, 3),
            is_systemic    = is_systemic,
            sector_impact  = sector,
            asset_class    = ac,
            rationale      = f"Triggers: {', '.join(triggers[:4])}",
            calendar_event = calendar_ev,
        )

    def to_market_event(self, detection: EventDetection, snap: FeedSnapshot) -> MarketEvent:
        return MarketEvent(
            description    = detection.description,
            price_shock    = detection.price_shock,
            news_sentiment = detection.news_sentiment,
            uncertainty    = detection.uncertainty,
            is_systemic    = detection.is_systemic,
            sector_impact  = detection.sector_impact,
        )

    def should_run_pipeline(
        self,
        detection:         EventDetection,
        min_significance:  float = 0.30,
        force_if_calendar: bool  = True,
    ) -> bool:
        if not detection.triggered:
            return False
        if force_if_calendar and detection.calendar_event:
            return True
        return detection.significance >= min_significance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_systemic(triggers: list[str], ac: str, snap: FeedSnapshot) -> bool:
    systemic_triggers = {
        "calendar_us_fomc", "calendar_us_cpi", "calendar_ecb_meeting",
        "extreme_gap_down", "extreme_gap_up", "vol_spike",
        "ma200_cross_down", "w52_low_breach",
        "atr_move_breakdown", "calendar_opec_meeting",
    }
    if set(triggers) & systemic_triggers:
        return True
    # High-vol systemic indicators
    if snap.vix > 30 or (snap.equity_vix > 25 and ac.startswith("equity")):
        return True
    return False


def _build_description(
    symbol:    str,
    triggers:  list[str],
    snap:      FeedSnapshot,
    chg:       float,
    today:     date,
    calendar:  Optional[str],
    spec,
) -> str:
    name = spec.name if spec else symbol
    parts = []

    # Lead: calendar event takes priority
    if calendar:
        calendar_labels = {
            "us_fomc":      "US FOMC decision day",
            "us_cpi":       "US CPI release day",
            "us_nfp":       "US Non-Farm Payrolls day",
            "ecb_meeting":  "ECB policy decision day",
            "rbi_policy":   "RBI MPC policy day",
            "boj_meeting":  "BOJ policy decision day",
            "boe_meeting":  "BOE policy decision day",
            "opec_meeting": "OPEC+ ministerial meeting day",
            "usda_wasde":   "USDA WASDE report day",
            "eia_inventory":"EIA inventory report day",
            "nse_expiry":   "NSE monthly F&O expiry",
        }
        parts.append(calendar_labels.get(calendar, f"Calendar: {calendar}"))

    # Price action
    if abs(chg) >= 0.005:
        dir_w = "surged" if chg > 0 else "fell"
        parts.append(f"{name} {dir_w} {abs(chg):.1%}")

    # Asset-specific context
    context_map = {
        "ma200_cross_up":          f"crossing above 200-DMA ({snap.ma_200:,.2f})",
        "ma200_cross_down":        f"breaking below 200-DMA ({snap.ma_200:,.2f})",
        "w52_high_breach":         f"at 52-week high ({snap.week_52_high:,.2f})",
        "w52_low_breach":          f"at 52-week low ({snap.week_52_low:,.2f})",
        "vol_spike":               f"vol spiked (HVol: {snap.hist_vol_20:.0f}%)",
        "inventory_large_draw":    f"inventory draw of {abs(snap.inventory_change):.1f} units",
        "inventory_large_build":   f"inventory build of {snap.inventory_change:.1f} units",
        "volume_surge":            f"volume {snap.volume_ratio:.1f}x average",
        "rsi_oversold_bounce":     f"bouncing — RSI {snap.rsi_14:.0f} (oversold)",
        "rsi_overbought_reversal": f"reversing — RSI {snap.rsi_14:.0f} (overbought)",
        "yield_curve_deeply_inverted": f"2s10s at {snap.curve_2s10s:.0f}bp (inverted)",
        "carry_extreme":           f"rate differential {snap.rate_differential:+.1f}%",
    }
    for t in triggers[:3]:
        ctx = context_map.get(t)
        if ctx and ctx not in " ".join(parts):
            parts.append(ctx)

    # Global macro always appended
    macro_str = (
        f"DXY:{snap.dxy:.1f} "
        f"VIX:{snap.vix:.1f} "
        f"US10Y:{snap.us_10y:.2f}%"
    )
    if snap.top_headlines:
        parts.append(f'"{snap.top_headlines[0][:55]}"')

    description = (
        f"{today.strftime('%d %b %Y')}: " +
        ". ".join(parts) +
        f" | {symbol}: {snap.spot:,.4f} {snap.currency} | {macro_str}"
    )
    return description[:600]


def snapshot_to_pipeline_inputs(
    snap:        FeedSnapshot,
    factory:     EventFactory,
    instrument:  str,
    exchange:    str  = "",
    asset_class: str  = "",
) -> tuple[Optional[MarketEvent], dict, EventDetection]:
    detection   = factory.detect(snap)
    market_data = snap.to_market_data_dict()
    if not detection.triggered:
        return None, market_data, detection
    event = factory.to_market_event(detection, snap)
    return event, market_data, detection


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data.feeds import FeedSnapshot

    print("\n=== Global EventFactory Test ===\n")

    factory = EventFactory(sensitivity="medium")

    cases = [
        ("CL — OPEC meeting day + crude drop", FeedSnapshot(
            instrument="CL", exchange="NYMEX", asset_class="commodity_energy",
            currency="USD", timestamp=datetime.now(),
            spot=71.50, prev_close=74.20, change_pct=-3.64,
            atr_pct=2.1, hist_vol_20=38.0, rsi_14=32,
            vix=22.0, dxy=105.2, us_10y=4.6,
            inventory_change=-5.2, volume_ratio=2.8,
        )),
        ("GC — Gold on FOMC day", FeedSnapshot(
            instrument="GC", exchange="COMEX", asset_class="commodity_metal",
            currency="USD", timestamp=datetime.now(),
            spot=2890.0, prev_close=2850.0, change_pct=1.40,
            atr_pct=0.9, hist_vol_20=12.5, rsi_14=68,
            vix=16.0, dxy=102.8, us_10y=4.2,
        )),
        ("EURUSD — post-ECB move", FeedSnapshot(
            instrument="EURUSD", exchange="FOREX", asset_class="fx_major",
            currency="USD", timestamp=datetime.now(),
            spot=1.0820, prev_close=1.0920, change_pct=-0.92,
            atr_pct=0.45, hist_vol_20=6.8, rsi_14=38,
            vix=19.5, dxy=106.1, us_10y=4.7,
            rate_differential=1.2,
        )),
        ("ZW — Wheat quiet day", FeedSnapshot(
            instrument="ZW", exchange="CBOT", asset_class="commodity_agri",
            currency="USD", timestamp=datetime.now(),
            spot=545.0, prev_close=542.0, change_pct=0.55,
            atr_pct=1.8, hist_vol_20=22.0, rsi_14=52,
            vix=17.0, dxy=104.5, us_10y=4.5,
        )),
        ("SPX — Vol spike + breakdown", FeedSnapshot(
            instrument="SPX", exchange="NYSE", asset_class="equity_index",
            currency="USD", timestamp=datetime.now(),
            spot=5280.0, prev_close=5410.0, change_pct=-2.40,
            atr_pct=0.85, hist_vol_20=16.0, rsi_14=31,
            vix=28.5, equity_vix=28.5, dxy=106.5, us_10y=4.85,
            volume_ratio=3.1,
        )),
    ]

    for name, snap in cases:
        factory._prev = {}    # fresh state
        d = factory.detect(snap)
        run = factory.should_run_pipeline(d)
        print(f"  {name}")
        print(f"    Triggered:    {d.triggered}  Sig={d.significance:.2f}  RunPipeline={run}")
        if d.triggered:
            print(f"    Triggers:     {d.triggers_fired}")
            print(f"    Price shock:  {d.price_shock:+.2%}  Sent={d.news_sentiment:+.2f}  Unc={d.uncertainty:.0%}")
            print(f"    Description:  {d.description[:100]}...")
        print()
