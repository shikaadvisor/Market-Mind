"""
data/universe.py
================
The global instrument universe — every instrument MarketMind can trade.

For a commodities quant strategist, the universe spans:
  - Equity indices (US, Europe, Asia, EM, India)
  - Commodities (Energy, Metals, Agriculturals, Softs)
  - FX (Major pairs, EM pairs, commodity currencies)
  - Fixed income (US Treasuries, Bund, JGB, EM bonds)
  - Volatility (VIX, VSTOXX, OVX, GVZ)
  - Crypto (as an alternative risk asset)

Each instrument is an InstrumentSpec that knows:
  - Its Yahoo Finance symbol (for data fetching)
  - Its asset class and sub-class
  - Its primary exchange and trading timezone
  - Its session hours (for the scheduler)
  - Its volatility regime (low/medium/high) — affects trigger thresholds
  - Its liquidity tier (1-3) — affects significance weighting
  - Its sector/region for correlation grouping
  - Its currency denomination
  - Its point value and contract size (for futures)

Design principle: the universe is the single source of truth.
If an instrument is in the universe, the whole system knows how to
handle it — feeds, event factory, scheduler, desk agents, signals.
Adding a new instrument = one entry in UNIVERSE dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Asset class taxonomy
# ---------------------------------------------------------------------------

class AssetClass:
    EQUITY_INDEX  = "equity_index"
    EQUITY_STOCK  = "equity_stock"
    COMMODITY_ENERGY    = "commodity_energy"
    COMMODITY_METAL     = "commodity_metal"
    COMMODITY_AGRI      = "commodity_agri"
    COMMODITY_SOFT      = "commodity_soft"
    FX_MAJOR      = "fx_major"
    FX_EM         = "fx_em"
    FIXED_INCOME  = "fixed_income"
    VOLATILITY    = "volatility"
    CRYPTO        = "crypto"


class VolatilityRegime:
    """
    Asset-class default volatility regime.
    Drives trigger threshold scaling in EventFactory.
    LOW = tight thresholds (bonds, short-dated FX)
    HIGH = wide thresholds (crypto, EM FX, agricultuals)
    """
    LOW    = "low"      # daily move 0.1–0.5%
    MEDIUM = "medium"   # daily move 0.5–1.5%
    HIGH   = "high"     # daily move 1.5–5%+


# ---------------------------------------------------------------------------
# Instrument spec
# ---------------------------------------------------------------------------

@dataclass
class InstrumentSpec:
    """Full specification for one tradeable instrument."""

    # Identity
    symbol:       str              # Our canonical symbol (e.g. "NIFTY50", "CL", "XAUUSD")
    yf_symbol:    str              # Yahoo Finance ticker
    name:         str              # Human-readable name
    asset_class:  str              # AssetClass constant
    currency:     str              # Price denomination (USD, INR, EUR, GBP, JPY...)

    # Market structure
    exchange:     str              # Primary exchange / venue
    timezone:     str              # pytz timezone of primary session
    session_open: tuple            # (hour, minute) in local timezone
    session_close: tuple           # (hour, minute) in local timezone
    trading_days: str = "mon-fri"  # "mon-fri" | "mon-sat" | "24/7"

    # Risk profile
    vol_regime:   str   = VolatilityRegime.MEDIUM
    liquidity_tier: int = 2        # 1=highest, 3=lowest
    typical_spread_bps: float = 2.0  # typical bid-ask in basis points

    # Instrument metadata
    region:       str   = "global"
    sector:       str   = "all"
    sub_class:    str   = ""       # e.g. "precious_metal", "crude_oil", "grain"
    is_futures:   bool  = False
    contract_size: float = 1.0    # e.g. 1000 for CL (1000 barrels)
    point_value:  float = 1.0     # USD per point

    # Correlation groups (for risk manager)
    corr_group:   str   = ""       # e.g. "em_equity", "energy", "risk_off"

    # Calendar events specific to this instrument
    key_events:   list[str] = field(default_factory=list)

    @property
    def is_commodity(self) -> bool:
        return self.asset_class.startswith("commodity")

    @property
    def is_equity(self) -> bool:
        return self.asset_class.startswith("equity")

    @property
    def is_fx(self) -> bool:
        return self.asset_class.startswith("fx")

    @property
    def is_fixed_income(self) -> bool:
        return self.asset_class == "fixed_income"

    @property
    def threshold_scale(self) -> float:
        """Scale factor for EventFactory thresholds. High-vol = wider thresholds."""
        return {
            VolatilityRegime.LOW:    0.5,
            VolatilityRegime.MEDIUM: 1.0,
            VolatilityRegime.HIGH:   2.2,
        }.get(self.vol_regime, 1.0)

    def describe(self) -> str:
        return (
            f"{self.symbol} ({self.name}) | {self.asset_class} | "
            f"{self.currency} | {self.exchange} | vol={self.vol_regime}"
        )


# ---------------------------------------------------------------------------
# The global universe
# ---------------------------------------------------------------------------

UNIVERSE: dict[str, InstrumentSpec] = {

    # =========================================================================
    # EQUITY INDICES — Global
    # =========================================================================

    # ── US ───────────────────────────────────────────────────────────────────
    "SPX": InstrumentSpec(
        symbol="SPX", yf_symbol="^GSPC", name="S&P 500",
        asset_class=AssetClass.EQUITY_INDEX, currency="USD",
        exchange="NYSE", timezone="America/New_York",
        session_open=(9, 30), session_close=(16, 0),
        vol_regime=VolatilityRegime.MEDIUM, liquidity_tier=1,
        region="us", sector="all", corr_group="us_equity",
        key_events=["us_cpi", "us_fomc", "us_nfp", "us_earnings_season"],
    ),
    "NDX": InstrumentSpec(
        symbol="NDX", yf_symbol="^NDX", name="Nasdaq 100",
        asset_class=AssetClass.EQUITY_INDEX, currency="USD",
        exchange="NASDAQ", timezone="America/New_York",
        session_open=(9, 30), session_close=(16, 0),
        vol_regime=VolatilityRegime.MEDIUM, liquidity_tier=1,
        region="us", sector="technology", corr_group="us_equity",
        key_events=["us_cpi", "us_fomc", "mag7_earnings"],
    ),
    "DJI": InstrumentSpec(
        symbol="DJI", yf_symbol="^DJI", name="Dow Jones Industrial",
        asset_class=AssetClass.EQUITY_INDEX, currency="USD",
        exchange="NYSE", timezone="America/New_York",
        session_open=(9, 30), session_close=(16, 0),
        vol_regime=VolatilityRegime.MEDIUM, liquidity_tier=1,
        region="us", sector="all", corr_group="us_equity",
    ),
    "RUT": InstrumentSpec(
        symbol="RUT", yf_symbol="^RUT", name="Russell 2000",
        asset_class=AssetClass.EQUITY_INDEX, currency="USD",
        exchange="NYSE", timezone="America/New_York",
        session_open=(9, 30), session_close=(16, 0),
        vol_regime=VolatilityRegime.MEDIUM, liquidity_tier=1,
        region="us", sector="small_cap", corr_group="us_equity",
    ),

    # ── Europe ────────────────────────────────────────────────────────────────
    "DAX": InstrumentSpec(
        symbol="DAX", yf_symbol="^GDAXI", name="DAX 40 (Germany)",
        asset_class=AssetClass.EQUITY_INDEX, currency="EUR",
        exchange="XETRA", timezone="Europe/Berlin",
        session_open=(9, 0), session_close=(17, 30),
        vol_regime=VolatilityRegime.MEDIUM, liquidity_tier=1,
        region="europe", sector="all", corr_group="eu_equity",
        key_events=["ecb_meeting", "german_cpi", "german_pmi"],
    ),
    "FTSE": InstrumentSpec(
        symbol="FTSE", yf_symbol="^FTSE", name="FTSE 100 (UK)",
        asset_class=AssetClass.EQUITY_INDEX, currency="GBP",
        exchange="LSE", timezone="Europe/London",
        session_open=(8, 0), session_close=(16, 30),
        vol_regime=VolatilityRegime.MEDIUM, liquidity_tier=1,
        region="europe", sector="all", corr_group="eu_equity",
        key_events=["boe_meeting", "uk_cpi", "uk_gdp"],
    ),
    "CAC": InstrumentSpec(
        symbol="CAC", yf_symbol="^FCHI", name="CAC 40 (France)",
        asset_class=AssetClass.EQUITY_INDEX, currency="EUR",
        exchange="EURONEXT", timezone="Europe/Paris",
        session_open=(9, 0), session_close=(17, 30),
        vol_regime=VolatilityRegime.MEDIUM, liquidity_tier=1,
        region="europe", sector="all", corr_group="eu_equity",
    ),
    "IBEX": InstrumentSpec(
        symbol="IBEX", yf_symbol="^IBEX", name="IBEX 35 (Spain)",
        asset_class=AssetClass.EQUITY_INDEX, currency="EUR",
        exchange="BME", timezone="Europe/Madrid",
        session_open=(9, 0), session_close=(17, 30),
        vol_regime=VolatilityRegime.MEDIUM, liquidity_tier=2,
        region="europe", sector="all", corr_group="eu_equity",
    ),
    "STOXX50": InstrumentSpec(
        symbol="STOXX50", yf_symbol="^STOXX50E", name="Euro Stoxx 50",
        asset_class=AssetClass.EQUITY_INDEX, currency="EUR",
        exchange="EUREX", timezone="Europe/Berlin",
        session_open=(9, 0), session_close=(17, 30),
        vol_regime=VolatilityRegime.MEDIUM, liquidity_tier=1,
        region="europe", sector="all", corr_group="eu_equity",
        key_events=["ecb_meeting"],
    ),

    # ── Asia-Pacific ──────────────────────────────────────────────────────────
    "NIKKEI": InstrumentSpec(
        symbol="NIKKEI", yf_symbol="^N225", name="Nikkei 225 (Japan)",
        asset_class=AssetClass.EQUITY_INDEX, currency="JPY",
        exchange="TSE", timezone="Asia/Tokyo",
        session_open=(9, 0), session_close=(15, 30),
        vol_regime=VolatilityRegime.MEDIUM, liquidity_tier=1,
        region="asia", sector="all", corr_group="asia_equity",
        key_events=["boj_meeting", "japan_cpi", "tankan"],
    ),
    "TOPIX": InstrumentSpec(
        symbol="TOPIX", yf_symbol="^TPX", name="TOPIX (Japan)",
        asset_class=AssetClass.EQUITY_INDEX, currency="JPY",
        exchange="TSE", timezone="Asia/Tokyo",
        session_open=(9, 0), session_close=(15, 30),
        vol_regime=VolatilityRegime.MEDIUM, liquidity_tier=1,
        region="asia", sector="all", corr_group="asia_equity",
    ),
    "HSCEI": InstrumentSpec(
        symbol="HSCEI", yf_symbol="^HSCE", name="Hang Seng China Enterprises",
        asset_class=AssetClass.EQUITY_INDEX, currency="HKD",
        exchange="HKEX", timezone="Asia/Hong_Kong",
        session_open=(9, 30), session_close=(16, 0),
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=1,
        region="china", sector="all", corr_group="china_equity",
        key_events=["pboc_mlf", "china_cpi", "china_pmi", "npc_meeting"],
    ),
    "HSI": InstrumentSpec(
        symbol="HSI", yf_symbol="^HSI", name="Hang Seng Index",
        asset_class=AssetClass.EQUITY_INDEX, currency="HKD",
        exchange="HKEX", timezone="Asia/Hong_Kong",
        session_open=(9, 30), session_close=(16, 0),
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=1,
        region="china", sector="all", corr_group="china_equity",
    ),
    "KOSPI": InstrumentSpec(
        symbol="KOSPI", yf_symbol="^KS11", name="KOSPI (South Korea)",
        asset_class=AssetClass.EQUITY_INDEX, currency="KRW",
        exchange="KRX", timezone="Asia/Seoul",
        session_open=(9, 0), session_close=(15, 30),
        vol_regime=VolatilityRegime.MEDIUM, liquidity_tier=1,
        region="asia", sector="all", corr_group="asia_equity",
    ),
    "TWII": InstrumentSpec(
        symbol="TWII", yf_symbol="^TWII", name="TAIEX (Taiwan)",
        asset_class=AssetClass.EQUITY_INDEX, currency="TWD",
        exchange="TWSE", timezone="Asia/Taipei",
        session_open=(9, 0), session_close=(13, 30),
        vol_regime=VolatilityRegime.MEDIUM, liquidity_tier=1,
        region="asia", sector="technology", corr_group="asia_equity",
    ),
    "STI": InstrumentSpec(
        symbol="STI", yf_symbol="^STI", name="Straits Times Index (Singapore)",
        asset_class=AssetClass.EQUITY_INDEX, currency="SGD",
        exchange="SGX", timezone="Asia/Singapore",
        session_open=(9, 0), session_close=(17, 0),
        vol_regime=VolatilityRegime.MEDIUM, liquidity_tier=2,
        region="asia", sector="all", corr_group="asia_equity",
    ),
    "ASX200": InstrumentSpec(
        symbol="ASX200", yf_symbol="^AXJO", name="ASX 200 (Australia)",
        asset_class=AssetClass.EQUITY_INDEX, currency="AUD",
        exchange="ASX", timezone="Australia/Sydney",
        session_open=(10, 0), session_close=(16, 0),
        vol_regime=VolatilityRegime.MEDIUM, liquidity_tier=1,
        region="oceania", sector="all", corr_group="asia_equity",
        key_events=["rba_meeting", "aus_cpi", "aus_employment"],
    ),

    # ── India ─────────────────────────────────────────────────────────────────
    "NIFTY50": InstrumentSpec(
        symbol="NIFTY50", yf_symbol="^NSEI", name="Nifty 50 (India)",
        asset_class=AssetClass.EQUITY_INDEX, currency="INR",
        exchange="NSE", timezone="Asia/Kolkata",
        session_open=(9, 15), session_close=(15, 30),
        vol_regime=VolatilityRegime.MEDIUM, liquidity_tier=1,
        region="india", sector="all", corr_group="em_equity",
        key_events=["rbi_policy", "india_budget", "india_cpi", "india_gdp"],
    ),
    "BANKNIFTY": InstrumentSpec(
        symbol="BANKNIFTY", yf_symbol="^NSEBANK", name="Bank Nifty (India)",
        asset_class=AssetClass.EQUITY_INDEX, currency="INR",
        exchange="NSE", timezone="Asia/Kolkata",
        session_open=(9, 15), session_close=(15, 30),
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=1,
        region="india", sector="banking", corr_group="em_equity",
        key_events=["rbi_policy", "india_budget"],
    ),
    "SENSEX": InstrumentSpec(
        symbol="SENSEX", yf_symbol="^BSESN", name="BSE Sensex (India)",
        asset_class=AssetClass.EQUITY_INDEX, currency="INR",
        exchange="BSE", timezone="Asia/Kolkata",
        session_open=(9, 15), session_close=(15, 30),
        vol_regime=VolatilityRegime.MEDIUM, liquidity_tier=1,
        region="india", sector="all", corr_group="em_equity",
    ),

    # ── EM ────────────────────────────────────────────────────────────────────
    "EWZ": InstrumentSpec(
        symbol="EWZ", yf_symbol="EWZ", name="iShares Brazil ETF",
        asset_class=AssetClass.EQUITY_INDEX, currency="USD",
        exchange="NYSE", timezone="America/Sao_Paulo",
        session_open=(9, 30), session_close=(16, 0),
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=2,
        region="latam", sector="all", corr_group="em_equity",
        key_events=["copom_meeting", "brazil_cpi"],
    ),
    "EEM": InstrumentSpec(
        symbol="EEM", yf_symbol="EEM", name="iShares MSCI EM ETF",
        asset_class=AssetClass.EQUITY_INDEX, currency="USD",
        exchange="NYSE", timezone="America/New_York",
        session_open=(9, 30), session_close=(16, 0),
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=1,
        region="em", sector="all", corr_group="em_equity",
    ),

    # =========================================================================
    # COMMODITIES — Energy
    # =========================================================================

    "CL": InstrumentSpec(
        symbol="CL", yf_symbol="CL=F", name="WTI Crude Oil (front month)",
        asset_class=AssetClass.COMMODITY_ENERGY, currency="USD",
        exchange="NYMEX", timezone="America/New_York",
        session_open=(0, 0), session_close=(23, 59), trading_days="mon-fri",
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=1,
        region="global", sector="energy", sub_class="crude_oil",
        is_futures=True, contract_size=1000.0, point_value=1000.0,
        corr_group="energy",
        key_events=["opec_meeting", "eia_inventory", "api_inventory",
                    "us_rig_count", "opec_production_cut"],
    ),
    "BZ": InstrumentSpec(
        symbol="BZ", yf_symbol="BZ=F", name="Brent Crude Oil",
        asset_class=AssetClass.COMMODITY_ENERGY, currency="USD",
        exchange="ICE", timezone="Europe/London",
        session_open=(1, 0), session_close=(23, 0), trading_days="mon-fri",
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=1,
        region="global", sector="energy", sub_class="crude_oil",
        is_futures=True, contract_size=1000.0, point_value=1000.0,
        corr_group="energy",
        key_events=["opec_meeting", "eia_inventory", "iea_report"],
    ),
    "NG": InstrumentSpec(
        symbol="NG", yf_symbol="NG=F", name="Henry Hub Natural Gas",
        asset_class=AssetClass.COMMODITY_ENERGY, currency="USD",
        exchange="NYMEX", timezone="America/New_York",
        session_open=(0, 0), session_close=(23, 59), trading_days="mon-fri",
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=1,
        region="us", sector="energy", sub_class="natural_gas",
        is_futures=True, contract_size=10000.0,
        corr_group="energy",
        key_events=["eia_gas_storage", "weather_forecast"],
    ),
    "HO": InstrumentSpec(
        symbol="HO", yf_symbol="HO=F", name="Heating Oil",
        asset_class=AssetClass.COMMODITY_ENERGY, currency="USD",
        exchange="NYMEX", timezone="America/New_York",
        session_open=(0, 0), session_close=(23, 59), trading_days="mon-fri",
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=2,
        region="us", sector="energy", sub_class="distillate",
        is_futures=True, corr_group="energy",
    ),
    "RB": InstrumentSpec(
        symbol="RB", yf_symbol="RB=F", name="RBOB Gasoline",
        asset_class=AssetClass.COMMODITY_ENERGY, currency="USD",
        exchange="NYMEX", timezone="America/New_York",
        session_open=(0, 0), session_close=(23, 59), trading_days="mon-fri",
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=2,
        region="us", sector="energy", sub_class="gasoline",
        is_futures=True, corr_group="energy",
    ),

    # =========================================================================
    # COMMODITIES — Metals
    # =========================================================================

    "GC": InstrumentSpec(
        symbol="GC", yf_symbol="GC=F", name="Gold (front month)",
        asset_class=AssetClass.COMMODITY_METAL, currency="USD",
        exchange="COMEX", timezone="America/New_York",
        session_open=(0, 0), session_close=(23, 59), trading_days="mon-fri",
        vol_regime=VolatilityRegime.MEDIUM, liquidity_tier=1,
        region="global", sector="precious_metals", sub_class="precious_metal",
        is_futures=True, contract_size=100.0, point_value=100.0,
        corr_group="safe_haven",
        key_events=["us_fomc", "us_cpi", "us_dollar_index", "geopolitical_event"],
    ),
    "XAUUSD": InstrumentSpec(
        symbol="XAUUSD", yf_symbol="GC=F", name="Gold Spot (USD)",
        asset_class=AssetClass.COMMODITY_METAL, currency="USD",
        exchange="LBMA", timezone="Europe/London",
        session_open=(0, 0), session_close=(23, 59), trading_days="mon-fri",
        vol_regime=VolatilityRegime.MEDIUM, liquidity_tier=1,
        region="global", sector="precious_metals", sub_class="precious_metal",
        corr_group="safe_haven",
        key_events=["us_fomc", "us_cpi", "dxy_move"],
    ),
    "SI": InstrumentSpec(
        symbol="SI", yf_symbol="SI=F", name="Silver (front month)",
        asset_class=AssetClass.COMMODITY_METAL, currency="USD",
        exchange="COMEX", timezone="America/New_York",
        session_open=(0, 0), session_close=(23, 59), trading_days="mon-fri",
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=1,
        region="global", sector="precious_metals", sub_class="precious_metal",
        is_futures=True, contract_size=5000.0,
        corr_group="safe_haven",
    ),
    "PL": InstrumentSpec(
        symbol="PL", yf_symbol="PL=F", name="Platinum",
        asset_class=AssetClass.COMMODITY_METAL, currency="USD",
        exchange="NYMEX", timezone="America/New_York",
        session_open=(0, 0), session_close=(23, 59), trading_days="mon-fri",
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=2,
        region="global", sector="precious_metals", sub_class="precious_metal",
        is_futures=True, corr_group="industrial_metals",
    ),
    "PA": InstrumentSpec(
        symbol="PA", yf_symbol="PA=F", name="Palladium",
        asset_class=AssetClass.COMMODITY_METAL, currency="USD",
        exchange="NYMEX", timezone="America/New_York",
        session_open=(0, 0), session_close=(23, 59), trading_days="mon-fri",
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=2,
        region="global", sector="precious_metals", sub_class="precious_metal",
        is_futures=True, corr_group="industrial_metals",
    ),
    "HG": InstrumentSpec(
        symbol="HG", yf_symbol="HG=F", name="Copper",
        asset_class=AssetClass.COMMODITY_METAL, currency="USD",
        exchange="COMEX", timezone="America/New_York",
        session_open=(0, 0), session_close=(23, 59), trading_days="mon-fri",
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=1,
        region="global", sector="industrial_metals", sub_class="base_metal",
        is_futures=True, contract_size=25000.0,
        corr_group="industrial_metals",
        key_events=["china_pmi", "china_gdp", "global_pmi"],
    ),
    "ALI": InstrumentSpec(
        symbol="ALI", yf_symbol="ALI=F", name="Aluminium",
        asset_class=AssetClass.COMMODITY_METAL, currency="USD",
        exchange="COMEX", timezone="America/New_York",
        session_open=(0, 0), session_close=(23, 59), trading_days="mon-fri",
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=2,
        region="global", sector="industrial_metals", sub_class="base_metal",
        is_futures=True, corr_group="industrial_metals",
    ),

    # =========================================================================
    # COMMODITIES — Agriculture
    # =========================================================================

    "ZW": InstrumentSpec(
        symbol="ZW", yf_symbol="ZW=F", name="CBOT Wheat",
        asset_class=AssetClass.COMMODITY_AGRI, currency="USD",
        exchange="CBOT", timezone="America/Chicago",
        session_open=(8, 30), session_close=(13, 20),
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=1,
        region="global", sector="grains", sub_class="grain",
        is_futures=True, contract_size=5000.0,
        corr_group="agriculture",
        key_events=["usda_wasde", "usda_crop_progress", "ukraine_export"],
    ),
    "ZC": InstrumentSpec(
        symbol="ZC", yf_symbol="ZC=F", name="CBOT Corn",
        asset_class=AssetClass.COMMODITY_AGRI, currency="USD",
        exchange="CBOT", timezone="America/Chicago",
        session_open=(8, 30), session_close=(13, 20),
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=1,
        region="global", sector="grains", sub_class="grain",
        is_futures=True, contract_size=5000.0,
        corr_group="agriculture",
        key_events=["usda_wasde", "usda_crop_progress"],
    ),
    "ZS": InstrumentSpec(
        symbol="ZS", yf_symbol="ZS=F", name="CBOT Soybeans",
        asset_class=AssetClass.COMMODITY_AGRI, currency="USD",
        exchange="CBOT", timezone="America/Chicago",
        session_open=(8, 30), session_close=(13, 20),
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=1,
        region="global", sector="oilseeds", sub_class="oilseed",
        is_futures=True, contract_size=5000.0,
        corr_group="agriculture",
        key_events=["usda_wasde", "la_nina_el_nino"],
    ),

    # ── Softs ─────────────────────────────────────────────────────────────────
    "KC": InstrumentSpec(
        symbol="KC", yf_symbol="KC=F", name="Coffee (Arabica)",
        asset_class=AssetClass.COMMODITY_SOFT, currency="USD",
        exchange="ICE", timezone="America/New_York",
        session_open=(3, 30), session_close=(12, 30),
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=2,
        region="global", sector="softs", sub_class="soft",
        is_futures=True, contract_size=37500.0,
        corr_group="softs",
        key_events=["brazil_harvest", "vietnam_export", "weather_brazil"],
    ),
    "SB": InstrumentSpec(
        symbol="SB", yf_symbol="SB=F", name="Sugar #11",
        asset_class=AssetClass.COMMODITY_SOFT, currency="USD",
        exchange="ICE", timezone="America/New_York",
        session_open=(2, 30), session_close=(12, 0),
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=2,
        region="global", sector="softs", sub_class="soft",
        is_futures=True, contract_size=112000.0,
        corr_group="softs",
        key_events=["brazil_harvest", "india_sugar_policy"],
    ),
    "CT": InstrumentSpec(
        symbol="CT", yf_symbol="CT=F", name="Cotton",
        asset_class=AssetClass.COMMODITY_SOFT, currency="USD",
        exchange="ICE", timezone="America/New_York",
        session_open=(8, 0), session_close=(14, 0),
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=2,
        region="global", sector="softs", sub_class="soft",
        is_futures=True, contract_size=50000.0,
        corr_group="softs",
    ),

    # =========================================================================
    # FX — Majors
    # =========================================================================

    "EURUSD": InstrumentSpec(
        symbol="EURUSD", yf_symbol="EURUSD=X", name="EUR/USD",
        asset_class=AssetClass.FX_MAJOR, currency="USD",
        exchange="FOREX", timezone="Europe/London",
        session_open=(0, 0), session_close=(23, 59), trading_days="mon-fri",
        vol_regime=VolatilityRegime.LOW, liquidity_tier=1,
        typical_spread_bps=0.5,
        region="global", sector="fx", sub_class="major_pair",
        corr_group="risk_fx",
        key_events=["us_fomc", "ecb_meeting", "us_cpi", "eu_cpi"],
    ),
    "GBPUSD": InstrumentSpec(
        symbol="GBPUSD", yf_symbol="GBPUSD=X", name="GBP/USD",
        asset_class=AssetClass.FX_MAJOR, currency="USD",
        exchange="FOREX", timezone="Europe/London",
        session_open=(0, 0), session_close=(23, 59), trading_days="mon-fri",
        vol_regime=VolatilityRegime.LOW, liquidity_tier=1,
        region="global", sector="fx", sub_class="major_pair",
        corr_group="risk_fx",
        key_events=["boe_meeting", "uk_cpi", "uk_gdp"],
    ),
    "USDJPY": InstrumentSpec(
        symbol="USDJPY", yf_symbol="JPY=X", name="USD/JPY",
        asset_class=AssetClass.FX_MAJOR, currency="JPY",
        exchange="FOREX", timezone="Asia/Tokyo",
        session_open=(0, 0), session_close=(23, 59), trading_days="mon-fri",
        vol_regime=VolatilityRegime.LOW, liquidity_tier=1,
        region="global", sector="fx", sub_class="major_pair",
        corr_group="safe_haven",
        key_events=["boj_meeting", "us_fomc", "japan_cpi", "carry_trade"],
    ),
    "USDCHF": InstrumentSpec(
        symbol="USDCHF", yf_symbol="CHF=X", name="USD/CHF",
        asset_class=AssetClass.FX_MAJOR, currency="CHF",
        exchange="FOREX", timezone="Europe/Zurich",
        session_open=(0, 0), session_close=(23, 59), trading_days="mon-fri",
        vol_regime=VolatilityRegime.LOW, liquidity_tier=1,
        region="global", sector="fx", sub_class="major_pair",
        corr_group="safe_haven",
    ),
    "AUDUSD": InstrumentSpec(
        symbol="AUDUSD", yf_symbol="AUDUSD=X", name="AUD/USD",
        asset_class=AssetClass.FX_MAJOR, currency="USD",
        exchange="FOREX", timezone="Australia/Sydney",
        session_open=(0, 0), session_close=(23, 59), trading_days="mon-fri",
        vol_regime=VolatilityRegime.MEDIUM, liquidity_tier=1,
        region="global", sector="fx", sub_class="commodity_currency",
        corr_group="risk_fx",
        key_events=["rba_meeting", "china_pmi", "aus_employment"],
    ),
    "USDCAD": InstrumentSpec(
        symbol="USDCAD", yf_symbol="CAD=X", name="USD/CAD",
        asset_class=AssetClass.FX_MAJOR, currency="CAD",
        exchange="FOREX", timezone="America/Toronto",
        session_open=(0, 0), session_close=(23, 59), trading_days="mon-fri",
        vol_regime=VolatilityRegime.MEDIUM, liquidity_tier=1,
        region="global", sector="fx", sub_class="commodity_currency",
        corr_group="energy",
        key_events=["boc_meeting", "canada_cpi", "crude_eia"],
    ),
    "NZDUSD": InstrumentSpec(
        symbol="NZDUSD", yf_symbol="NZDUSD=X", name="NZD/USD",
        asset_class=AssetClass.FX_MAJOR, currency="USD",
        exchange="FOREX", timezone="Pacific/Auckland",
        session_open=(0, 0), session_close=(23, 59), trading_days="mon-fri",
        vol_regime=VolatilityRegime.MEDIUM, liquidity_tier=1,
        region="global", sector="fx", sub_class="commodity_currency",
        corr_group="risk_fx",
    ),
    "DXY": InstrumentSpec(
        symbol="DXY", yf_symbol="DX-Y.NYB", name="US Dollar Index",
        asset_class=AssetClass.FX_MAJOR, currency="USD",
        exchange="ICE", timezone="America/New_York",
        session_open=(0, 0), session_close=(23, 59), trading_days="mon-fri",
        vol_regime=VolatilityRegime.LOW, liquidity_tier=1,
        region="global", sector="fx", sub_class="index",
        corr_group="safe_haven",
        key_events=["us_fomc", "us_cpi", "us_nfp"],
    ),

    # ── EM FX ─────────────────────────────────────────────────────────────────
    "USDINR": InstrumentSpec(
        symbol="USDINR", yf_symbol="INR=X", name="USD/INR",
        asset_class=AssetClass.FX_EM, currency="INR",
        exchange="NSE", timezone="Asia/Kolkata",
        session_open=(9, 0), session_close=(17, 0),
        vol_regime=VolatilityRegime.LOW, liquidity_tier=1,
        region="india", sector="fx", sub_class="em_pair",
        corr_group="em_fx",
        key_events=["rbi_policy", "us_fomc", "india_cad"],
    ),
    "USDBRL": InstrumentSpec(
        symbol="USDBRL", yf_symbol="BRL=X", name="USD/BRL",
        asset_class=AssetClass.FX_EM, currency="BRL",
        exchange="B3", timezone="America/Sao_Paulo",
        session_open=(9, 0), session_close=(18, 0),
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=2,
        region="latam", sector="fx", sub_class="em_pair",
        corr_group="em_fx",
    ),
    "USDMXN": InstrumentSpec(
        symbol="USDMXN", yf_symbol="MXN=X", name="USD/MXN",
        asset_class=AssetClass.FX_EM, currency="MXN",
        exchange="FOREX", timezone="America/Mexico_City",
        session_open=(7, 0), session_close=(17, 0),
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=2,
        region="latam", sector="fx", sub_class="em_pair",
        corr_group="em_fx",
    ),
    "USDCNH": InstrumentSpec(
        symbol="USDCNH", yf_symbol="CNH=X", name="USD/CNH (Offshore RMB)",
        asset_class=AssetClass.FX_EM, currency="CNH",
        exchange="FOREX", timezone="Asia/Hong_Kong",
        session_open=(0, 0), session_close=(23, 59),
        vol_regime=VolatilityRegime.LOW, liquidity_tier=1,
        region="china", sector="fx", sub_class="em_pair",
        corr_group="em_fx",
        key_events=["pboc_fixing", "china_cpi"],
    ),

    # =========================================================================
    # FIXED INCOME
    # =========================================================================

    "ZN": InstrumentSpec(
        symbol="ZN", yf_symbol="ZN=F", name="10Y US Treasury Note",
        asset_class=AssetClass.FIXED_INCOME, currency="USD",
        exchange="CBOT", timezone="America/Chicago",
        session_open=(0, 0), session_close=(23, 59), trading_days="mon-fri",
        vol_regime=VolatilityRegime.LOW, liquidity_tier=1,
        region="us", sector="rates", sub_class="government_bond",
        is_futures=True,
        corr_group="safe_haven",
        key_events=["us_fomc", "us_cpi", "us_nfp", "treasury_auction"],
    ),
    "ZB": InstrumentSpec(
        symbol="ZB", yf_symbol="ZB=F", name="30Y US Treasury Bond",
        asset_class=AssetClass.FIXED_INCOME, currency="USD",
        exchange="CBOT", timezone="America/Chicago",
        session_open=(0, 0), session_close=(23, 59), trading_days="mon-fri",
        vol_regime=VolatilityRegime.LOW, liquidity_tier=1,
        region="us", sector="rates", sub_class="government_bond",
        is_futures=True, corr_group="safe_haven",
    ),
    "FGBL": InstrumentSpec(
        symbol="FGBL", yf_symbol="^TNX", name="Euro Bund (10Y German)",
        asset_class=AssetClass.FIXED_INCOME, currency="EUR",
        exchange="EUREX", timezone="Europe/Berlin",
        session_open=(8, 0), session_close=(22, 0),
        vol_regime=VolatilityRegime.LOW, liquidity_tier=1,
        region="europe", sector="rates", sub_class="government_bond",
        is_futures=True, corr_group="safe_haven",
        key_events=["ecb_meeting", "german_cpi"],
    ),
    "US10Y": InstrumentSpec(
        symbol="US10Y", yf_symbol="^TNX", name="US 10Y Yield",
        asset_class=AssetClass.FIXED_INCOME, currency="USD",
        exchange="OTC", timezone="America/New_York",
        session_open=(0, 0), session_close=(23, 59),
        vol_regime=VolatilityRegime.LOW, liquidity_tier=1,
        region="us", sector="rates", sub_class="yield",
        corr_group="safe_haven",
        key_events=["us_fomc", "us_cpi", "us_nfp"],
    ),

    # =========================================================================
    # VOLATILITY INDICES
    # =========================================================================

    "VIX": InstrumentSpec(
        symbol="VIX", yf_symbol="^VIX", name="CBOE VIX (US Fear Index)",
        asset_class=AssetClass.VOLATILITY, currency="USD",
        exchange="CBOE", timezone="America/New_York",
        session_open=(9, 30), session_close=(16, 15),
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=1,
        region="us", sector="volatility", sub_class="equity_vol",
        corr_group="risk_off",
        key_events=["spx_options_expiry"],
    ),
    "INDIAVIX": InstrumentSpec(
        symbol="INDIAVIX", yf_symbol="^INDIAVIX", name="India VIX",
        asset_class=AssetClass.VOLATILITY, currency="INR",
        exchange="NSE", timezone="Asia/Kolkata",
        session_open=(9, 15), session_close=(15, 30),
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=1,
        region="india", sector="volatility", sub_class="equity_vol",
        corr_group="risk_off",
    ),
    "OVX": InstrumentSpec(
        symbol="OVX", yf_symbol="^OVX", name="Crude Oil Volatility (OVX)",
        asset_class=AssetClass.VOLATILITY, currency="USD",
        exchange="CBOE", timezone="America/New_York",
        session_open=(9, 30), session_close=(16, 15),
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=2,
        region="global", sector="volatility", sub_class="commodity_vol",
        corr_group="energy",
    ),
    "GVZ": InstrumentSpec(
        symbol="GVZ", yf_symbol="^GVZ", name="Gold Volatility (GVZ)",
        asset_class=AssetClass.VOLATILITY, currency="USD",
        exchange="CBOE", timezone="America/New_York",
        session_open=(9, 30), session_close=(16, 15),
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=2,
        region="global", sector="volatility", sub_class="commodity_vol",
        corr_group="safe_haven",
    ),

    # =========================================================================
    # CRYPTO (as alternative risk assets)
    # =========================================================================

    "BTCUSD": InstrumentSpec(
        symbol="BTCUSD", yf_symbol="BTC-USD", name="Bitcoin",
        asset_class=AssetClass.CRYPTO, currency="USD",
        exchange="CRYPTO", timezone="UTC",
        session_open=(0, 0), session_close=(23, 59), trading_days="24/7",
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=1,
        typical_spread_bps=5.0,
        region="global", sector="crypto", sub_class="layer1",
        corr_group="risk_on",
        key_events=["bitcoin_halving", "etf_flows", "regulatory_news"],
    ),
    "ETHUSD": InstrumentSpec(
        symbol="ETHUSD", yf_symbol="ETH-USD", name="Ethereum",
        asset_class=AssetClass.CRYPTO, currency="USD",
        exchange="CRYPTO", timezone="UTC",
        session_open=(0, 0), session_close=(23, 59), trading_days="24/7",
        vol_regime=VolatilityRegime.HIGH, liquidity_tier=1,
        region="global", sector="crypto", sub_class="layer1",
        corr_group="risk_on",
    ),
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def get_instrument(symbol: str) -> InstrumentSpec:
    """Get instrument spec by symbol. Case-insensitive."""
    key = symbol.upper()
    if key not in UNIVERSE:
        raise KeyError(
            f"Instrument '{symbol}' not in universe. "
            f"Available: {sorted(UNIVERSE.keys())}"
        )
    return UNIVERSE[key]


def get_by_asset_class(asset_class: str) -> list[InstrumentSpec]:
    return [i for i in UNIVERSE.values() if i.asset_class == asset_class]


def get_by_region(region: str) -> list[InstrumentSpec]:
    return [i for i in UNIVERSE.values() if i.region == region]


def get_by_corr_group(group: str) -> list[InstrumentSpec]:
    return [i for i in UNIVERSE.values() if i.corr_group == group]


def get_commodities() -> list[InstrumentSpec]:
    return [i for i in UNIVERSE.values() if i.is_commodity]


def get_equity_indices() -> list[InstrumentSpec]:
    return [i for i in UNIVERSE.values() if i.is_equity]


def get_fx() -> list[InstrumentSpec]:
    return [i for i in UNIVERSE.values() if i.is_fx]


def list_symbols(asset_class: str | None = None) -> list[str]:
    if asset_class:
        return sorted(s for s, i in UNIVERSE.items() if i.asset_class == asset_class)
    return sorted(UNIVERSE.keys())


# ---------------------------------------------------------------------------
# Session utilities — global-aware
# ---------------------------------------------------------------------------

def is_instrument_tradeable(symbol: str) -> bool:
    """
    Check if the instrument's primary exchange is currently in session.
    Handles all timezones correctly.
    """
    try:
        import pytz
        spec  = get_instrument(symbol)
        tz    = pytz.timezone(spec.timezone)
        now   = datetime.now(tz)

        if spec.trading_days == "24/7":
            return True

        if now.weekday() >= 5 and spec.trading_days == "mon-fri":
            return False

        open_t  = now.replace(hour=spec.session_open[0],  minute=spec.session_open[1],
                               second=0, microsecond=0)
        close_t = now.replace(hour=spec.session_close[0], minute=spec.session_close[1],
                               second=0, microsecond=0)
        return open_t <= now <= close_t
    except Exception:
        return True   # assume tradeable if we can't determine


def seconds_until_open(symbol: str) -> float:
    """Seconds until the instrument's exchange opens next."""
    try:
        import pytz
        from datetime import timedelta
        spec  = get_instrument(symbol)

        if spec.trading_days == "24/7":
            return 0.0

        tz    = pytz.timezone(spec.timezone)
        now   = datetime.now(tz)
        open_today = now.replace(hour=spec.session_open[0],
                                  minute=spec.session_open[1], second=0)
        if now < open_today and now.weekday() < 5:
            return (open_today - now).total_seconds()

        days = 1
        while True:
            candidate = open_today + timedelta(days=days)
            if candidate.weekday() < 5:
                return (candidate - now).total_seconds()
            days += 1
    except Exception:
        return 3600.0


# ---------------------------------------------------------------------------
# Print universe summary
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{'═'*65}")
    print(f"  MarketMind Global Instrument Universe — {len(UNIVERSE)} instruments")
    print(f"{'═'*65}\n")

    from collections import Counter
    counts = Counter(i.asset_class for i in UNIVERSE.values())
    for ac, n in sorted(counts.items()):
        symbols = [s for s, i in UNIVERSE.items() if i.asset_class == ac]
        print(f"  {ac:<30} {n:>3}  {', '.join(symbols[:6])}"
              + ("..." if n > 6 else ""))

    print(f"\n  Commodities: {len(get_commodities())}")
    print(f"  Equity indices: {len(get_equity_indices())}")
    print(f"  FX pairs: {len(get_fx())}")

    print(f"\n  Sample — CL (WTI Crude):")
    cl = get_instrument("CL")
    print(f"    {cl.describe()}")
    print(f"    Contract size: {cl.contract_size:,.0f} barrels")
    print(f"    Threshold scale: {cl.threshold_scale}x (HIGH vol regime)")
    print(f"    Key events: {cl.key_events}")

    print(f"\n  Tradeable now:")
    for sym in ["SPX", "NIFTY50", "CL", "EURUSD", "GC", "BTCUSD"]:
        try:
            t = is_instrument_tradeable(sym)
            secs = seconds_until_open(sym)
            print(f"    {sym:<12} {'OPEN' if t else f'closed ({int(secs//3600)}h {int((secs%3600)//60)}m)'}")
        except Exception as e:
            print(f"    {sym:<12} error: {e}")
