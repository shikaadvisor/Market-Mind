#!/usr/bin/env python3
"""
main.py — MarketMind Global Trading Signal System

Usage:
    python main.py --mock                          # demo scenario, no API keys
    python main.py --scenario rbi_hold             # named scenario
    python main.py --demo --mock                   # all scenarios
    python main.py --log                           # view signal history
    python main.py --universe                      # list all tradeable instruments

    # Live single instrument:
    python main.py --live --instrument CL          # WTI Crude Oil
    python main.py --live --instrument GC          # Gold
    python main.py --live --instrument EURUSD      # EUR/USD
    python main.py --live --instrument ZW          # Wheat
    python main.py --live --detect-only            # event detection only, no pipeline

    # Watch modes:
    python main.py --watch --instrument CL         # continuous single instrument
    python main.py --portfolio energy              # watch energy basket
    python main.py --portfolio metals              # watch metals basket
    python main.py --portfolio fx                  # watch FX majors
    python main.py --portfolio india               # watch India markets
    python main.py --portfolio us_equity           # watch US indices
    python main.py --portfolio commodities         # all commodities

    # Custom portfolio:
    python main.py --watch-list CL BZ GC SI EURUSD DXY
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from simulation.tiers.tier3_mesa import MarketEvent
from pipeline.runner import MarketMindPipeline, PipelineConfig
from pipeline.dashboard import display_pipeline_result, display_signal_log


# ---------------------------------------------------------------------------
# Predefined baskets for --portfolio
# ---------------------------------------------------------------------------

PORTFOLIOS = {
    "energy":      ["CL", "BZ", "NG", "HO", "RB"],
    "metals":      ["GC", "SI", "HG", "PL", "PA"],
    "precious":    ["GC", "SI", "PL", "PA"],
    "agriculture": ["ZW", "ZC", "ZS", "KC", "SB", "CT"],
    "commodities": ["CL", "BZ", "GC", "SI", "HG", "ZW", "ZC"],
    "fx":          ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "DXY"],
    "em_fx":       ["USDINR", "USDBRL", "USDMXN", "USDCNH"],
    "us_equity":   ["SPX", "NDX", "DJI", "RUT", "VIX"],
    "asia":        ["NIKKEI", "HSI", "KOSPI", "NIFTY50", "ASX200"],
    "europe":      ["DAX", "FTSE", "CAC", "STOXX50"],
    "india":       ["NIFTY50", "BANKNIFTY", "USDINR", "INDIAVIX"],
    "rates":       ["ZN", "ZB", "US10Y"],
    "safe_haven":  ["GC", "XAUUSD", "USDJPY", "VIX", "ZN"],
    "risk_on":     ["SPX", "CL", "HG", "EURUSD", "BTCUSD"],
    "crypto":      ["BTCUSD", "ETHUSD"],
    "global":      ["SPX", "CL", "GC", "EURUSD", "NIFTY50", "VIX", "US10Y"],
}

# ---------------------------------------------------------------------------
# Named scenarios (keep existing India ones + add global)
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, dict] = {
    "rbi_hold": {
        "event": MarketEvent(
            description    = "RBI holds repo rate at 6.5% — tone slightly dovish, "
                             "signals potential cut in June if inflation stays benign",
            price_shock=+0.008, news_sentiment=+0.45, uncertainty=0.30,
            is_systemic=True, sector_impact="banking",
        ),
        "instrument": "NIFTY50", "exchange": "NSE", "asset_class": "equity",
        "market_data": {"spot": 24280, "india_vix": 13.2, "fii_net_weekly": 1200,
                        "dii_net_weekly": 3400, "usd_inr": 83.8},
    },
    "us_cpi_hot": {
        "event": MarketEvent(
            description    = "US CPI prints 3.4% vs 2.9% expected — hot inflation, "
                             "Fed cuts pushed to Q4 2026",
            price_shock=-0.018, news_sentiment=-0.65, uncertainty=0.70,
            is_systemic=True, sector_impact="all",
        ),
        "instrument": "SPX", "exchange": "NYSE", "asset_class": "equity",
        "market_data": {"spot": 5280, "vix": 22.5, "dxy": 106.2,
                        "us_10y": 4.85, "gold_usd": 2890.0},
    },
    "opec_cut": {
        "event": MarketEvent(
            description    = "OPEC+ agrees surprise 500kbpd additional production cut "
                             "starting April 2026 — Saudi Arabia leads the decision",
            price_shock=+0.038, news_sentiment=+0.72, uncertainty=0.45,
            is_systemic=False, sector_impact="energy",
        ),
        "instrument": "CL", "exchange": "NYMEX", "asset_class": "commodity",
        "market_data": {"spot": 76.80, "atr_pct": 2.1, "hist_vol_20": 38.0,
                        "inventory_change": -4.2, "vix": 16.5, "dxy": 103.8},
    },
    "gold_breakout": {
        "event": MarketEvent(
            description    = "Gold breaks $3,000/oz — new all-time high driven by "
                             "safe-haven demand amid geopolitical escalation and "
                             "central bank buying accelerating",
            price_shock=+0.022, news_sentiment=+0.55, uncertainty=0.50,
            is_systemic=False, sector_impact="precious_metals",
        ),
        "instrument": "GC", "exchange": "COMEX", "asset_class": "commodity",
        "market_data": {"spot": 3010.0, "atr_pct": 0.9, "hist_vol_20": 13.5,
                        "rsi_14": 74, "vix": 18.2, "dxy": 102.1, "us_10y": 4.25},
    },
    "wheat_wasde": {
        "event": MarketEvent(
            description    = "USDA WASDE cuts 2026/27 global wheat ending stocks by "
                             "12mt on poor Australia harvest and Ukraine export disruption",
            price_shock=+0.045, news_sentiment=+0.60, uncertainty=0.55,
            is_systemic=False, sector_impact="grains",
        ),
        "instrument": "ZW", "exchange": "CBOT", "asset_class": "commodity",
        "market_data": {"spot": 588.0, "atr_pct": 1.9, "hist_vol_20": 28.0,
                        "rsi_14": 62, "vix": 15.8, "dxy": 104.5},
    },
    "eurusd_ecb": {
        "event": MarketEvent(
            description    = "ECB delivers 50bp cut surprise vs 25bp expected — "
                             "Lagarde signals more cuts ahead if disinflation continues",
            price_shock=+0.012, news_sentiment=+0.58, uncertainty=0.40,
            is_systemic=True, sector_impact="fx",
        ),
        "instrument": "EURUSD", "exchange": "FOREX", "asset_class": "fx",
        "market_data": {"spot": 1.0920, "atr_pct": 0.42, "hist_vol_20": 6.8,
                        "rsi_14": 58, "vix": 16.0, "dxy": 102.5,
                        "rate_differential": -1.5},
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MarketMind Global — AI trading signals across all asset classes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mock",        action="store_true")
    p.add_argument("--scenario",    type=str, default="rbi_hold",
                   choices=list(SCENARIOS.keys()))
    p.add_argument("--demo",        action="store_true",
                   help="Run all 6 scenarios")
    p.add_argument("--log",         action="store_true")
    p.add_argument("--outcomes",    action="store_true",
                   help="View outcome log")
    p.add_argument("--report",      action="store_true",
                   help="Generate signal quality report")
    p.add_argument("--record",      nargs="+", metavar="ARG",
                   help="Record: --record SIG_ID win 74.20 76.80 notes")
    p.add_argument("--universe",    action="store_true",
                   help="List all instruments in the universe and exit")

    # Live modes
    p.add_argument("--live",        action="store_true",
                   help="Fetch live data + run pipeline if event detected")
    p.add_argument("--watch",       action="store_true",
                   help="Continuous watch for single instrument")
    p.add_argument("--portfolio",   type=str, choices=list(PORTFOLIOS.keys()),
                   help="Watch a predefined portfolio basket")
    p.add_argument("--watch-list",  nargs="+", metavar="SYMBOL",
                   help="Watch custom list: --watch-list CL GC EURUSD")

    # Instrument selection
    p.add_argument("--instrument",  type=str, default="NIFTY50",
                   help="Instrument symbol for --live/--watch")

    # Options
    p.add_argument("--poll",        type=int, default=300)
    p.add_argument("--sensitivity", type=str, default="medium",
                   choices=["high", "medium", "low"])
    p.add_argument("--detect-only", action="store_true")
    p.add_argument("--force-run",   action="store_true")
    p.add_argument("--no-brief",    action="store_true")
    p.add_argument("--quiet",       action="store_true")
    return p.parse_args()


def _load_keys(mock: bool) -> tuple[str, str, str, str, str]:
    """Returns (openai, anthropic, newsapi, eia, fred)"""
    if mock:
        return "mock", "mock", "", "", ""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    openai_key    = os.environ.get("OPENAI_API_KEY",    "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    newsapi_key   = os.environ.get("NEWSAPI_KEY",       "")
    eia_key       = os.environ.get("EIA_API_KEY",       "")
    fred_key      = os.environ.get("FRED_API_KEY",      "")
    if not openai_key or not anthropic_key:
        print("\n  ✗ API keys not found. Set in .env or use --mock.\n")
        sys.exit(1)
    return openai_key, anthropic_key, newsapi_key, eia_key, fred_key


def _run_live(args, pipeline, feed_kwargs) -> int:
    from data.feeds import GlobalFeed
    from data.event_factory import EventFactory, snapshot_to_pipeline_inputs
    from pipeline.dashboard import display_pipeline_result

    sym  = args.instrument.upper()
    print(f"\n  Fetching live data for {sym}...")

    feed    = GlobalFeed(**feed_kwargs)
    factory = EventFactory(sensitivity=args.sensitivity)
    snap    = feed.fetch_safe(sym)

    print(f"\n  Live snapshot — {sym} ({snap.asset_class})")
    print(f"    Spot:       {snap.spot:>14,.5f} {snap.currency}")
    print(f"    Change:     {snap.change_pct:>+8.3f}%")
    print(f"    ATR%:       {snap.atr_pct:>8.2f}%")
    print(f"    HVol-20:    {snap.hist_vol_20:>8.1f}%")
    print(f"    RSI-14:     {snap.rsi_14:>8.0f}")
    print(f"    Vol ratio:  {snap.volume_ratio:>8.2f}x")
    print(f"    ── Global macro ──")
    print(f"    DXY:        {snap.dxy:>8.2f}")
    print(f"    VIX:        {snap.vix:>8.2f}")
    print(f"    US10Y:      {snap.us_10y:>8.2f}%")
    print(f"    Gold:       {snap.gold_usd:>8.0f}")
    print(f"    Crude WTI:  {snap.crude_wti:>8.2f}")
    if snap.top_headlines:
        print(f"    News:       {snap.top_headlines[0][:65]}")
    if snap.errors:
        print(f"    Errors:     {snap.errors}")

    event, market_data, detection = snapshot_to_pipeline_inputs(
        snap, factory, sym
    )

    print(f"\n  Event detection:")
    print(f"    Triggered:    {detection.triggered}")
    if detection.triggered:
        print(f"    Triggers:     {detection.triggers_fired}")
        print(f"    Significance: {detection.significance:.3f}")
        print(f"    Description:  {detection.description[:100]}...")
    else:
        print(f"    Reason:       {detection.rationale}")

    if args.detect_only:
        print("\n  [--detect-only] stopping before pipeline.")
        return 0

    if not detection.triggered and not args.force_run:
        print(f"\n  No event. Use --force-run to run anyway.")
        return 0

    if event is None:
        try:
            from data.universe import get_instrument
            spec = get_instrument(sym)
            name = spec.name
        except Exception:
            name = sym
        event = MarketEvent(
            description    = f"Live analysis: {name} at {snap.spot:,.4f} {snap.currency} "
                             f"({snap.change_pct:+.2f}% today). HVol {snap.hist_vol_20:.0f}%.",
            price_shock    = max(-0.05, min(0.05, snap.change_pct / 100.0)),
            news_sentiment = snap.news_sentiment_score or snap.change_pct / 10.0,
            uncertainty    = 0.35,
            is_systemic    = False,
            sector_impact  = "all",
        )

    try:
        from data.universe import get_instrument
        spec = get_instrument(sym)
        exchange, asset_class = spec.exchange, spec.asset_class
    except Exception:
        exchange, asset_class = snap.exchange, snap.asset_class

    result = pipeline.run(
        event=event, instrument=sym, exchange=exchange,
        asset_class=asset_class, market_data=market_data,
    )
    display_pipeline_result(result, show_brief=False)
    return 0 if result.success else 1


def _run_watch(args, pipeline, feed_kwargs) -> int:
    from data.scheduler import LiveScheduler
    from pipeline.dashboard import display_signal

    def on_signal(sig, det, snap):
        display_signal(sig)

    scheduler = LiveScheduler(
        pipeline      = pipeline,
        instrument    = args.instrument.upper(),
        poll_interval = args.poll,
        sensitivity   = args.sensitivity,
        feed_kwargs   = feed_kwargs,
        on_signal     = on_signal,
        verbose       = not args.quiet,
    )
    scheduler.watch()
    return 0


def _run_portfolio(args, pipeline, feed_kwargs, symbols) -> int:
    from data.scheduler import PortfolioScheduler
    from pipeline.dashboard import display_signal

    def on_signal(sig, det, snap):
        display_signal(sig)

    scheduler = PortfolioScheduler(
        pipeline      = pipeline,
        instruments   = symbols,
        poll_interval = args.poll,
        sensitivity   = args.sensitivity,
        feed_kwargs   = feed_kwargs,
        on_signal     = on_signal,
        verbose       = not args.quiet,
    )
    scheduler.watch()
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if args.log:
        display_signal_log(n=20)
        return 0

    if args.outcomes:
        from pipeline.outcome_tracker import OutcomeTracker
        tracker = OutcomeTracker()
        tracker.print_recent(n=20)
        return 0

    if args.report:
        from pipeline.outcome_tracker import OutcomeTracker
        tracker = OutcomeTracker()
        report  = tracker.generate_report()
        print(report.summary())
        print("\n  Tuning recommendations:")
        for rec in report.tuning_recommendations():
            print(f"  • {rec}")
        return 0

    if args.record:
        from pipeline.outcome_tracker import OutcomeTracker
        tracker = OutcomeTracker()
        parts   = args.record
        if len(parts) < 2:
            print("  Usage: --record SIGNAL_ID outcome [entry] [exit] ['notes']")
            return 1
        signal_id   = parts[0]
        outcome     = parts[1]
        actual_entry = float(parts[2]) if len(parts) > 2 else None
        actual_exit  = float(parts[3]) if len(parts) > 3 else None
        notes        = parts[4] if len(parts) > 4 else ""
        tracker.record_outcome(signal_id, outcome, actual_entry, actual_exit, notes)
        return 0

    if args.universe:
        from data.universe import UNIVERSE
        print(f"\n  MarketMind Universe — {len(UNIVERSE)} instruments\n")
        from collections import Counter
        counts = Counter(i.asset_class for i in UNIVERSE.values())
        for ac, n in sorted(counts.items()):
            syms = sorted(s for s, i in UNIVERSE.items() if i.asset_class == ac)
            print(f"  {ac:<30} ({n:>2})  {', '.join(syms)}")
        return 0

    openai_key, anthropic_key, newsapi_key, eia_key, fred_key = _load_keys(args.mock)

    cfg = PipelineConfig(
        openai_api_key    = openai_key,
        anthropic_api_key = anthropic_key,
        mock              = args.mock,
        verbose           = not args.quiet,
        show_brief        = not args.no_brief and not args.quiet,
    )
    pipeline = MarketMindPipeline(cfg)

    feed_kwargs = {
        "eia_key":     eia_key,
        "fred_key":    fred_key,
        "newsapi_key": newsapi_key,
    }

    # ── Portfolio watch ───────────────────────────────────────────────────────
    if args.watch_list:
        return _run_portfolio(args, pipeline, feed_kwargs, args.watch_list)

    if args.portfolio:
        symbols = PORTFOLIOS[args.portfolio]
        print(f"\n  Portfolio: {args.portfolio} ({len(symbols)} instruments)")
        print(f"  Symbols: {', '.join(symbols)}")
        return _run_portfolio(args, pipeline, feed_kwargs, symbols)

    # ── Single instrument live modes ─────────────────────────────────────────
    if args.watch:
        return _run_watch(args, pipeline, feed_kwargs)

    if args.live:
        return _run_live(args, pipeline, feed_kwargs)

    # ── Named scenario mode ───────────────────────────────────────────────────
    scenarios_to_run = (
        list(SCENARIOS.items()) if args.demo
        else [(args.scenario, SCENARIOS[args.scenario])]
    )

    exit_code = 0
    for name, scenario in scenarios_to_run:
        if args.demo:
            print(f"\n{'#'*64}\n  SCENARIO: {name.upper()}\n{'#'*64}")

        result = pipeline.run(
            event        = scenario["event"],
            instrument   = scenario.get("instrument", "NIFTY50"),
            exchange     = scenario.get("exchange",   "NSE"),
            asset_class  = scenario.get("asset_class", "equity"),
            market_data  = scenario.get("market_data"),
        )
        display_pipeline_result(result, show_brief=False)

        if not result.success:
            exit_code = 1

        if args.demo and len(scenarios_to_run) > 1:
            input("\n  Press Enter for next scenario...\n")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
