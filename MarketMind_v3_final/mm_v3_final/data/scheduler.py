"""
data/scheduler.py
=================
Global market scheduler — timezone-aware, 24/5 (or 24/7 for crypto).

Key difference from the India-only version:
  - Sessions are looked up from universe.InstrumentSpec, not hardcoded
  - The 24-hour commodity markets (energy, metals, FX) are handled correctly
  - Multi-instrument watch: monitor a portfolio of instruments simultaneously
  - Cross-asset alert: notify when correlated instruments diverge
"""

from __future__ import annotations

import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.feeds import GlobalFeed, FeedSnapshot
from data.event_factory import EventFactory, EventDetection, snapshot_to_pipeline_inputs
from data.universe import (
    get_instrument, is_instrument_tradeable, seconds_until_open, UNIVERSE
)
from simulation.tiers.tier3_mesa import MarketEvent


# ---------------------------------------------------------------------------
# Scheduler result
# ---------------------------------------------------------------------------

class SchedulerResult:
    def __init__(
        self,
        timestamp:    datetime,
        instrument:   str,
        session:      str,
        detection:    Optional[EventDetection],
        pipeline_ran: bool,
        signal:       object = None,
        error:        str    = "",
    ):
        self.timestamp    = timestamp
        self.instrument   = instrument
        self.session      = session
        self.detection    = detection
        self.pipeline_ran = pipeline_ran
        self.signal       = signal
        self.error        = error

    def __repr__(self) -> str:
        sig = ""
        if self.signal and hasattr(self.signal, "signal_summary"):
            sig = f" → {self.signal.signal_summary[:80]}"
        return (
            f"[{self.timestamp.strftime('%H:%M:%S')} UTC] "
            f"{self.instrument} {self.session} "
            f"{'▶ RAN' if self.pipeline_ran else '─ skip'}"
            f"{sig}"
        )


# ---------------------------------------------------------------------------
# Session status for any instrument
# ---------------------------------------------------------------------------

def get_session_status(symbol: str) -> dict:
    """Return session info dict for any instrument."""
    try:
        spec   = get_instrument(symbol)
        open_  = is_instrument_tradeable(symbol)
        secs   = seconds_until_open(symbol) if not open_ else 0.0
        h, m   = int(secs // 3600), int((secs % 3600) // 60)
        return {
            "symbol":        symbol,
            "name":          spec.name,
            "exchange":      spec.exchange,
            "timezone":      spec.timezone,
            "asset_class":   spec.asset_class,
            "is_open":       open_,
            "trading_days":  spec.trading_days,
            "next_open_in":  f"{h}h {m}m" if not open_ else "NOW",
            "session":       "market" if open_ else "closed",
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


# ---------------------------------------------------------------------------
# Single-instrument scheduler
# ---------------------------------------------------------------------------

class LiveScheduler:
    """
    Polls a single instrument and triggers the pipeline on events.
    Handles all timezones, all asset classes, all session schedules.
    """

    def __init__(
        self,
        pipeline,
        instrument:    str   = "CL",
        exchange:      str   = "",
        asset_class:   str   = "",
        poll_interval: int   = 300,
        sensitivity:   str   = "medium",
        feed_kwargs:   dict  = None,
        on_signal:     Optional[Callable] = None,
        verbose:       bool  = True,
    ):
        self.pipeline      = pipeline
        self.instrument    = instrument.upper()
        self.poll_interval = poll_interval
        self.on_signal     = on_signal
        self.verbose       = verbose

        # Resolve instrument spec
        try:
            spec = get_instrument(self.instrument)
            self.exchange    = exchange    or spec.exchange
            self.asset_class = asset_class or spec.asset_class
        except KeyError:
            self.exchange    = exchange    or "UNKNOWN"
            self.asset_class = asset_class or "equity_index"

        self.feed    = GlobalFeed(**(feed_kwargs or {}))
        self.factory = EventFactory(sensitivity=sensitivity)

        self._running      = False
        self._poll_count   = 0
        self._signal_count = 0
        self._last_snap:   Optional[FeedSnapshot] = None

        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, sig, frame):
        print(f"\n  [Scheduler] Stopped. {self._poll_count} polls, "
              f"{self._signal_count} signals.")
        self._running = False
        sys.exit(0)

    def run_once(
        self,
        force_run:  bool = False,
    ) -> SchedulerResult:
        self._poll_count += 1
        ts      = datetime.utcnow()
        status  = get_session_status(self.instrument)
        session = status.get("session", "unknown")

        if self.verbose:
            print(f"\n  [{ts.strftime('%H:%M')} UTC] {self.instrument} "
                  f"({status.get('exchange', '')} {session})...")

        snap = self.feed.fetch_safe(self.instrument)
        self._last_snap = snap

        if snap.errors and self.verbose:
            print(f"  ⚠ {snap.errors}")

        if self.verbose and not snap.is_stale:
            print(f"  {snap.spot:>14,.5f} {snap.currency}  "
                  f"Chg: {snap.change_pct:>+6.2f}%  "
                  f"HVol: {snap.hist_vol_20:.0f}%  "
                  f"VIX: {snap.vix:.1f}  "
                  f"ATR%: {snap.atr_pct:.2f}%")

        event, market_data, detection = snapshot_to_pipeline_inputs(
            snap, self.factory, self.instrument, self.exchange, self.asset_class
        )

        if self.verbose:
            if detection.triggered:
                print(f"  ⚡ [{detection.trigger_type}] sig={detection.significance:.2f} "
                      f"triggers={detection.triggers_fired[:3]}")
            else:
                print(f"  ─ {detection.rationale}")

        # Always run on first poll (pre-session snapshot)
        if self._poll_count == 1:
            force_run = True

        should_run = force_run or self.factory.should_run_pipeline(detection)

        if not should_run:
            return SchedulerResult(
                timestamp=ts, instrument=self.instrument,
                session=session, detection=detection, pipeline_ran=False,
            )

        # Build neutral event if no trigger but forcing run
        if event is None:
            event = MarketEvent(
                description    = _neutral_description(snap, session),
                price_shock    = max(-0.05, min(0.05, snap.change_pct / 100.0)),
                news_sentiment = snap.news_sentiment_score or snap.change_pct / 10.0,
                uncertainty    = 0.30,
                is_systemic    = False,
                sector_impact  = "all",
            )

        if self.verbose:
            print(f"  ▶ Running pipeline...")

        try:
            result = self.pipeline.run(
                event        = event,
                instrument   = self.instrument,
                exchange     = self.exchange,
                asset_class  = self.asset_class,
                market_data  = market_data,
            )
            self._signal_count += 1

            if self.on_signal and result.signal:
                try:
                    self.on_signal(result.signal, detection, snap)
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠ callback error: {e}")

            return SchedulerResult(
                timestamp=ts, instrument=self.instrument, session=session,
                detection=detection, pipeline_ran=True, signal=result.signal,
            )
        except Exception as e:
            if self.verbose:
                print(f"  ✗ Pipeline error: {e}")
            return SchedulerResult(
                timestamp=ts, instrument=self.instrument, session=session,
                detection=detection, pipeline_ran=True, error=str(e),
            )

    def watch(self, run_outside_hours: bool = False, max_signals: int = 0) -> None:
        self._running = True
        spec = None
        try:
            spec = get_instrument(self.instrument)
        except Exception:
            pass

        print(f"\n{'═'*62}")
        print(f"  MarketMind Live Scheduler — {self.instrument}")
        print(f"  {spec.name if spec else ''} | {self.exchange} | {self.asset_class}")
        print(f"  Poll: {self.poll_interval}s | Sensitivity: {self.factory.sensitivity}")
        print(f"  Press Ctrl+C to stop")
        print(f"{'═'*62}\n")

        while self._running:
            is_open = is_instrument_tradeable(self.instrument)

            # For instruments that trade 24/7 or near-24h, always run
            trading_days = spec.trading_days if spec else "mon-fri"
            always_open  = trading_days == "24/7"

            if not is_open and not always_open and not run_outside_hours:
                secs = seconds_until_open(self.instrument)
                h, m = int(secs // 3600), int((secs % 3600) // 60)
                print(f"  Market closed. Opens in {h}h {m}m. Sleeping 15 min...")
                for _ in range(min(900, int(secs))):
                    if not self._running:
                        break
                    time.sleep(1)
                continue

            self.run_once()

            if max_signals > 0 and self._signal_count >= max_signals:
                print(f"\n  Max signals ({max_signals}) reached.")
                break

            if self.verbose:
                print(f"  Next poll in {self.poll_interval}s...\n")
            for _ in range(self.poll_interval):
                if not self._running:
                    break
                time.sleep(1)


# ---------------------------------------------------------------------------
# Multi-instrument portfolio scheduler
# ---------------------------------------------------------------------------

class PortfolioScheduler:
    """
    Watch multiple instruments simultaneously.
    Useful for a commodities quant monitoring a basket:
    e.g. [CL, BZ, GC, SI, HG, ZW, ZC, EURUSD, DXY]

    Runs each instrument on its own poll cycle respecting its session.
    Fires pipeline when any instrument triggers an event.
    Optionally fires cross-asset alert when instruments diverge.
    """

    def __init__(
        self,
        pipeline,
        instruments:   list[str],
        poll_interval: int  = 300,
        sensitivity:   str  = "medium",
        feed_kwargs:   dict = None,
        on_signal:     Optional[Callable] = None,
        verbose:       bool = True,
    ):
        self.pipeline      = pipeline
        self.instruments   = [i.upper() for i in instruments]
        self.poll_interval = poll_interval
        self.on_signal     = on_signal
        self.verbose       = verbose

        self.feed    = GlobalFeed(**(feed_kwargs or {}))
        self.factory = EventFactory(sensitivity=sensitivity)

        self._running      = False
        self._total_polls  = 0
        self._total_signals = 0
        self._snapshots:    dict[str, FeedSnapshot] = {}

        signal.signal(signal.SIGINT, self._handle_interrupt)

    def _handle_interrupt(self, sig, frame):
        print(f"\n  [Portfolio] Stopped. {self._total_polls} polls, "
              f"{self._total_signals} signals across {len(self.instruments)} instruments.")
        self._running = False
        sys.exit(0)

    def poll_all(self) -> list[SchedulerResult]:
        """Poll all instruments once. Returns list of results."""
        self._total_polls += 1
        results = []

        for sym in self.instruments:
            try:
                spec    = get_instrument(sym)
                is_open = is_instrument_tradeable(sym)
                # Skip closed instruments (except 24h markets)
                if not is_open and spec.trading_days == "mon-fri":
                    continue
            except Exception:
                pass

            snap = self.feed.fetch_safe(sym)
            self._snapshots[sym] = snap

            event, market_data, detection = snapshot_to_pipeline_inputs(
                snap, self.factory, sym
            )

            if self.verbose and detection.triggered:
                print(f"  ⚡ {sym:<12} [{detection.trigger_type:<25}] "
                      f"sig={detection.significance:.2f}")

            if not self.factory.should_run_pipeline(detection):
                continue

            if event is None:
                event = MarketEvent(
                    description    = _neutral_description(snap, "market"),
                    price_shock    = max(-0.05, min(0.05, snap.change_pct / 100.0)),
                    news_sentiment = snap.news_sentiment_score or snap.change_pct / 10.0,
                    uncertainty    = 0.30,
                    is_systemic    = False,
                    sector_impact  = "all",
                )

            try:
                spec2 = get_instrument(sym)
                ac    = spec2.asset_class
                exch  = spec2.exchange
            except Exception:
                ac, exch = snap.asset_class, snap.exchange

            try:
                result = self.pipeline.run(
                    event=event, instrument=sym,
                    exchange=exch, asset_class=ac,
                    market_data=market_data,
                )
                self._total_signals += 1

                if self.on_signal and result.signal:
                    self.on_signal(result.signal, detection, snap)

                results.append(SchedulerResult(
                    timestamp=datetime.utcnow(), instrument=sym, session="market",
                    detection=detection, pipeline_ran=True, signal=result.signal,
                ))
            except Exception as e:
                results.append(SchedulerResult(
                    timestamp=datetime.utcnow(), instrument=sym, session="market",
                    detection=detection, pipeline_ran=True, error=str(e),
                ))

        return results

    def watch(self, max_signals: int = 0) -> None:
        self._running = True

        print(f"\n{'═'*64}")
        print(f"  MarketMind Portfolio Scheduler — {len(self.instruments)} instruments")
        print(f"  Watching: {', '.join(self.instruments)}")
        print(f"  Poll: {self.poll_interval}s | Sensitivity: {self.factory.sensitivity}")
        print(f"  Press Ctrl+C to stop")
        print(f"{'═'*64}\n")

        while self._running:
            ts = datetime.utcnow().strftime("%H:%M UTC")
            print(f"  ── Poll {self._total_polls+1} [{ts}] ──────────────────────────────")

            results = self.poll_all()
            if results:
                for r in results:
                    print(f"  {r}")

            if max_signals > 0 and self._total_signals >= max_signals:
                print(f"\n  Max signals reached.")
                break

            print(f"\n  Next poll in {self.poll_interval}s...\n")
            for _ in range(self.poll_interval):
                if not self._running:
                    break
                time.sleep(1)

    def status_board(self) -> None:
        """Print a live status board for all watched instruments."""
        print(f"\n  {'Symbol':<12} {'Spot':>14} {'Chg%':>8} {'HVol':>6} {'RSI':>5} {'Open?':<7} {'Session'}")
        print(f"  {'─'*75}")
        for sym in self.instruments:
            try:
                snap   = self._snapshots.get(sym)
                is_open = is_instrument_tradeable(sym)
                sess   = "OPEN" if is_open else "closed"
                if snap:
                    print(f"  {sym:<12} {snap.spot:>14,.4f} {snap.change_pct:>+7.2f}% "
                          f"{snap.hist_vol_20:>5.0f}% {snap.rsi_14:>5.0f} {sess:<7}")
                else:
                    print(f"  {sym:<12} {'no data':>14}                    {sess:<7}")
            except Exception as e:
                print(f"  {sym:<12} error: {e}")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _neutral_description(snap: FeedSnapshot, session: str) -> str:
    label = {"pre_open": "Pre-session snapshot", "post_close": "Post-close snapshot"}.get(
        session, "Scheduled analysis"
    )
    return (
        f"{label}: {snap.instrument} at {snap.spot:,.5f} {snap.currency} "
        f"({snap.change_pct:+.2f}% today). "
        f"HVol: {snap.hist_vol_20:.0f}%, RSI: {snap.rsi_14:.0f}, "
        f"ATR%: {snap.atr_pct:.2f}%, VIX: {snap.vix:.1f}."
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data.feeds import FeedSnapshot
    from data.event_factory import EventFactory, snapshot_to_pipeline_inputs

    print("\n=== Global Scheduler Test ===\n")

    factory = EventFactory(sensitivity="medium")

    # Test with a simulated CL snapshot (no live data needed)
    cl_snap = FeedSnapshot(
        instrument="CL", exchange="NYMEX", asset_class="commodity_energy",
        currency="USD", timestamp=datetime.now(),
        spot=71.50, prev_close=74.20, change_pct=-3.64,
        atr_pct=2.1, hist_vol_20=38.0, rsi_14=32,
        vix=22.0, dxy=105.2, us_10y=4.6, gold_usd=2920.0, crude_wti=71.5,
        inventory_change=-5.2, volume_ratio=2.8,
    )

    event, market_data, detection = snapshot_to_pipeline_inputs(
        cl_snap, factory, "CL", "NYMEX", "commodity_energy"
    )

    print(f"  CL event detection:")
    print(f"    Triggered:    {detection.triggered}")
    print(f"    Triggers:     {detection.triggers_fired}")
    print(f"    Significance: {detection.significance:.3f}")
    print(f"    Asset class:  {detection.asset_class}")
    if event:
        print(f"    Event desc:   {event.description[:100]}...")

    print(f"\n  Session status for key instruments:")
    for sym in ["CL", "GC", "EURUSD", "SPX", "NIFTY50", "BTCUSD", "ZW"]:
        s = get_session_status(sym)
        print(f"    {sym:<10} {s.get('exchange',''):<8} "
              f"{'OPEN' if s.get('is_open') else 'closed':>7}  "
              f"({s.get('next_open_in',''):<12}) "
              f"{s.get('asset_class','')}")
