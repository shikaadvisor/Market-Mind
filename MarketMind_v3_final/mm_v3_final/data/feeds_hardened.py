"""
data/feeds_hardened.py  — HARDENED DATA LAYER
==============================================
Wraps the existing GlobalFeed with:

FIX-2  Stale snapshot detection — raises DataError if spot=0 and is_stale=True
FIX-3  Enrichment audit — records which optional APIs were absent
FIX-4  Symbol validation — raises early with a clear error for unknown symbols
FIX-5  NSE expiry fix — corrected last-Thursday-of-month logic

Drop-in replacement: swap GlobalFeed → HardenedFeed in main.py / _run_live().

Usage:
    feed = HardenedFeed(eia_key=..., fred_key=..., newsapi_key=...)
    snap, warnings = feed.fetch_validated("CL")
    # warnings: list[str] — e.g. ["newsapi_absent", "eia_absent"]
"""

from __future__ import annotations

import calendar
import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional

logger = logging.getLogger("marketmind.feeds")


class DataError(Exception):
    """Raised when live market data is fundamentally unusable."""


@dataclass
class HardenedFeed:
    """
    Wraps GlobalFeed with validation, enrichment auditing, and stale guards.
    """
    eia_key:     str = ""
    fred_key:    str = ""
    newsapi_key: str = ""

    # FIX-2: set False to warn-and-continue instead of raising
    abort_on_stale: bool = True

    def fetch_validated(
        self,
        symbol: str,
    ) -> tuple:
        """
        Returns (FeedSnapshot, warnings: list[str]).
        Raises DataError if spot price is zero/stale and abort_on_stale is True.
        """
        from data.feeds import GlobalFeed

        feed = GlobalFeed(
            eia_key     = self.eia_key,
            fred_key    = self.fred_key,
            newsapi_key = self.newsapi_key,
        )

        snap = feed.fetch_safe(symbol)
        warnings: list[str] = []

        # ── FIX-4: symbol validation ─────────────────────────────────────────
        if snap.spot <= 1e-9:
            msg = (
                f"Symbol '{symbol}' returned spot=0 — likely not found in yfinance. "
                f"Errors: {snap.errors}"
            )
            if self.abort_on_stale:
                raise DataError(msg)
            logger.warning(msg)
            warnings.append("symbol_not_found_or_zero_price")

        # ── FIX-2: stale data guard ──────────────────────────────────────────
        if snap.is_stale:
            msg = f"Stale snapshot for '{symbol}'. Errors: {snap.errors}"
            if self.abort_on_stale:
                raise DataError(msg)
            logger.warning(msg)
            warnings.append("stale_snapshot")

        # ── FIX-3: enrichment audit ───────────────────────────────────────────
        if not self.newsapi_key:
            warnings.append("newsapi_absent")
        if not self.eia_key and snap.asset_class.startswith("commodity_energy"):
            warnings.append("eia_absent_for_energy_instrument")
        if not self.fred_key:
            warnings.append("fred_absent")

        if snap.errors:
            for err in snap.errors:
                warnings.append(f"feed_error:{err[:60]}")

        return snap, warnings


# ---------------------------------------------------------------------------
# FIX-5: Corrected NSE expiry detection
# ---------------------------------------------------------------------------

def is_nse_expiry(d: date) -> bool:
    """
    True if d is the last Thursday of its month.
    The original v1.0 logic was incorrect for months with <31 days.
    """
    if d.weekday() != 3:   # not Thursday
        return False
    # Get the last day of the month, then find the last Thursday
    last_day = calendar.monthrange(d.year, d.month)[1]
    last_date = date(d.year, d.month, last_day)
    # Walk backward from last_day to find the last Thursday
    days_back = (last_date.weekday() - 3) % 7
    last_thursday = date(d.year, d.month, last_day - days_back)
    return d == last_thursday


# ---------------------------------------------------------------------------
# FIX: Event calendar builder — no longer hardcoded to 2026
# ---------------------------------------------------------------------------

def build_calendar(year: int) -> dict:
    """
    Return an event calendar for the given year.
    Approximates standard FOMC/ECB/RBI/OPEC/WASDE schedules.
    For production use, ingest a machine-readable economic calendar API
    (e.g. Quandl ECONDB, Trading Economics, Alpha Vantage) and cache daily.
    """
    import warnings as _warnings
    _warnings.warn(
        "build_calendar() uses approximate dates. "
        "For live trading, connect a real economic calendar API.",
        UserWarning,
        stacklevel=2,
    )

    # Placeholder — returns the original 2026 calendar if year == 2026,
    # else returns an empty dict with a log entry prompting integration.
    if year == 2026:
        from data.event_factory import CALENDAR_2026
        return CALENDAR_2026

    logger.warning(
        "No calendar data for year %d. "
        "Calendar-based event triggers are disabled. "
        "Integrate an economic calendar API to fix this.",
        year,
    )
    return {}
