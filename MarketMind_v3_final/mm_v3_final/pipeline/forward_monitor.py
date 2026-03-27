"""
pipeline/forward_monitor.py  —  v3 FIXED
=========================================
Risk 5 fix:  SUSPENDED status added for market halts / circuit breakers.
             suspend_by_instrument() pauses all active signals for a symbol
             without resolving them — they resume automatically when trading
             resumes (price polling only runs for non-suspended signals).
             Suspension is logged with reason and timestamp.

Seal fix:    do_not_trade_flags and warning_flags now covered by hash.
             (Seal was missing these — a retroactive veto injection would
             not have broken it. Now it does.)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger("marketmind.monitor")

TIMEFRAME_EXPIRY_DAYS = {
    "intraday":   1,
    "swing":      10,
    "positional": 56,
}

# All terminal statuses (not resumable)
TERMINAL_STATUSES = {"stopped", "targeted", "expired", "skipped"}

# Statuses that block price checking but are resumable
PAUSED_STATUSES = {"suspended"}


# ---------------------------------------------------------------------------
# Signal seal (now covers flags — seal fix)
# ---------------------------------------------------------------------------

def seal_signal(signal) -> str:
    """
    SHA-256 hash of the signal's immutable fields.
    v3: also covers do_not_trade_flags and warning_flags so retroactive
    veto injection is detected.
    """
    def _val(v):
        return v.value if hasattr(v, "value") else str(v)

    payload = {
        "signal_id":         str(signal.signal_id),
        "instrument":        signal.instrument,
        "direction":         _val(signal.direction),
        "conviction":        round(float(signal.conviction), 4),
        "entry_low":         round(float(signal.entry_zone.low), 6),
        "entry_high":        round(float(signal.entry_zone.high), 6),
        "stop_loss":         round(float(signal.risk.stop_loss), 6),
        "target_1":          round(float(signal.risk.target_1), 6),
        "point_in_time_date": str(signal.point_in_time_date),
        # Seal now covers flags (veto injection would break this)
        "do_not_trade_flags": sorted(list(signal.do_not_trade_flags)),
        "warning_flags":      sorted(list(signal.warning_flags)),
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def verify_seal(signal, stored_seal: str) -> bool:
    return seal_signal(signal) == stored_seal


# ---------------------------------------------------------------------------
# Live signal record
# ---------------------------------------------------------------------------

@dataclass
class LiveSignalRecord:
    signal_id:           str
    signal_seal:         str
    instrument:          str
    asset_class:         str
    direction:           str
    conviction:          float
    timeframe:           str
    regime:              str

    entry_low:           float
    entry_high:          float
    stop_loss:           float
    target_1:            float
    target_2:            Optional[float]

    created_date:        date
    expiry_date:         date
    status:              str = "active"   # active|suspended|stopped|targeted|expired|skipped

    resolved_date:       Optional[date]  = None
    resolved_price:      Optional[float] = None
    r_multiple:          Optional[float] = None
    outcome:             Optional[str]   = None

    # Risk 5: suspension fields
    suspended_at:        Optional[str]   = None
    suspension_reason:   str             = ""
    resumed_at:          Optional[str]   = None

    # Attribution
    dominant_universe:   str   = ""
    contrarian_setup:    bool  = False
    universe_divergence: float = 0.0
    universe_sentiments: dict  = field(default_factory=dict)
    devil_strength:      float = 0.0

    notes:               str   = ""

    @property
    def is_resolved(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def is_suspended(self) -> bool:
        return self.status == "suspended"

    @property
    def days_live(self) -> int:
        end = self.resolved_date or date.today()
        return (end - self.created_date).days

    def check_price(self, current_price: float) -> Optional[str]:
        """Returns new terminal status string or None if still active."""
        if self.status != "active":
            return None  # suspended / already resolved — skip
        if date.today() > self.expiry_date:
            return "expired"
        d = self.direction.lower()
        if d == "long":
            if current_price <= self.stop_loss: return "stopped"
            if current_price >= self.target_1:  return "targeted"
        elif d == "short":
            if current_price >= self.stop_loss: return "stopped"
            if current_price <= self.target_1:  return "targeted"
        return None

    def resolve(self, new_status: str, price: float, notes: str = "") -> None:
        self.status         = new_status
        self.resolved_date  = date.today()
        self.resolved_price = price
        self.notes          = notes
        mid = (self.entry_low + self.entry_high) / 2
        if mid > 0 and self.stop_loss > 0:
            risk = abs(mid - self.stop_loss)
            if risk > 0:
                pnl = (price - mid) if self.direction == "long" else (mid - price)
                self.r_multiple = round(pnl / risk, 3)
                self.outcome = ("win" if self.r_multiple > 0.1 else
                                "loss" if self.r_multiple < -0.1 else "breakeven")

    def suspend(self, reason: str = "market_halt") -> None:
        """Risk 5: pause without resolving — resumes when market reopens."""
        if self.status == "active":
            self.status           = "suspended"
            self.suspended_at     = datetime.utcnow().isoformat()
            self.suspension_reason = reason
            logger.info("Signal %s SUSPENDED: %s", self.signal_id[:8], reason)

    def resume(self) -> None:
        """Risk 5: un-suspend when market reopens."""
        if self.status == "suspended":
            self.status     = "active"
            self.resumed_at = datetime.utcnow().isoformat()
            logger.info("Signal %s RESUMED", self.signal_id[:8])


# ---------------------------------------------------------------------------
# Forward monitor
# ---------------------------------------------------------------------------

class ForwardMonitor:

    def __init__(self, log_dir: Path | None = None):
        self.log_dir  = log_dir or Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self._db_path = self.log_dir / "live_signals.jsonl"
        self._records: dict[str, LiveSignalRecord] = {}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._db_path.exists():
            return
        seen: dict[str, LiveSignalRecord] = {}
        with open(self._db_path) as f:
            for line in f:
                try:
                    d   = json.loads(line.strip())
                    rec = self._dict_to_record(d)
                    seen[rec.signal_id] = rec  # last write wins
                except Exception as e:
                    logger.warning("Skipped corrupt record: %s", e)
        self._records = seen
        logger.info("Loaded %d signal records", len(self._records))

    def _save_record(self, rec: LiveSignalRecord) -> None:
        d = asdict(rec)
        for k, v in list(d.items()):
            if isinstance(v, date):
                d[k] = v.isoformat()
        with open(self._db_path, "a") as f:
            f.write(json.dumps(d) + "\n")

    def _dict_to_record(self, d: dict) -> LiveSignalRecord:
        for k in ("created_date", "expiry_date", "resolved_date"):
            if d.get(k):
                d[k] = date.fromisoformat(d[k])
        # Fill new fields that might be absent in old records
        d.setdefault("suspended_at", None)
        d.setdefault("suspension_reason", "")
        d.setdefault("resumed_at", None)
        d.setdefault("devil_strength", 0.0)
        return LiveSignalRecord(**d)

    # ── Registration ──────────────────────────────────────────────────────────

    def register_signal(self, signal, swarm_report=None) -> LiveSignalRecord:
        seal = seal_signal(signal)
        _v  = lambda attr, default: getattr(signal, attr).value if hasattr(getattr(signal, attr, None), "value") else str(getattr(signal, attr, default))
        direction = _v("direction", "neutral")
        timeframe = _v("timeframe", "swing")
        expiry_days = TIMEFRAME_EXPIRY_DAYS.get(timeframe, 10)
        expiry = signal.point_in_time_date + timedelta(days=expiry_days)

        dom_univ = getattr(swarm_report, "dominant_universe", "") if swarm_report else ""
        contrarian = getattr(swarm_report, "contrarian_setup", False) if swarm_report else False
        div = getattr(swarm_report, "universe_divergence", 0.0) if swarm_report else 0.0
        sentiments = {
            name: getattr(res, "aggregate_sentiment", 0.0)
            for name, res in getattr(swarm_report, "universe_results", {}).items()
        } if swarm_report else {}
        devil_str = 0.0
        if hasattr(signal, "devil_advocate"):
            devil_str = float(getattr(signal, "devil_advocate", {}).get("strength", 0.0) if isinstance(signal.devil_advocate, dict) else 0.0)

        rec = LiveSignalRecord(
            signal_id           = str(signal.signal_id),
            signal_seal         = seal,
            instrument          = signal.instrument,
            asset_class         = str(_v("asset_class", "equity")),
            direction           = direction,
            conviction          = float(signal.conviction),
            timeframe           = timeframe,
            regime              = str(_v("regime", "range_bound")),
            entry_low           = float(signal.entry_zone.low),
            entry_high          = float(signal.entry_zone.high),
            stop_loss           = float(signal.risk.stop_loss),
            target_1            = float(signal.risk.target_1),
            target_2            = float(signal.risk.target_2) if signal.risk.target_2 else None,
            created_date        = signal.point_in_time_date,
            expiry_date         = expiry,
            dominant_universe   = dom_univ,
            contrarian_setup    = bool(contrarian),
            universe_divergence = float(div),
            universe_sentiments = sentiments,
            devil_strength      = float(devil_str),
        )
        self._records[rec.signal_id] = rec
        self._save_record(rec)
        logger.info("Registered %s [%s %s conv=%.0f%% dev=%.2f]",
                    rec.signal_id[:8], direction, signal.instrument,
                    signal.conviction * 100, devil_str)
        return rec

    # ── Price polling ─────────────────────────────────────────────────────────

    def check_all(self, feed_kwargs: dict | None = None) -> list[LiveSignalRecord]:
        """
        Poll live prices for all ACTIVE (not suspended) signals.
        Suspended signals are skipped — they resume automatically
        when resume() is called (or via resume_by_instrument()).
        """
        active = [r for r in self._records.values() if r.is_active]
        if not active:
            return []

        from data.feeds import GlobalFeed
        feed = GlobalFeed(**(feed_kwargs or {}))
        prices: dict[str, float] = {}
        for sym in {r.instrument for r in active}:
            try:
                snap = feed.fetch_safe(sym)
                if snap.spot > 0 and not snap.is_stale:
                    prices[sym] = snap.spot
                elif snap.is_stale:
                    logger.warning("Stale price for %s — skipping resolution check", sym)
            except Exception as e:
                logger.warning("Price fetch failed for %s: %s", sym, e)

        resolved = []
        for rec in active:
            price = prices.get(rec.instrument)
            if price is None:
                continue
            new_status = rec.check_price(price)
            if new_status:
                rec.resolve(new_status, price)
                self._save_record(rec)
                resolved.append(rec)
                logger.info("Resolved %s: %s  price=%.4f  R=%.2f",
                            rec.signal_id[:8], new_status.upper(),
                            price, rec.r_multiple or 0.0)
        return resolved

    # ── Risk 5: Market halt / suspension ─────────────────────────────────────

    def suspend_by_instrument(self, instrument: str, reason: str = "market_halt") -> int:
        """
        Suspend all active signals for an instrument.
        Used when a circuit breaker or trading halt is detected.
        Returns number of signals suspended.
        """
        count = 0
        for rec in self._records.values():
            if rec.instrument == instrument and rec.is_active:
                rec.suspend(reason)
                self._save_record(rec)
                count += 1
        if count:
            logger.warning("Suspended %d signals for %s: %s", count, instrument, reason)
        return count

    def resume_by_instrument(self, instrument: str) -> int:
        """Resume all suspended signals for an instrument when trading restores."""
        count = 0
        for rec in self._records.values():
            if rec.instrument == instrument and rec.is_suspended:
                rec.resume()
                self._save_record(rec)
                count += 1
        if count:
            logger.info("Resumed %d signals for %s", count, instrument)
        return count

    def skip_signal(self, signal_id: str, notes: str = "") -> None:
        if signal_id in self._records:
            rec = self._records[signal_id]
            if not rec.is_resolved:
                rec.status = "skipped"
                rec.notes  = notes
                self._save_record(rec)

    # ── Queries ───────────────────────────────────────────────────────────────

    @property
    def active_signals(self) -> list[LiveSignalRecord]:
        return [r for r in self._records.values() if r.is_active]

    @property
    def suspended_signals(self) -> list[LiveSignalRecord]:
        return [r for r in self._records.values() if r.is_suspended]

    @property
    def resolved_signals(self) -> list[LiveSignalRecord]:
        return [r for r in self._records.values() if r.is_resolved]

    def get_outcomes_for_feedback(self) -> list[dict]:
        return [
            {
                "signal_id":           r.signal_id,
                "instrument":          r.instrument,
                "asset_class":         r.asset_class,
                "direction":           r.direction,
                "conviction":          r.conviction,
                "regime":              r.regime,
                "outcome":             r.outcome,
                "r_multiple":          r.r_multiple,
                "days_live":           r.days_live,
                "dominant_universe":   r.dominant_universe,
                "contrarian_setup":    r.contrarian_setup,
                "universe_divergence": r.universe_divergence,
                "universe_sentiments": r.universe_sentiments,
                "devil_strength":      r.devil_strength,
            }
            for r in self.resolved_signals
            if r.outcome is not None
        ]

    def summary(self) -> str:
        res   = self.resolved_signals
        wins  = sum(1 for r in res if r.outcome == "win")
        total = len(res)
        rmult = [r.r_multiple for r in res if r.r_multiple is not None]
        avg_r = sum(rmult) / len(rmult) if rmult else 0.0
        susp  = len(self.suspended_signals)
        return (
            f"Forward test: {total} resolved  |  "
            f"Win rate: {wins/total:.0%}  |  Avg R: {avg_r:+.2f}  |  "
            f"Active: {len(self.active_signals)}  |  Suspended: {susp}"
            if total else
            f"No resolved signals yet. Active: {len(self.active_signals)}  Suspended: {susp}"
        )
