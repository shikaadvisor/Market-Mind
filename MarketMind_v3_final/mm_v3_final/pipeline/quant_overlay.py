"""
pipeline/quant_overlay.py
==========================
Quant Model Overlay — tracks your existing model's signals alongside MarketMind.

Your existing daily quant model generates long/short/flat signals.
MarketMind generates signals independently.
This overlay lets you:

  1. LOG your model's direction at the time MarketMind fires a signal
  2. TRACK agreement rate between the two systems over time
  3. MEASURE: when they agree, does performance improve?
              when they disagree, which is right more often?

After 30-40 signals this gives you statistically meaningful data on:
  - Whether MarketMind adds alpha on top of your model (agreement filter)
  - Whether MarketMind is a useful contrarian indicator when it disagrees
  - Which instruments/regimes benefit most from the sentiment overlay

Usage:
    overlay = QuantModelOverlay(model_name="my_daily_model")
    # When MarketMind fires a signal:
    overlay.log_model_view(
        signal_id   = result.live_signal_record.signal_id,
        instrument  = "ZC",
        mm_direction = "long",
        quant_direction = "long",   # your model's current view
        quant_conviction = 0.7,     # optional: your model's conviction
        notes = "momentum breakout above 20dma"
    )
    # After signal resolves:
    print(overlay.agreement_report())
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("marketmind.quant_overlay")


@dataclass
class QuantView:
    signal_id:          str
    instrument:         str
    timestamp:          str           # ISO string

    # MarketMind signal
    mm_direction:       str           # "long" | "short" | "neutral"
    mm_conviction:      float
    mm_regime:          str
    mm_timeframe:       str
    mm_devil_strength:  float
    mm_universe_divergence: float

    # Your quant model's view
    quant_direction:    str           # "long" | "short" | "flat"
    quant_conviction:   Optional[float] = None
    notes:              str = ""

    # Filled when signal resolves
    outcome:            Optional[str]   = None  # "win" | "loss" | "breakeven"
    r_multiple:         Optional[float] = None
    resolved_date:      Optional[str]   = None

    @property
    def agreement(self) -> Optional[bool]:
        """True if both models agree on direction."""
        if not self.quant_direction:
            return None
        mm = self.mm_direction.lower()
        qm = self.quant_direction.lower()
        if mm == "neutral" or qm == "flat":
            return None
        return (mm == "long" and qm == "long") or (mm == "short" and qm == "short")

    @property
    def conflict(self) -> bool:
        """True if models point in opposite directions."""
        mm = self.mm_direction.lower()
        qm = self.quant_direction.lower()
        return (mm == "long" and qm == "short") or (mm == "short" and qm == "long")


class QuantModelOverlay:
    """
    Logs your quant model's view alongside each MarketMind signal
    and tracks the comparative performance.
    """

    def __init__(self, model_name: str, log_dir: Path | None = None):
        self.model_name = model_name
        self.log_dir    = log_dir or Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self._path      = self.log_dir / f"quant_overlay_{model_name}.jsonl"
        self._views:    dict[str, QuantView] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        with open(self._path) as f:
            for line in f:
                try:
                    d   = json.loads(line.strip())
                    v   = QuantView(**d)
                    self._views[v.signal_id] = v
                except Exception as e:
                    logger.warning("Corrupt overlay record: %s", e)

    def _save(self, v: QuantView) -> None:
        d = asdict(v)
        with open(self._path, "a") as f:
            f.write(json.dumps(d) + "\n")

    def log_model_view(
        self,
        signal_id:        str,
        instrument:       str,
        mm_direction:     str,
        quant_direction:  str,
        mm_conviction:    float = 0.0,
        mm_regime:        str   = "",
        mm_timeframe:     str   = "swing",
        mm_devil_strength: float = 0.0,
        mm_universe_divergence: float = 0.0,
        quant_conviction: Optional[float] = None,
        notes:            str   = "",
    ) -> QuantView:
        """
        Log your model's current view at the time MarketMind fires a signal.
        Call this immediately after pipeline.run() returns a live signal.
        """
        view = QuantView(
            signal_id           = signal_id,
            instrument          = instrument,
            timestamp           = datetime.utcnow().isoformat(),
            mm_direction        = mm_direction,
            mm_conviction       = mm_conviction,
            mm_regime           = mm_regime,
            mm_timeframe        = mm_timeframe,
            mm_devil_strength   = mm_devil_strength,
            mm_universe_divergence = mm_universe_divergence,
            quant_direction     = quant_direction,
            quant_conviction    = quant_conviction,
            notes               = notes,
        )
        self._views[signal_id] = view
        self._save(view)
        agree_str = "AGREE" if view.agreement else ("CONFLICT" if view.conflict else "neutral")
        logger.info("Overlay [%s] %s: MM=%s QM=%s → %s",
                    signal_id[:8], instrument, mm_direction, quant_direction, agree_str)
        return view

    def update_outcome(self, signal_id: str, outcome: str, r_multiple: float) -> None:
        """Call when a signal resolves to track comparative performance."""
        if signal_id in self._views:
            v = self._views[signal_id]
            v.outcome       = outcome
            v.r_multiple    = r_multiple
            v.resolved_date = date.today().isoformat()
            self._save(v)

    def agreement_report(self) -> str:
        """Summary statistics on agreement vs disagreement performance."""
        resolved = [v for v in self._views.values() if v.outcome is not None]
        if not resolved:
            return f"\n  {self.model_name} overlay: no resolved signals yet.\n"

        agree_wins  = [v for v in resolved if v.agreement is True  and v.outcome == "win"]
        agree_total = [v for v in resolved if v.agreement is True]
        conf_wins   = [v for v in resolved if v.conflict   and v.outcome == "win"]
        conf_total  = [v for v in resolved if v.conflict]

        def _wr(wins, total) -> str:
            if not total: return "n/a"
            return f"{len(wins)/len(total):.0%} ({len(wins)}/{len(total)})"

        def _avgr(views) -> str:
            rs = [v.r_multiple for v in views if v.r_multiple is not None]
            return f"{sum(rs)/len(rs):+.2f}" if rs else "n/a"

        lines = [
            f"\n  Quant overlay — {self.model_name}",
            f"  Resolved signals: {len(resolved)}",
            f"\n  When models AGREE:    win rate={_wr(agree_wins, agree_total)}  avg R={_avgr(agree_total)}",
            f"  When models CONFLICT: win rate={_wr(conf_wins, conf_total)}  avg R={_avgr(conf_total)}",
        ]

        # Per-instrument breakdown
        instruments = sorted({v.instrument for v in resolved})
        if instruments:
            lines.append(f"\n  Per instrument (MarketMind win rate):")
            for instr in instruments:
                instr_resolved = [v for v in resolved if v.instrument == instr]
                wins = sum(1 for v in instr_resolved if v.outcome == "win")
                lines.append(f"    {instr:<8} {wins}/{len(instr_resolved)} = {wins/len(instr_resolved):.0%}")

        # Regime breakdown
        lines.append(f"\n  Per regime:")
        regimes = sorted({v.mm_regime for v in resolved if v.mm_regime})
        for reg in regimes:
            reg_resolved = [v for v in resolved if v.mm_regime == reg]
            wins = sum(1 for v in reg_resolved if v.outcome == "win")
            lines.append(f"    {reg:<20} {wins}/{len(reg_resolved)} = {wins/len(reg_resolved):.0%}")

        # Devil's advocate insight
        devil_triggered = [v for v in resolved if v.mm_devil_strength >= 0.70]
        if devil_triggered:
            d_wins = sum(1 for v in devil_triggered if v.outcome == "win")
            lines.append(f"\n  When devil's advocate triggered (strength ≥0.70):")
            lines.append(f"    Win rate: {d_wins/len(devil_triggered):.0%} ({d_wins}/{len(devil_triggered)})")
            if d_wins / len(devil_triggered) < 0.40:
                lines.append(f"    → Devil is reliable: consider raising do_not_trade threshold")

        return "\n".join(lines) + "\n"

    @property
    def all_views(self) -> list[QuantView]:
        return list(self._views.values())

    def pending_log(self, instrument: str | None = None) -> list[QuantView]:
        """Active MarketMind signals where you haven't yet logged your model's view."""
        return [
            v for v in self._views.values()
            if not v.quant_direction
            and (instrument is None or v.instrument == instrument)
        ]
