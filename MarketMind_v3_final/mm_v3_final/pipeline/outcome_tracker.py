"""
pipeline/outcome_tracker.py
===========================
Session 7 — Outcome tracking and signal quality analytics.

After you review a signal and (optionally) trade it, you record what happened.
This file handles:
  1. OutcomeLogger   — records actual trade outcomes against signal predictions
  2. SignalAnalytics — computes win rates, R-multiple stats, calibration scores
  3. CalibrationReport — answers "are my high-conviction signals actually better?"

Design principle: the system is only as useful as your ability to improve it.
Without outcome tracking you are flying blind. With it, after 30–50 signals
you can tune weights, conviction thresholds, and agent prompts based on evidence.

Usage:
    # After a trade completes:
    tracker = OutcomeTracker()
    tracker.record_outcome(
        signal_id   = "abc-123",
        outcome     = "win",
        actual_entry = 74.20,
        actual_exit  = 76.80,
        notes        = "OPEC cut catalyst played out over 3 sessions"
    )

    # Get analytics:
    report = tracker.generate_report()
    print(report.summary())
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Outcome record
# ---------------------------------------------------------------------------

@dataclass
class OutcomeRecord:
    signal_id:       str
    recorded_at:     datetime
    outcome:         str          # "win" | "loss" | "breakeven" | "skipped"
    actual_entry:    Optional[float]
    actual_exit:     Optional[float]
    actual_pnl_pct:  Optional[float]  # % P&L on position (not portfolio)
    r_multiple:      Optional[float]  # actual_pnl / initial_risk
    notes:           str = ""

    # Signal metadata (filled from log)
    instrument:      str = ""
    asset_class:     str = ""
    direction:       str = ""
    conviction:      float = 0.0
    timeframe:       str = ""
    was_actionable:  bool = False
    do_not_trade:    str = ""
    crowd_sentiment: float = 0.0
    herd_index:      float = 0.0
    narrative:       str = ""


# ---------------------------------------------------------------------------
# Analytics report
# ---------------------------------------------------------------------------

@dataclass
class SignalQualityReport:
    """Full analytics on all tracked signals."""

    # Counts
    total_signals:       int = 0
    actionable_signals:  int = 0
    traded_signals:      int = 0
    skipped_signals:     int = 0

    # Win/loss (traded signals only)
    wins:                int = 0
    losses:              int = 0
    breakevens:          int = 0
    win_rate:            float = 0.0   # wins / (wins + losses)
    avg_r_multiple:      float = 0.0   # avg R:R realised
    expectancy:          float = 0.0   # win_rate * avg_win_r - (1-win_rate) * avg_loss_r

    # Conviction calibration
    # For each conviction bucket (0.5–0.6, 0.6–0.7, 0.7–0.8, 0.8–0.9, 0.9–1.0)
    # what was the actual win rate?
    conviction_calibration: dict = field(default_factory=dict)

    # Asset class breakdown
    by_asset_class:      dict = field(default_factory=dict)

    # Timeframe breakdown
    by_timeframe:        dict = field(default_factory=dict)

    # Crowd intelligence validation
    # Did contrarian crowd signals actually predict reversals?
    crowd_signal_accuracy: float = 0.0

    # Direction accuracy
    long_win_rate:       float = 0.0
    short_win_rate:      float = 0.0

    # Worst signal types (for tuning)
    worst_triggers:      list[str] = field(default_factory=list)
    best_triggers:       list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"\n{'═'*56}",
            f"  MarketMind Signal Quality Report",
            f"{'═'*56}",
            f"  Total signals:      {self.total_signals}",
            f"  Actionable:         {self.actionable_signals} ({self.actionable_signals/max(1,self.total_signals):.0%})",
            f"  Traded:             {self.traded_signals}",
            f"  Win rate:           {self.win_rate:.1%}  (W:{self.wins} L:{self.losses} BE:{self.breakevens})",
            f"  Avg R-multiple:     {self.avg_r_multiple:.2f}R",
            f"  Expectancy:         {self.expectancy:.3f}R per trade",
            f"",
            f"  Conviction calibration:",
        ]
        for bucket, stats in sorted(self.conviction_calibration.items()):
            n   = stats.get("n", 0)
            wr  = stats.get("win_rate", 0)
            bar = "█" * int(wr * 10) + "░" * (10 - int(wr * 10))
            lines.append(f"    {bucket}  [{bar}]  {wr:.0%}  (n={n})")

        if self.by_asset_class:
            lines.append(f"")
            lines.append(f"  By asset class:")
            for ac, stats in sorted(self.by_asset_class.items()):
                n  = stats.get("n", 0)
                wr = stats.get("win_rate", 0)
                lines.append(f"    {ac:<25}  {wr:.0%}  (n={n})")

        lines.append(f"")
        lines.append(f"  Long win rate:  {self.long_win_rate:.1%}")
        lines.append(f"  Short win rate: {self.short_win_rate:.1%}")
        lines.append(f"  Crowd signal accuracy: {self.crowd_signal_accuracy:.1%}")
        lines.append(f"{'═'*56}")
        return "\n".join(lines)

    def tuning_recommendations(self) -> list[str]:
        """
        Evidence-based recommendations for tuning the system.
        Only fires when there's enough data (n >= 10 per bucket).
        """
        recs = []

        if self.total_signals < 20:
            return ["Insufficient data — need at least 20 tracked signals before tuning."]

        # Conviction calibration
        for bucket, stats in self.conviction_calibration.items():
            n, wr = stats.get("n", 0), stats.get("win_rate", 0)
            if n < 5:
                continue
            lo, hi = (float(x) for x in bucket.split("–"))
            expected_wr = (lo + hi) / 2  # conviction should predict win rate
            if wr < expected_wr - 0.15 and n >= 8:
                recs.append(
                    f"Conviction bucket {bucket} is OVERCONFIDENT: "
                    f"predicted {expected_wr:.0%} win rate, got {wr:.0%}. "
                    f"Reduce coordinator conviction threshold by 0.05."
                )
            elif wr > expected_wr + 0.15 and n >= 8:
                recs.append(
                    f"Conviction bucket {bucket} is UNDERCONFIDENT: "
                    f"got {wr:.0%} win rate at {expected_wr:.0%} conviction. "
                    f"These signals are better than the model thinks."
                )

        # Direction bias
        if abs(self.long_win_rate - self.short_win_rate) > 0.20 and self.traded_signals >= 15:
            better = "long" if self.long_win_rate > self.short_win_rate else "short"
            worse  = "short" if better == "long" else "long"
            recs.append(
                f"Strong {better} bias: {better} win rate {max(self.long_win_rate, self.short_win_rate):.0%} "
                f"vs {worse} {min(self.long_win_rate, self.short_win_rate):.0%}. "
                f"Consider raising conviction threshold for {worse} signals."
            )

        # Crowd intelligence
        if self.crowd_signal_accuracy < 0.45 and self.traded_signals >= 15:
            recs.append(
                f"Crowd intelligence is underperforming ({self.crowd_signal_accuracy:.0%} accuracy). "
                f"Reduce WEIGHT_CROWD_SIM from 0.25 to 0.15 in config.py."
            )
        elif self.crowd_signal_accuracy > 0.65 and self.traded_signals >= 15:
            recs.append(
                f"Crowd intelligence is strong ({self.crowd_signal_accuracy:.0%} accuracy). "
                f"Consider raising WEIGHT_CROWD_SIM from 0.25 to 0.30 in config.py."
            )

        # Asset class
        for ac, stats in self.by_asset_class.items():
            n, wr = stats.get("n", 0), stats.get("win_rate", 0)
            if n >= 8 and wr < 0.35:
                recs.append(
                    f"Poor performance on {ac} ({wr:.0%} win rate, n={n}). "
                    f"Review persona prompts for this asset class — "
                    f"the simulation agents may not be calibrated for it."
                )

        if not recs:
            recs.append("No significant tuning recommendations — system is well-calibrated.")

        return recs


# ---------------------------------------------------------------------------
# Outcome tracker
# ---------------------------------------------------------------------------

class OutcomeTracker:
    """
    Records trade outcomes and generates analytics.

    Stores outcomes in:
      - outcomes.csv    (flat, one row per outcome, easy to open in Excel)
      - outcomes.jsonl  (full fidelity, includes all signal metadata)
    """

    def __init__(self, log_dir: Path | None = None):
        self.log_dir = log_dir or Path(__file__).parent.parent / "logs"
        self.log_dir.mkdir(exist_ok=True)
        self.outcomes_csv   = self.log_dir / "outcomes.csv"
        self.outcomes_jsonl = self.log_dir / "outcomes.jsonl"
        self.signals_csv    = self.log_dir / "signals.csv"

    # -----------------------------------------------------------------------
    # Record an outcome
    # -----------------------------------------------------------------------

    def record_outcome(
        self,
        signal_id:    str,
        outcome:      str,                    # "win" | "loss" | "breakeven" | "skipped"
        actual_entry: Optional[float] = None,
        actual_exit:  Optional[float] = None,
        notes:        str = "",
    ) -> OutcomeRecord:
        """
        Record what happened with a signal.
        Looks up the original signal in signals.csv to enrich the record.
        """
        assert outcome in ("win", "loss", "breakeven", "skipped"), \
            f"outcome must be win/loss/breakeven/skipped, got '{outcome}'"

        # Look up original signal metadata
        signal_meta = self._lookup_signal(signal_id)

        # Compute P&L and R-multiple
        actual_pnl_pct = None
        r_multiple     = None

        if actual_entry and actual_exit and actual_entry > 0:
            direction = signal_meta.get("direction", "long")
            if direction == "long":
                actual_pnl_pct = (actual_exit - actual_entry) / actual_entry * 100
            else:
                actual_pnl_pct = (actual_entry - actual_exit) / actual_entry * 100

        if actual_pnl_pct is not None:
            stop      = float(signal_meta.get("stop_loss",   0) or 0)
            entry_mid = float(signal_meta.get("entry_high",  actual_entry or 0) or 0)
            if stop > 0 and entry_mid > 0:
                risk_pct = abs(entry_mid - stop) / entry_mid * 100
                if risk_pct > 0:
                    r_multiple = round(actual_pnl_pct / risk_pct, 2)

        record = OutcomeRecord(
            signal_id       = signal_id,
            recorded_at     = datetime.utcnow(),
            outcome         = outcome,
            actual_entry    = actual_entry,
            actual_exit     = actual_exit,
            actual_pnl_pct  = round(actual_pnl_pct, 3) if actual_pnl_pct else None,
            r_multiple      = r_multiple,
            notes           = notes,
            instrument      = signal_meta.get("instrument", ""),
            asset_class     = signal_meta.get("asset_class", ""),
            direction       = signal_meta.get("direction", ""),
            conviction      = float(signal_meta.get("conviction", 0) or 0),
            timeframe       = signal_meta.get("timeframe", ""),
            was_actionable  = signal_meta.get("is_actionable", "False") == "True",
            do_not_trade    = signal_meta.get("do_not_trade_flags", ""),
            crowd_sentiment = float(signal_meta.get("crowd_sentiment", 0) or 0),
            herd_index      = float(signal_meta.get("herd_index", 0) or 0),
        )

        self._write_outcome(record)
        print(f"  ✓ Outcome recorded: {signal_id[:16]}... → {outcome.upper()}"
              + (f" ({actual_pnl_pct:+.2f}%)" if actual_pnl_pct else "")
              + (f" [{r_multiple:+.2f}R]" if r_multiple else ""))
        return record

    def _lookup_signal(self, signal_id: str) -> dict:
        """Look up signal metadata from signals.csv."""
        if not self.signals_csv.exists():
            return {}
        try:
            with open(self.signals_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("signal_id", "").startswith(signal_id[:16]):
                        return row
        except Exception:
            pass
        return {}

    def _write_outcome(self, record: OutcomeRecord) -> None:
        """Write outcome to CSV and JSONL."""
        row = {
            "signal_id":      record.signal_id,
            "recorded_at":    record.recorded_at.isoformat(),
            "outcome":        record.outcome,
            "actual_entry":   record.actual_entry,
            "actual_exit":    record.actual_exit,
            "actual_pnl_pct": record.actual_pnl_pct,
            "r_multiple":     record.r_multiple,
            "instrument":     record.instrument,
            "asset_class":    record.asset_class,
            "direction":      record.direction,
            "conviction":     record.conviction,
            "timeframe":      record.timeframe,
            "was_actionable": record.was_actionable,
            "crowd_sentiment": record.crowd_sentiment,
            "herd_index":     record.herd_index,
            "notes":          record.notes,
        }

        write_header = not self.outcomes_csv.exists()
        with open(self.outcomes_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        with open(self.outcomes_jsonl, "a") as f:
            f.write(json.dumps(row) + "\n")

    # -----------------------------------------------------------------------
    # Analytics
    # -----------------------------------------------------------------------

    def generate_report(self) -> SignalQualityReport:
        """Compute full analytics from all recorded outcomes."""
        records = self._load_outcomes()

        if not records:
            print("  No outcomes recorded yet.")
            return SignalQualityReport()

        report = SignalQualityReport()
        report.total_signals = len(records)

        # Load original signals for totals
        all_signals = self._load_signals()
        report.actionable_signals = sum(
            1 for s in all_signals if s.get("is_actionable") == "True"
        )

        traded  = [r for r in records if r["outcome"] in ("win", "loss", "breakeven")]
        skipped = [r for r in records if r["outcome"] == "skipped"]
        wins    = [r for r in traded if r["outcome"] == "win"]
        losses  = [r for r in traded if r["outcome"] == "loss"]
        bes     = [r for r in traded if r["outcome"] == "breakeven"]

        report.traded_signals = len(traded)
        report.skipped_signals = len(skipped)
        report.wins      = len(wins)
        report.losses    = len(losses)
        report.breakevens = len(bes)
        report.win_rate  = len(wins) / max(1, len(wins) + len(losses))

        # R-multiples
        r_mults = []
        for r in traded:
            v = r.get("r_multiple")
            if v is not None and v not in ("", "None"):
                try:
                    r_mults.append(float(v))
                except (ValueError, TypeError):
                    pass
        if r_mults:
            report.avg_r_multiple = round(sum(r_mults) / len(r_mults), 3)
            win_r  = [r for r in r_mults if r > 0]
            loss_r = [abs(r) for r in r_mults if r < 0]
            avg_win_r  = sum(win_r)  / max(1, len(win_r))
            avg_loss_r = sum(loss_r) / max(1, len(loss_r))
            report.expectancy = round(
                report.win_rate * avg_win_r - (1 - report.win_rate) * avg_loss_r, 3
            )

        # Conviction calibration
        buckets = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
        for lo, hi in buckets:
            bucket_trades = [
                r for r in traded
                if lo <= (r.get("conviction") or 0) < hi
            ]
            if bucket_trades:
                bw = sum(1 for r in bucket_trades if r["outcome"] == "win")
                report.conviction_calibration[f"{lo:.1f}–{min(hi, 1.0):.1f}"] = {
                    "n":        len(bucket_trades),
                    "wins":     bw,
                    "win_rate": bw / len(bucket_trades),
                }

        # Asset class breakdown
        asset_classes = set(r.get("asset_class", "unknown") for r in traded)
        for ac in asset_classes:
            ac_trades = [r for r in traded if r.get("asset_class") == ac]
            ac_wins   = sum(1 for r in ac_trades if r["outcome"] == "win")
            report.by_asset_class[ac] = {
                "n":        len(ac_trades),
                "wins":     ac_wins,
                "win_rate": ac_wins / max(1, len(ac_trades)),
            }

        # Timeframe breakdown
        timeframes = set(r.get("timeframe", "unknown") for r in traded)
        for tf in timeframes:
            tf_trades = [r for r in traded if r.get("timeframe") == tf]
            tf_wins   = sum(1 for r in tf_trades if r["outcome"] == "win")
            report.by_timeframe[tf] = {
                "n":        len(tf_trades),
                "wins":     tf_wins,
                "win_rate": tf_wins / max(1, len(tf_trades)),
            }

        # Direction breakdown
        longs  = [r for r in traded if r.get("direction") == "long"]
        shorts = [r for r in traded if r.get("direction") == "short"]
        report.long_win_rate  = sum(1 for r in longs  if r["outcome"] == "win") / max(1, len(longs))
        report.short_win_rate = sum(1 for r in shorts if r["outcome"] == "win") / max(1, len(shorts))

        # Crowd signal accuracy
        # Contrarian crowd signal = crowd bearish while institutional bullish (or vice versa)
        crowd_contrarian = [r for r in traded if abs(r.get("crowd_sentiment") or 0) > 0.3]
        if crowd_contrarian:
            # Contrarian = if crowd is bearish but signal is long (or vice versa)
            crowd_correct = sum(
                1 for r in crowd_contrarian
                if (r.get("crowd_sentiment", 0) < -0.2 and r.get("direction") == "long"
                    and r["outcome"] == "win") or
                   (r.get("crowd_sentiment", 0) > 0.2  and r.get("direction") == "short"
                    and r["outcome"] == "win")
            )
            report.crowd_signal_accuracy = crowd_correct / max(1, len(crowd_contrarian))

        return report

    def _load_outcomes(self) -> list[dict]:
        if not self.outcomes_csv.exists():
            return []
        try:
            with open(self.outcomes_csv) as f:
                rows = list(csv.DictReader(f))
            # Convert numeric fields
            for row in rows:
                for field in ("conviction", "actual_pnl_pct", "r_multiple",
                              "crowd_sentiment", "herd_index"):
                    if row.get(field) and row[field] not in ("", "None"):
                        try:
                            row[field] = float(row[field])
                        except (ValueError, TypeError):
                            row[field] = None
            return rows
        except Exception:
            return []

    def _load_signals(self) -> list[dict]:
        if not self.signals_csv.exists():
            return []
        try:
            with open(self.signals_csv) as f:
                return list(csv.DictReader(f))
        except Exception:
            return []

    # -----------------------------------------------------------------------
    # CLI view
    # -----------------------------------------------------------------------

    def print_recent(self, n: int = 15) -> None:
        records = self._load_outcomes()[-n:]
        if not records:
            print("  No outcomes recorded yet.")
            return

        print(f"\n  Recent Outcomes (last {len(records)})")
        print(f"  {'─'*80}")
        print(f"  {'Signal':<18} {'Instr':<12} {'Dir':<6} {'Conv':>5} "
              f"{'Outcome':<10} {'PnL%':>7} {'R':>6}  Notes")
        print(f"  {'─'*80}")

        for r in records:
            outcome = r.get("outcome", "?")
            colour  = {"win": "✓", "loss": "✗", "breakeven": "=", "skipped": "–"}.get(outcome, "?")
            pnl     = r.get("actual_pnl_pct")
            pnl_str = f"{float(pnl):>+6.2f}%" if pnl is not None and pnl != "" else "     —"
            r_str   = r.get("r_multiple")
            r_disp  = f"{float(r_str):>+5.2f}R" if r_str is not None and r_str != "" else "     —"
            notes   = (r.get("notes") or "")[:25]

            print(f"  {r.get('signal_id','')[:16]:<18} "
                  f"{r.get('instrument',''):<12} "
                  f"{r.get('direction',''):<6} "
                  f"{float(r.get('conviction') or 0):>4.0%} "
                  f"{colour} {outcome:<8} "
                  f"{pnl_str}  {r_disp}  {notes}")

        print(f"  {'─'*80}")


# ---------------------------------------------------------------------------
# Standalone usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uuid

    print("\n=== OutcomeTracker Demo ===\n")
    tracker = OutcomeTracker()

    # Simulate recording some outcomes
    demo_outcomes = [
        ("demo-001", "win",       74.20, 76.80, "OPEC cut catalyst played out"),
        ("demo-002", "loss",      2890.0, 2855.0, "Gold reversed on strong USD"),
        ("demo-003", "win",       1.0820, 1.0940, "ECB dovish surprise"),
        ("demo-004", "skipped",   None,   None,   "Missed — was on holiday"),
        ("demo-005", "win",       542.0,  575.0,  "WASDE bearish — short worked"),
        ("demo-006", "breakeven", 5280.0, 5282.0, "Flat exit — conviction too low"),
        ("demo-007", "win",       24200.0, 24800.0, "RBI hold rally"),
        ("demo-008", "loss",      71.50, 69.20, "Inventory build surprised"),
    ]

    for i, (sid, outcome, entry, exit_, notes) in enumerate(demo_outcomes):
        tracker.record_outcome(
            signal_id    = f"demo-signal-{i:04d}",
            outcome      = outcome,
            actual_entry = entry,
            actual_exit  = exit_,
            notes        = notes,
        )

    print()
    tracker.print_recent()

    print("\n  Generating analytics report...")
    report = tracker.generate_report()
    print(report.summary())

    print("\n  Tuning recommendations:")
    for rec in report.tuning_recommendations():
        print(f"  • {rec}")
