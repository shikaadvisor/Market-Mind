"""
pipeline/dashboard.py
=====================
Rich terminal dashboard for reviewing MarketMind signals.

Displays a full signal in structured, colour-coded terminal output.
Also shows the simulation brief, agent contributions, and actionability status.

Uses only Python stdlib (no rich/blessed required) — falls back to plain
text if the terminal doesn't support ANSI. If rich IS installed, output
is beautiful. Either way it works.

Usage:
    from pipeline.dashboard import display_signal, display_pipeline_result
    display_pipeline_result(result)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# ANSI colour codes — degrade gracefully if not supported
# ---------------------------------------------------------------------------

_USE_COLOUR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    if not _USE_COLOUR:
        return text
    codes = {
        "green":   "\033[92m", "red":     "\033[91m", "yellow":  "\033[93m",
        "blue":    "\033[94m", "cyan":    "\033[96m", "white":   "\033[97m",
        "bold":    "\033[1m",  "dim":     "\033[2m",  "reset":   "\033[0m",
        "bg_green": "\033[42m\033[30m", "bg_red": "\033[41m\033[97m",
        "bg_blue":  "\033[44m\033[97m", "bg_yellow": "\033[43m\033[30m",
    }
    return f"{codes.get(code, '')}{text}{codes['reset']}"


def _box_top(width: int = 62) -> str:
    return "╔" + "═" * width + "╗"

def _box_bot(width: int = 62) -> str:
    return "╚" + "═" * width + "╝"

def _box_mid(width: int = 62) -> str:
    return "╠" + "═" * width + "╣"

def _box_row(content: str, width: int = 62) -> str:
    content = content[:width]
    return "║ " + content + " " * (width - len(content) - 1) + "║"

def _sep(width: int = 62) -> str:
    return "╟" + "─" * width + "╢"


# ---------------------------------------------------------------------------
# Conviction bar
# ---------------------------------------------------------------------------

def _conviction_bar(conviction: float, width: int = 20) -> str:
    filled  = int(conviction * width)
    empty   = width - filled
    bar     = "█" * filled + "░" * empty

    if conviction >= 0.85:
        colour = "green"
    elif conviction >= 0.70:
        colour = "cyan"
    elif conviction >= 0.50:
        colour = "yellow"
    else:
        colour = "red"

    return _c(colour, bar) + f" {conviction:.0%}"


# ---------------------------------------------------------------------------
# Sentiment bar
# ---------------------------------------------------------------------------

def _sentiment_bar(sentiment: float, width: int = 20) -> str:
    """Visual bar for -1 to +1 sentiment."""
    mid    = width // 2
    val    = int(abs(sentiment) * mid)
    val    = min(val, mid)

    if sentiment >= 0:
        bar = "─" * mid + "▶" * val + " " * (mid - val)
        colour = "green"
    else:
        bar = " " * (mid - val) + "◀" * val + "─" * mid
        colour = "red"

    label = f"{sentiment:+.2f}"
    return _c(colour, bar) + " " + label


# ---------------------------------------------------------------------------
# Main display functions
# ---------------------------------------------------------------------------

def display_pipeline_result(result, show_brief: bool = True) -> None:
    """Display the full pipeline result in the terminal."""
    if result.error:
        print(_c("red", f"\n  ✗ PIPELINE ERROR: {result.error}"))
        return

    display_signal(result.signal, result.simulation_report, show_brief=show_brief)
    _display_timing(result)


def display_signal(signal, sim_report: dict | None = None, show_brief: bool = True) -> None:
    """Display a TradingSignal in the terminal dashboard."""

    W = 62

    # ── Header ───────────────────────────────────────────────────────────────
    print()
    print(_c("bold", _box_top(W)))

    # Title
    is_act = signal.is_actionable
    status_label = _c("bg_green", " ✓ ACTIONABLE ") if is_act else _c("bg_yellow", " ⚠ REVIEW    ")
    title = f"  MarketMind Signal  {status_label}"
    print(_c("bold", _box_row(title, W)))
    print(_c("bold", _box_row(f"  ID: {signal.signal_id[:32]}...", W)))
    print(_box_mid(W))

    # ── Instrument + Core ────────────────────────────────────────────────────
    direction_colour = "green" if signal.direction.value == "long" else \
                       "red"   if signal.direction.value == "short" else "yellow"

    dir_str = _c(direction_colour, _c("bold", signal.direction.value.upper()))
    print(_box_row(f"  {signal.instrument:<12} {signal.exchange:<6} {signal.asset_class.value}", W))
    print(_box_row(f"  Direction:    {dir_str}", W))
    print(_box_row(f"  Conviction:   {_conviction_bar(signal.conviction)}", W))
    print(_box_row(f"  Timeframe:    {signal.timeframe.value}    Regime: {signal.regime.value}", W))
    print(_box_row(f"  Date:         {signal.point_in_time_date}", W))

    # ── Price Levels ──────────────────────────────────────────────────────────
    print(_sep(W))
    print(_box_row("  PRICE LEVELS", W))
    print(_sep(W))
    print(_box_row(f"  Entry zone:     {signal.entry_zone.low:>10,.2f} – {signal.entry_zone.high:>10,.2f}", W))
    print(_box_row(f"  Stop loss:      {signal.risk.stop_loss:>10,.2f}", W))
    print(_box_row(f"  Target 1:       {signal.risk.target_1:>10,.2f}", W))
    if signal.risk.target_2:
        print(_box_row(f"  Target 2:       {signal.risk.target_2:>10,.2f}", W))
    print(_box_row(f"  R:R ratio:      {signal.risk.risk_reward:>10.2f}x", W))
    print(_box_row(f"  Position size:  {signal.risk.max_position_pct:>10.0%}", W))
    print(_box_row(f"  Invalidation:   {signal.risk.invalidation_level:>10,.2f}", W))

    # ── Crowd Intelligence ────────────────────────────────────────────────────
    print(_sep(W))
    print(_box_row("  CROWD INTELLIGENCE (Digital World)", W))
    print(_sep(W))
    print(_box_row(f"  Aggregate:      {_sentiment_bar(signal.crowd.aggregate_sentiment, 18)}", W))
    print(_box_row(f"  Retail:         {_sentiment_bar(signal.crowd.retail_sentiment,    18)}", W))
    print(_box_row(f"  Institutional:  {_sentiment_bar(signal.crowd.institutional_sentiment, 18)}", W))
    print(_box_row(f"  Herd index:     {'█' * int(signal.crowd.herd_index * 14) + '░' * (14 - int(signal.crowd.herd_index * 14))} {signal.crowd.herd_index:.0%}", W))
    print(_box_row(f"  Panic prob:     {signal.crowd.panic_probability:.0%}", W))
    print(_box_row(f"  Contrarian:     {signal.crowd.contrarian_pressure:.0%}", W))
    print(_box_row(f"  Agents:         {signal.crowd.agent_count:,}", W))

    # ── Flags ─────────────────────────────────────────────────────────────────
    if signal.do_not_trade_flags or signal.warning_flags or signal.confidence_flags:
        print(_sep(W))
        print(_box_row("  FLAGS", W))
        print(_sep(W))

        for f in signal.do_not_trade_flags:
            print(_box_row(_c("red", f"  ✗ VETO: {f}"), W))

        for f in signal.warning_flags:
            print(_box_row(_c("yellow", f"  ⚠ WARN: {f}"), W))

        for f in signal.confidence_flags:
            print(_box_row(_c("green", f"  ✓ CONF: {f}"), W))

    # ── Agent Contributions ───────────────────────────────────────────────────
    print(_sep(W))
    print(_box_row("  AGENT CONTRIBUTIONS", W))
    print(_sep(W))

    for ac in signal.agent_contributions:
        dir_c = "green" if ac.view in ("long", "bullish") else \
                "red"   if ac.view in ("short", "bearish") else "yellow"
        bar = _conviction_bar(ac.conviction, 10)
        role_str = f"{ac.agent_role:<25}"
        view_str = _c(dir_c, f"{ac.view:<8}")
        print(_box_row(f"  {role_str} {view_str} {bar}", W))
        if ac.dissent:
            print(_box_row(_c("dim", f"    ↳ dissent: {ac.dissent[:40]}"), W))

    # ── Reasoning ─────────────────────────────────────────────────────────────
    print(_sep(W))
    print(_box_row("  COORDINATOR THESIS", W))
    print(_sep(W))

    thesis = signal.coordinator_thesis
    # Word-wrap at W-4 chars
    words = thesis.split()
    line  = "  "
    for word in words:
        if len(line) + len(word) + 1 > W - 2:
            print(_box_row(line, W))
            line = "  " + word + " "
        else:
            line += word + " "
    if line.strip():
        print(_box_row(line, W))

    # ── Footer ────────────────────────────────────────────────────────────────
    print(_box_mid(W))

    actionable_str = (
        _c("green", "✓ ACTIONABLE — review entry zone and execute with defined risk")
        if is_act else
        _c("yellow", "⚠ NOT ACTIONABLE — log only, do not trade")
        if not signal.do_not_trade_flags else
        _c("red", "✗ BLOCKED — risk manager veto in effect")
    )
    print(_box_row(f"  {actionable_str}", W))
    print(_c("bold", _box_bot(W)))
    print()


def _display_timing(result) -> None:
    print(_c("dim",
        f"  Timing: sim={result.elapsed_sim_sec:.1f}s  "
        f"desk={result.elapsed_desk_sec:.1f}s  "
        f"total={result.elapsed_total_sec:.1f}s\n"
    ))


# ---------------------------------------------------------------------------
# Signal history viewer
# ---------------------------------------------------------------------------

def display_signal_log(log_dir: Path | None = None, n: int = 10) -> None:
    """Print the last N signals from the CSV log."""
    import csv

    log_dir  = log_dir or Path(__file__).parent.parent / "logs"
    csv_path = log_dir / "signals.csv"

    if not csv_path.exists():
        print(_c("yellow", "  No signals logged yet."))
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(_c("yellow", "  Signal log is empty."))
        return

    recent = rows[-n:]
    W = 90
    print()
    print(_c("bold", f"  Signal Log — last {len(recent)} signals"))
    print("  " + "─" * W)

    header = (
        f"  {'Date':<12} {'Instrument':<12} {'Dir':<8} {'Conv':>6}  "
        f"{'R:R':>5}  {'Actionable':<12} {'Outcome':<12} {'Vetoes'}"
    )
    print(_c("bold", header))
    print("  " + "─" * W)

    for row in recent:
        conv   = float(row.get("conviction", 0))
        is_act = row.get("is_actionable", "False") == "True"
        vetoes = row.get("do_not_trade_flags", "")
        rr     = float(row.get("risk_reward", 0))

        dir_val = row.get("direction", "?")
        dir_c   = "green" if dir_val == "long" else "red" if dir_val == "short" else "yellow"

        act_str = _c("green", "YES") if is_act else _c("yellow", "NO")
        outcome = row.get("outcome", "pending")
        out_c   = "green" if outcome == "win" else "red" if outcome == "loss" else "dim"

        line = (
            f"  {row.get('created_at', '')[:10]:<12} "
            f"{row.get('instrument', ''):<12} "
            f"{_c(dir_c, dir_val):<8} "
            f"{conv:>6.0%}  "
            f"{rr:>5.1f}  "
            f"{act_str:<12} "
            f"{_c(out_c, outcome):<12} "
            f"{_c('dim', vetoes[:25]) if vetoes else _c('green', 'none')}"
        )
        print(line)

    print("  " + "─" * W)
    print(_c("dim", f"  Full log: {csv_path}\n"))


def display_outcome_prompt(signal) -> None:
    """
    Display a prompt asking the user to record the outcome of a signal.
    Called when reviewing a past signal.
    """
    if not signal:
        return
    print("\n  ── Record Outcome ────────────────────────────────────────")
    print(f"  Signal: {signal.signal_id[:32]}...")
    print(f"  {signal.signal_summary}")
    print("\n  To record outcome, run:")
    print(f"    python main.py --record-outcome {signal.signal_id[:16]}")
    print(f"  or use the OutcomeTracker API directly in Python.")
    print()
