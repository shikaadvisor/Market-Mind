"""
main_v3.py — MarketMind v3  (full workflow edition)
=====================================================
START TODAY (run in order):

  python main_v3.py --validate-config              # check all fixes applied
  python main_v3.py --sources                      # show which data keys are set
  python main_v3.py --cost-estimate                # understand the cost structure
  python main_v3.py --mock --workflow ags_daily    # validate ags pipeline (no keys needed)
  python main_v3.py --mock --workflow ags_intraday # validate lite mode
  python main_v3.py --live --workflow ags_daily    # first real run across all 5 ags
  python main_v3.py --watch-intraday ZC            # 30min session-aware intraday loop

QUANT MODEL OVERLAY (compare with your existing model):
  python main_v3.py --log-quant SIG_ID ZC long 0.7 "above 20dma"
  python main_v3.py --overlay-report

FORWARD TEST MANAGEMENT:
  python main_v3.py --signal-status
  python main_v3.py --check-signals
  python main_v3.py --suspend ZC                  # market halt
  python main_v3.py --resume ZC
  python main_v3.py --skip SIG_ID "reason"
  python main_v3.py --run-feedback
  python main_v3.py --feedback-report

COST / DIAGNOSTICS:
  python main_v3.py --cost-today
  python main_v3.py --cost-estimate
  python main_v3.py --regression                  # verify model pinning (needs real keys)
  python main_v3.py --workflows                   # list all workflow configs
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from workflows import ALL_WORKFLOWS, get_workflow, print_workflow_summary, WorkflowConfig


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="MarketMind v3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mock",       action="store_true", help="Mock LLM — no API keys needed")
    p.add_argument("--quiet",      action="store_true")
    p.add_argument("--force-run",  action="store_true", help="Run pipeline even without event trigger")

    # Workflow / instrument selection
    p.add_argument("--workflow",   choices=list(ALL_WORKFLOWS), help="Named workflow config")
    p.add_argument("--live",       action="store_true", help="Single live run")
    p.add_argument("--watch",      action="store_true", help="Continuous workflow watch")
    p.add_argument("--watch-intraday", metavar="SYM",  help="30min session-aware intraday loop")
    p.add_argument("--instrument", default="NIFTY50",  help="Symbol for single --live run")
    p.add_argument("--sensitivity",default="medium",   choices=["high","medium","low"])

    # Diagnostics
    p.add_argument("--validate-config",  action="store_true")
    p.add_argument("--sources",          action="store_true", help="Show data source status")
    p.add_argument("--cost-estimate",    action="store_true")
    p.add_argument("--cost-today",       action="store_true")
    p.add_argument("--regression",       action="store_true")
    p.add_argument("--workflows",        action="store_true", help="List workflow configs")

    # Forward test management
    p.add_argument("--signal-status",   action="store_true")
    p.add_argument("--check-signals",   action="store_true")
    p.add_argument("--suspend",         metavar="SYM")
    p.add_argument("--resume",          metavar="SYM")
    p.add_argument("--skip",            nargs="+")
    p.add_argument("--run-feedback",    action="store_true")
    p.add_argument("--feedback-report", action="store_true")

    # Quant model overlay
    p.add_argument("--log-quant", nargs="*", metavar="ARG",
                   help="--log-quant SIG_ID SYM DIRECTION [CONVICTION] [NOTES...]")
    p.add_argument("--overlay-report", action="store_true")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Key loading
# ---------------------------------------------------------------------------

def _load_keys(mock: bool) -> dict[str, str]:
    if mock:
        return {k: "mock" for k in [
            "OPENAI_API_KEY","ANTHROPIC_API_KEY","FRED_API_KEY",
            "FINNHUB_API_KEY","ALPHA_VANTAGE_API_KEY","EIA_API_KEY","NEWSAPI_KEY",
        ]}
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    keys = {k: os.environ.get(k, "") for k in [
        "OPENAI_API_KEY","ANTHROPIC_API_KEY","FRED_API_KEY",
        "FINNHUB_API_KEY","ALPHA_VANTAGE_API_KEY","EIA_API_KEY","NEWSAPI_KEY",
    ]}
    if not keys["OPENAI_API_KEY"] or not keys["ANTHROPIC_API_KEY"]:
        print("\n  ✗ OPENAI_API_KEY and ANTHROPIC_API_KEY are required.")
        print("  Set them in .env or use --mock for testing.\n")
        sys.exit(1)
    return keys


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def _make_pipeline(keys: dict, wf: WorkflowConfig | None, mock: bool):
    from pipeline.runner_v3 import MarketMindV3, PipelineConfigV3
    from config import cfg
    return MarketMindV3(PipelineConfigV3(
        openai_api_key     = keys["OPENAI_API_KEY"],
        anthropic_api_key  = keys["ANTHROPIC_API_KEY"],
        mock               = mock,
        daily_cost_cap_usd = wf.daily_cost_cap_usd if wf else cfg.DAILY_COST_CAP_USD,
        min_conviction_log = wf.min_conviction if wf else 0.30,
        verbose            = True,
        n_tier2_per_universe = wf.n_tier2_per_universe if wf else 50,
    ))


def _make_lite_pipeline(keys: dict, wf: WorkflowConfig):
    """Pipeline wired with LiteSwarmRunner (T3-only) for intraday polling."""
    from pipeline.runner_v3 import MarketMindV3, PipelineConfigV3
    from simulation.lite_runner import LiteSwarmRunner

    pipeline = MarketMindV3(PipelineConfigV3(
        openai_api_key     = keys["OPENAI_API_KEY"],
        anthropic_api_key  = keys["ANTHROPIC_API_KEY"],
        mock               = False,
        daily_cost_cap_usd = wf.daily_cost_cap_usd,
        min_conviction_log = wf.min_conviction,
        verbose            = True,
    ))
    pipeline._swarm = LiteSwarmRunner(n_agents=2_500, n_steps=30, verbose=False)
    return pipeline


# ---------------------------------------------------------------------------
# Session awareness
# ---------------------------------------------------------------------------

def _is_session_open(symbol: str) -> bool:
    try:
        from data.universe import is_instrument_tradeable
        return is_instrument_tradeable(symbol)
    except Exception:
        return True  # assume open if lookup fails


# ---------------------------------------------------------------------------
# Shared live fetch helper
# ---------------------------------------------------------------------------

def _fetch_and_detect(sym: str, sensitivity: str, feed_kwargs: dict):
    """Fetch data, detect event. Returns (snap, data_warnings, event, market_data, detection)."""
    from data.feeds_hardened import HardenedFeed, DataError
    from data.event_factory  import EventFactory, snapshot_to_pipeline_inputs

    feed    = HardenedFeed(**feed_kwargs, abort_on_stale=True)
    snap, data_warnings = feed.fetch_validated(sym)
    factory = EventFactory(sensitivity=sensitivity)
    event, market_data, detection = snapshot_to_pipeline_inputs(snap, factory, sym)
    return snap, data_warnings, event, market_data, detection


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    keys = _load_keys(args.mock)

    feed_kwargs = {
        "eia_key":     keys.get("EIA_API_KEY",    ""),
        "fred_key":    keys.get("FRED_API_KEY",   ""),
        "newsapi_key": keys.get("NEWSAPI_KEY",    ""),
    }

    # ── Pure diagnostics (no pipeline) ───────────────────────────────────

    if args.workflows:
        print_workflow_summary()
        return 0

    if args.validate_config:
        from config import cfg
        errs = cfg.validate()
        if errs:
            print("\n  ✗ Config errors:\n" + "\n".join(f"    {e}" for e in errs))
            return 1
        from data.event_factory import _is_crop_progress_day, _is_export_inspection_day
        today = date.today()
        print(f"\n  ✓ Config valid")
        print(f"  Today ({today}):")
        print(f"    Crop Progress day:    {_is_crop_progress_day(today)}")
        print(f"    Export Inspection day:{_is_export_inspection_day(today)}")

        # Verify RSI fix
        from data.event_factory import BASE_THRESHOLDS
        from data.universe import get_instrument
        zc = get_instrument("ZC")
        rsi_effective = BASE_THRESHOLDS["rsi_oversold"] * zc.threshold_scale
        if rsi_effective > 100:
            print(f"\n  ✗ RSI bug NOT fixed: ZC RSI effective={rsi_effective}")
        else:
            print(f"\n  ✓ RSI fix confirmed (ZC uses fixed 25/75 thresholds)")
        print()
        return 0

    if args.sources:
        from data.source_registry import print_source_table, available_sources
        print_source_table()
        for ac in ["commodity_agri", "equity_index", "fx_em"]:
            avail = available_sources(ac, dict(os.environ))
            print(f"  Available for {ac}: {[s.name for s in avail]}")
        print()
        return 0

    if args.cost_estimate:
        from utils.cost_tracker import CostTracker
        t = CostTracker(log_dir=Path("logs"))
        full = t.estimate_run_cost()
        lite = 0.04
        print(f"\n  Cost per pipeline run:")
        print(f"    Full mode (T3+T2+T1+desk): ~${full:.4f}")
        print(f"    Lite mode (T3+desk only):  ~${lite:.4f}")
        print(f"\n  Daily estimates:")
        print(f"    ags_daily    5 instr × 2 polls × 30% trigger × ${full:.3f}: ~${5*2*0.3*full:.2f}/day")
        print(f"    ags_intraday 5 instr × 10 polls × 60% trigger × ${lite:.3f}: ~${5*10*0.6*lite:.2f}/day")
        print(f"    india_swing  3 instr × 2 polls × 30% trigger × ${full:.3f}: ~${3*2*0.3*full:.2f}/day")
        print(f"\n  Today spent: ${t.today_spent:.4f} / cap ${t.daily_cap:.2f}\n")
        return 0

    if args.cost_today:
        from utils.cost_tracker import CostTracker
        print(f"\n  {CostTracker(log_dir=Path('logs')).summary()}\n")
        return 0

    if args.regression:
        import asyncio
        from utils.model_registry import run_regression
        print("\n  Running model regression tests...")
        results = asyncio.run(run_regression(keys["OPENAI_API_KEY"], Path("logs")))
        passed  = sum(1 for r in results if r.passed)
        print(f"\n  {passed}/{len(results)} passed\n")
        for r in results:
            status = "✓" if r.passed else f"✗  {'; '.join(r.failures)}"
            print(f"    [{r.role}/{r.model[:30]}]  {status}")
        print()
        return 0 if passed == len(results) else 1

    # ── Resolve workflow config ───────────────────────────────────────────
    wf = get_workflow(args.workflow) if args.workflow else None

    # ── Quant overlay commands ────────────────────────────────────────────
    if args.overlay_report:
        from pipeline.quant_overlay import QuantModelOverlay
        name    = wf.quant_model_name if wf and wf.quant_model_name else "default"
        overlay = QuantModelOverlay(model_name=name, log_dir=Path("logs"))
        print(overlay.agreement_report())
        return 0

    if args.log_quant is not None:
        parts = args.log_quant or []
        if len(parts) < 3:
            print("  Usage: --log-quant SIG_ID SYMBOL DIRECTION [CONVICTION] [NOTES...]")
            return 1
        from pipeline.quant_overlay import QuantModelOverlay
        name    = wf.quant_model_name if wf and wf.quant_model_name else "default"
        overlay = QuantModelOverlay(model_name=name, log_dir=Path("logs"))
        try:
            conv = float(parts[3]) if len(parts) > 3 else None
        except ValueError:
            conv = None
        notes = " ".join(parts[4:] if conv else parts[3:])
        overlay.log_model_view(
            signal_id       = parts[0],
            instrument      = parts[1].upper(),
            mm_direction    = "unknown",
            quant_direction = parts[2].lower(),
            quant_conviction = conv,
            notes           = notes,
        )
        print(f"  ✓ Logged quant view for signal {parts[0][:8]}: "
              f"{parts[1].upper()} {parts[2].lower()}"
              + (f" conv={conv:.2f}" if conv else "")
              + (f" — {notes}" if notes else ""))
        return 0

    # ── Build pipeline ────────────────────────────────────────────────────
    if wf and wf.lite_mode and not args.mock:
        pipeline = _make_lite_pipeline(keys, wf)
    else:
        pipeline = _make_pipeline(keys, wf, args.mock)

    # ── Forward test management ───────────────────────────────────────────
    if args.signal_status:
        mon    = pipeline.forward_monitor
        active = mon.active_signals
        susp   = mon.suspended_signals
        print(f"\n  Active signals:    {len(active)}")
        for r in sorted(active, key=lambda x: x.instrument):
            print(f"    [{r.signal_id[:8]}] {r.instrument:8} {r.direction.upper():5} "
                  f"conv={r.conviction:.0%}  expires={r.expiry_date}  "
                  f"seal={r.signal_seal}  universe={r.dominant_universe}  "
                  f"devil={r.devil_strength:.2f}")
        if susp:
            print(f"\n  Suspended signals: {len(susp)}")
            for r in susp:
                print(f"    [{r.signal_id[:8]}] {r.instrument:8} — {r.suspension_reason}")
        print(f"\n  {mon.summary()}\n")
        return 0

    if args.check_signals:
        print("\n  Polling prices for active signals...")
        resolved = pipeline.check_forward_signals(feed_kwargs)
        if not resolved:
            print("  No signals resolved.")
        else:
            # Update quant overlay with outcomes
            if wf and wf.quant_model_name:
                from pipeline.quant_overlay import QuantModelOverlay
                overlay = QuantModelOverlay(wf.quant_model_name, log_dir=Path("logs"))
                for rec in resolved:
                    if rec.outcome:
                        overlay.update_outcome(rec.signal_id, rec.outcome, rec.r_multiple or 0.0)
        print(f"\n  {pipeline.forward_monitor.summary()}\n")
        return 0

    if args.suspend:
        n = pipeline.suspend_signals(args.suspend.upper(), "manual_halt")
        print(f"\n  Suspended {n} signals for {args.suspend.upper()}\n")
        return 0

    if args.resume:
        n = pipeline.forward_monitor.resume_by_instrument(args.resume.upper())
        print(f"\n  Resumed {n} signals for {args.resume.upper()}\n")
        return 0

    if args.skip:
        sid   = args.skip[0]
        notes = " ".join(args.skip[1:]) if len(args.skip) > 1 else ""
        pipeline.forward_monitor.skip_signal(sid, notes)
        print(f"\n  Signal {sid[:8]} marked skipped.\n")
        return 0

    if args.run_feedback:
        print("\n  Running population feedback...")
        pipeline.run_feedback()
        return 0

    if args.feedback_report:
        print(pipeline.feedback.report())
        return 0

    # ── Workflow watch mode ───────────────────────────────────────────────
    if wf and args.watch:
        instruments = wf.instruments
        print(f"\n  {wf.name} — watching {len(instruments)} instruments")
        print(f"  Poll: {wf.poll_interval_sec//60}min  Sensitivity: {wf.sensitivity}")
        print(f"  Instruments: {', '.join(instruments)}  (Ctrl+C to stop)\n")

        from pipeline.quant_overlay import QuantModelOverlay
        overlay = QuantModelOverlay(
            model_name = wf.quant_model_name or "default",
            log_dir    = Path("logs"),
        ) if wf.quant_model_name else None

        from pipeline.dashboard import display_signal

        while True:
            for sym in instruments:
                if wf.session_aware and not _is_session_open(sym):
                    if not args.quiet:
                        print(f"  [{sym}] session closed — skip")
                    continue
                try:
                    snap, data_warnings, event, market_data, detection = _fetch_and_detect(
                        sym, wf.sensitivity, feed_kwargs
                    )
                    if not args.quiet:
                        print(f"  [{sym}] spot={snap.spot:,.3f}  chg={snap.change_pct:+.2f}%  "
                              f"triggered={detection.triggered}")

                    if not detection.triggered and not args.force_run:
                        continue

                    from data.universe import get_instrument
                    spec   = get_instrument(sym)
                    result = pipeline.run(
                        event        = event,
                        instrument   = sym,
                        exchange     = spec.exchange,
                        asset_class  = spec.asset_class,
                        market_data  = market_data,
                        data_warnings = data_warnings,
                    )
                    if result.success and result.live_signal_record:
                        display_signal(result.signal)
                        rec = result.live_signal_record
                        if overlay:
                            print(f"\n  Log your model's view for {sym}:")
                            print(f"    python main_v3.py --log-quant "
                                  f"{rec.signal_id[:8]} {sym} [long|short|flat] "
                                  f"[conviction] [notes]\n")
                except Exception as e:
                    print(f"  [{sym}] error: {e}")

            if not args.quiet:
                next_poll = datetime.utcnow()
                print(f"\n  Sleeping {wf.poll_interval_sec//60}min "
                      f"(next: {next_poll.strftime('%H:%M')} UTC + {wf.poll_interval_sec//60}min)\n")
            time.sleep(wf.poll_interval_sec)

    # ── Intraday session-aware loop ───────────────────────────────────────
    if args.watch_intraday:
        sym = args.watch_intraday.upper()
        print(f"\n  Intraday watch: {sym}  (30min polls · session-only · lite mode)")
        print("  Ctrl+C to stop\n")

        wf_intra   = get_workflow("ags_intraday")
        intra_pipe = _make_lite_pipeline(keys, wf_intra) if not args.mock else pipeline

        from pipeline.dashboard import display_signal as _dsig

        while True:
            if not _is_session_open(sym):
                if not args.quiet:
                    now = datetime.utcnow().strftime("%H:%M")
                    print(f"  [{sym} {now}] session closed — waiting 10min")
                time.sleep(600)
                continue

            try:
                snap, data_warnings, event, market_data, detection = _fetch_and_detect(
                    sym, "high", feed_kwargs
                )
                now = datetime.utcnow().strftime("%H:%M")
                print(f"  [{sym} {now} UTC] spot={snap.spot:,.3f}  "
                      f"chg={snap.change_pct:+.2f}%  "
                      f"atr%={snap.atr_pct:.2f}  triggered={detection.triggered}")

                if detection.triggered or args.force_run:
                    from data.universe import get_instrument
                    spec   = get_instrument(sym)
                    result = intra_pipe.run(
                        event        = event,
                        instrument   = sym,
                        exchange     = spec.exchange,
                        asset_class  = spec.asset_class,
                        market_data  = market_data,
                        data_warnings = data_warnings,
                    )
                    if result.success and result.live_signal_record:
                        sig = result.signal
                        print(f"\n  *** INTRADAY SIGNAL — {sym} ***")
                        print(f"    Direction:  {sig.direction.value.upper()}  "
                              f"conv={sig.conviction:.0%}")
                        print(f"    Entry:      {sig.entry_zone.low:.3f}–{sig.entry_zone.high:.3f}")
                        print(f"    Stop:       {sig.risk.stop_loss:.3f}")
                        print(f"    Target:     {sig.risk.target_1:.3f}")
                        print(f"    R:R:        {sig.risk.risk_reward:.2f}")
                        print(f"    Universe:   {result.live_signal_record.dominant_universe}")
                        print(f"    Divergence: {result.swarm_report.universe_divergence:.3f}")
                        print(f"    Seal:       {result.live_signal_record.signal_seal}\n")
                    elif result.error:
                        print(f"  Pipeline error: {result.error}")

            except Exception as e:
                print(f"  [{sym}] error: {e}")

            time.sleep(30 * 60)   # 30 minutes

    # ── Workflow single live run ──────────────────────────────────────────
    if wf and args.live:
        print(f"\n  {wf.name} — single live run across {len(wf.instruments)} instruments")
        from data.universe import get_instrument
        from pipeline.dashboard import display_pipeline_result

        exit_code = 0
        for sym in wf.instruments:
            print(f"\n  ── {sym} ───────────────────────────────────────")
            try:
                snap, data_warnings, event, market_data, detection = _fetch_and_detect(
                    sym, wf.sensitivity, feed_kwargs
                )
                print(f"  spot={snap.spot:,.4f}  chg={snap.change_pct:+.3f}%  "
                      f"triggered={detection.triggered}")
                if not detection.triggered and not args.force_run:
                    print(f"  No event ({detection.rationale})")
                    continue
                spec   = get_instrument(sym)
                result = pipeline.run(
                    event=event, instrument=sym,
                    exchange=spec.exchange, asset_class=spec.asset_class,
                    market_data=market_data, data_warnings=data_warnings,
                )
                display_pipeline_result(result, show_brief=not args.quiet)
                if not result.success:
                    exit_code = 1
            except Exception as e:
                print(f"  [{sym}] error: {e}")
                exit_code = 1
        return exit_code

    # ── Single instrument live run ────────────────────────────────────────
    if args.live:
        sym = args.instrument.upper()
        from data.feeds_hardened import HardenedFeed, DataError
        from data.universe import get_instrument
        from pipeline.dashboard import display_pipeline_result

        try:
            snap, data_warnings, event, market_data, detection = _fetch_and_detect(
                sym, args.sensitivity, feed_kwargs
            )
        except Exception as e:
            print(f"\n  ✗ Data error: {e}\n")
            return 1

        print(f"  {sym}: spot={snap.spot:,.4f}  chg={snap.change_pct:+.3f}%")
        print(f"  Event triggered: {detection.triggered}"
              + (f" — {detection.description[:70]}" if detection.triggered
                 else f" — {detection.rationale}"))
        if not detection.triggered and not args.force_run:
            print("  No event. Use --force-run to analyse anyway.")
            return 0

        spec   = get_instrument(sym)
        result = pipeline.run(
            event=event, instrument=sym,
            exchange=spec.exchange, asset_class=spec.asset_class,
            market_data=market_data, data_warnings=data_warnings,
        )
        display_pipeline_result(result, show_brief=not args.quiet)
        return 0 if result.success else 1

    # ── Mock / scenario mode ──────────────────────────────────────────────
    if args.mock:
        try:
            from main import SCENARIOS
        except Exception:
            SCENARIOS = {}

        if wf:
            # Pick ags scenario for ags workflows
            if "ags" in wf.name:
                scenario_name = "wheat_wasde"
                sym           = "ZW"
            else:
                scenario_name = "rbi_hold"
                sym           = "NIFTY50"
            scenario = SCENARIOS.get(scenario_name) or list(SCENARIOS.values())[0]
        else:
            scenario      = list(SCENARIOS.values())[0] if SCENARIOS else None
            scenario_name = "default"
            sym           = args.instrument.upper()

        if not scenario:
            print("  No scenarios available. Run --validate-config first.")
            return 1

        from data.universe import get_instrument
        from pipeline.dashboard import display_pipeline_result

        print(f"\n  Mock run: scenario={scenario_name}  instrument={sym}\n")
        spec   = get_instrument(sym)
        result = pipeline.run(
            event       = scenario["event"],
            instrument  = sym,
            exchange    = spec.exchange,
            asset_class = spec.asset_class,
            market_data = scenario.get("market_data", {}),
        )
        display_pipeline_result(result, show_brief=not args.quiet)
        return 0 if result.success else 1

    # ── No mode specified ─────────────────────────────────────────────────
    print("\n  Quick start:")
    print("    python main_v3.py --validate-config")
    print("    python main_v3.py --sources")
    print("    python main_v3.py --mock --workflow ags_daily")
    print("    python main_v3.py --live  --workflow ags_daily\n")
    print_workflow_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
