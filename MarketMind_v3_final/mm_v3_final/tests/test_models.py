"""
tests/test_models.py
====================
Run with: pytest tests/test_models.py -v

These tests validate the schema contracts before any agent is written.
If these pass, every agent that produces a TradingSignal or SimulationReport
is guaranteed to produce valid, internally consistent output — or fail loudly.

Tests are organised by schema section, then by validator rule.
"""

import pytest
from datetime import date, datetime, timedelta
from unittest.mock import patch

# ── We patch date.today() so the point_in_time validators don't fail in CI ──
TODAY = date.today()


# ===========================================================================
# TradingSignal tests
# ===========================================================================

class TestPriceZone:
    def test_valid_zone(self):
        from models.signal import PriceZone
        z = PriceZone(low=100.0, high=110.0)
        assert z.midpoint == 105.0
        assert z.width_pct > 0

    def test_low_must_be_below_high(self):
        from models.signal import PriceZone
        with pytest.raises(Exception):
            PriceZone(low=110.0, high=100.0)

    def test_equal_low_high_rejected(self):
        from models.signal import PriceZone
        with pytest.raises(Exception):
            PriceZone(low=100.0, high=100.0)


class TestRiskParameters:
    def test_valid_risk(self):
        from models.signal import RiskParameters
        r = RiskParameters(
            stop_loss=95.0, target_1=115.0, risk_reward=4.0,
            max_position_pct=0.04, invalidation_level=90.0
        )
        assert r.stop_loss == 95.0

    def test_position_pct_bounds(self):
        from models.signal import RiskParameters
        with pytest.raises(Exception):
            RiskParameters(
                stop_loss=95.0, target_1=115.0, risk_reward=4.0,
                max_position_pct=1.5,   # > 1.0 — invalid
                invalidation_level=90.0
            )


class TestCrowdMetrics:
    def test_valid_crowd(self):
        from models.signal import CrowdMetrics
        c = CrowdMetrics(
            aggregate_sentiment=0.3, retail_sentiment=0.5,
            institutional_sentiment=0.1, herd_index=0.6,
            panic_probability=0.2, narrative_momentum=0.4,
            contrarian_pressure=0.15, information_velocity=0.7,
            opinion_cluster_count=3, agent_count=10200,
            simulation_seed_event="RBI holds rates", simulation_run_id="abc123"
        )
        assert c.aggregate_sentiment == 0.3

    def test_sentiment_out_of_range(self):
        from models.signal import CrowdMetrics
        with pytest.raises(Exception):
            CrowdMetrics(
                aggregate_sentiment=1.5,  # > 1.0 — invalid
                retail_sentiment=0.5, institutional_sentiment=0.1,
                herd_index=0.6, panic_probability=0.2,
                narrative_momentum=0.4, contrarian_pressure=0.15,
                information_velocity=0.7, opinion_cluster_count=3,
                agent_count=10200, simulation_seed_event="RBI holds rates",
                simulation_run_id="abc123"
            )


def make_valid_signal(**overrides):
    """Factory for a valid TradingSignal with all required fields."""
    from models.signal import (
        TradingSignal, Direction, Timeframe, AssetClass,
        RegimeType, PriceZone, RiskParameters, CrowdMetrics, AgentContribution
    )
    defaults = dict(
        signal_id="sig-001",
        created_at=datetime.utcnow(),
        point_in_time_date=TODAY,
        instrument="NIFTY50",
        exchange="NSE",
        asset_class=AssetClass.EQUITY,
        direction=Direction.LONG,
        conviction=0.75,
        timeframe=Timeframe.SWING,
        regime=RegimeType.TRENDING_UP,
        entry_zone=PriceZone(low=24200, high=24350),
        risk=RiskParameters(
            stop_loss=23900, target_1=24900,
            risk_reward=2.3, max_position_pct=0.04,
            invalidation_level=23700
        ),
        crowd=CrowdMetrics(
            aggregate_sentiment=-0.35, retail_sentiment=-0.5,
            institutional_sentiment=0.1, herd_index=0.71,
            panic_probability=0.42, narrative_momentum=0.60,
            contrarian_pressure=0.25, information_velocity=0.65,
            opinion_cluster_count=4, agent_count=10200,
            simulation_seed_event="US inflation print beats expectations",
            simulation_run_id="run-abc123"
        ),
        macro_context="US CPI came in at 3.1%, above the 2.9% estimate. "
                      "Fed repricing hawkish. India macro relatively insulated "
                      "but FII outflows likely in the near term.",
        technical_view="NIFTY50 held the 24,000 support on the daily. "
                       "Structure is bullish above 24,000. Momentum indicators "
                       "turning up from oversold. Entry zone 24,200–24,350.",
        crowd_narrative="Simulation shows retail in fear (sentiment -0.5) "
                        "while institutions mildly bullish (+0.1). High herding "
                        "at 71% — crowd is correlated. Contrarian institutional "
                        "pressure at 25% suggests a potential reversal setup.",
        coordinator_thesis="Setup: retail fear + institutional buy = classic "
                           "contrarian long. Technical structure supports. "
                           "Risk is FII outflows accelerating — manage with "
                           "strict stop at 23,900. R:R 2.3 acceptable.",
        agent_contributions=[
            AgentContribution(
                agent_name="MacroAgent", agent_role="macro_strategist",
                view="long", conviction=0.60,
                key_factors=["india_macro_resilient", "rbi_on_hold"]
            ),
            AgentContribution(
                agent_name="TechnicalAgent", agent_role="technical_analyst",
                view="long", conviction=0.82,
                key_factors=["support_held", "momentum_reversal", "structure_intact"]
            ),
            AgentContribution(
                agent_name="RiskAgent", agent_role="risk_manager",
                view="long", conviction=0.70,
                key_factors=["rr_acceptable", "position_size_ok"]
            ),
        ],
    )
    defaults.update(overrides)
    return TradingSignal(**defaults)


class TestTradingSignal:
    def test_valid_signal_creates(self):
        s = make_valid_signal()
        assert s.instrument == "NIFTY50"
        assert s.is_actionable is True

    def test_point_in_time_must_be_today(self):
        yesterday = TODAY - timedelta(days=1)
        with pytest.raises(Exception, match="forward"):
            make_valid_signal(point_in_time_date=yesterday)

    def test_future_date_rejected(self):
        tomorrow = TODAY + timedelta(days=1)
        with pytest.raises(Exception):
            make_valid_signal(point_in_time_date=tomorrow)

    def test_neutral_high_conviction_rejected(self):
        from models.signal import Direction
        with pytest.raises(Exception, match="NEUTRAL"):
            make_valid_signal(direction=Direction.NEUTRAL, conviction=0.75)

    def test_neutral_low_conviction_accepted(self):
        from models.signal import Direction
        s = make_valid_signal(direction=Direction.NEUTRAL, conviction=0.35)
        assert s.direction == Direction.NEUTRAL

    def test_futures_requires_expiry(self):
        from models.signal import AssetClass
        with pytest.raises(Exception, match="expiry"):
            make_valid_signal(asset_class=AssetClass.FUTURES, expiry=None)

    def test_futures_with_expiry_accepted(self):
        from models.signal import AssetClass
        expiry = date(2026, 4, 24)
        s = make_valid_signal(
            asset_class=AssetClass.FUTURES,
            instrument="NIFTY26APRFUT",
            expiry=expiry
        )
        assert s.expiry == expiry

    def test_is_actionable_false_when_low_conviction(self):
        s = make_valid_signal(conviction=0.55)
        assert s.is_actionable is False

    def test_is_actionable_false_when_vetoed(self):
        s = make_valid_signal(
            conviction=0.85,
            do_not_trade_flags=["earnings_in_2_days"]
        )
        assert s.is_actionable is False

    def test_signal_summary_contains_key_fields(self):
        s = make_valid_signal()
        summary = s.signal_summary
        assert "LONG" in summary
        assert "NIFTY50" in summary
        assert "75%" in summary

    def test_crowd_summary_labels_extreme_fear(self):
        from models.signal import CrowdMetrics
        crowd = CrowdMetrics(
            aggregate_sentiment=-0.85, retail_sentiment=-0.9,
            institutional_sentiment=-0.6, herd_index=0.9,
            panic_probability=0.85, narrative_momentum=-0.7,
            contrarian_pressure=0.05, information_velocity=0.95,
            opinion_cluster_count=1, agent_count=10200,
            simulation_seed_event="Major crash trigger",
            simulation_run_id="run-crisis"
        )
        s = make_valid_signal(crowd=crowd)
        assert "extreme fear" in s.crowd_summary

    def test_to_log_dict_is_flat(self):
        s = make_valid_signal()
        d = s.to_log_dict()
        assert "entry_low" in d
        assert "crowd_sentiment" in d
        assert "herd_index" in d
        # No nested dicts in log output
        for v in d.values():
            assert not isinstance(v, dict)


# ===========================================================================
# SimulationReport tests
# ===========================================================================

def make_valid_report(**overrides):
    from models.simulation import (
        SimulationReport, NarrativeType, CascadeRisk,
        TierSnapshot, OpinionCluster
    )
    defaults = dict(
        run_id="run-001",
        created_at=datetime.utcnow(),
        point_in_time_date=TODAY,
        seed_event="RBI holds rates at 6.5% — in line with expectations",
        instrument="NIFTY50",
        simulation_steps=50,
        total_agents=10210,
        tier_snapshots=[
            TierSnapshot(
                tier=3, agent_count=10000,
                bullish_pct=0.35, bearish_pct=0.45, neutral_pct=0.20,
                avg_conviction=0.55,
                notable_behaviors=["panic_selling_cluster", "stop_cascade_risk"]
            ),
            TierSnapshot(
                tier=2, agent_count=200,
                bullish_pct=0.40, bearish_pct=0.35, neutral_pct=0.25,
                avg_conviction=0.62
            ),
            TierSnapshot(
                tier=1, agent_count=10,
                bullish_pct=0.60, bearish_pct=0.30, neutral_pct=0.10,
                avg_conviction=0.78
            ),
        ],
        dominant_narrative=NarrativeType.ACCUMULATION,
        narrative_confidence=0.68,
        narrative_shift_detected=False,
        opinion_clusters=[
            OpinionCluster(
                cluster_id="c1", size_pct=0.55,
                dominant_view="Bearish retail crowd expecting further downside",
                conviction=0.60, is_influential=False
            ),
            OpinionCluster(
                cluster_id="c2", size_pct=0.30,
                dominant_view="Institutional accumulation at these levels",
                conviction=0.75, is_influential=True
            ),
            OpinionCluster(
                cluster_id="c3", size_pct=0.15,
                dominant_view="Neutral, waiting for clarity",
                conviction=0.40, is_influential=False
            ),
        ],
        cascade_risk=CascadeRisk.LOW,
        aggregate_sentiment=-0.25,
        retail_sentiment=-0.45,
        institutional_sentiment=0.35,
        herd_index=0.58,
        panic_probability=0.22,
        narrative_momentum=0.30,
        contrarian_pressure=0.30,
        information_velocity=0.55,
        bull_case=(
            "Institutions are accumulating at these levels. RBI on hold removes "
            "rate uncertainty. Domestic flows remain strong. Any global risk-on "
            "catalyst could trigger a sharp rally from current oversold levels."
        ),
        bear_case=(
            "Retail crowd is heavily net short. FII outflows are persistent. "
            "US rate repricing creates headwind for EM equity flows. "
            "Technical breakdown below 23,800 could accelerate selling."
        ),
        key_risk="Sudden FII outflow acceleration triggered by US data surprise",
        contrarian_thesis=(
            "Retail fear at 6-month high while institutions accumulate — "
            "historically a bullish setup in Indian markets."
        ),
        sentiment_interpreter_brief=(
            "BRIEF FOR DESK: Simulation of 10,210 agents reacting to RBI hold. "
            "Key finding: retail (55% of population) is bearish and herded, "
            "while Tier 1 institutional agents (10 deep thinkers) are 60% bullish "
            "with high conviction. This divergence — retail fear vs institutional "
            "accumulation — historically precedes a recovery in NIFTY. "
            "Cascade risk is LOW. No panic trigger conditions met. "
            "Dominant narrative is ACCUMULATION with 68% confidence. "
            "Recommend: treat crowd sentiment as a contrarian bullish signal. "
            "Weight: moderate (0.25 of final conviction)."
        ),
    )
    defaults.update(overrides)
    return SimulationReport(**defaults)


class TestSimulationReport:
    def test_valid_report_creates(self):
        r = make_valid_report()
        assert r.instrument == "NIFTY50"
        assert r.total_agents == 10210

    def test_point_in_time_must_be_today(self):
        yesterday = TODAY - timedelta(days=1)
        with pytest.raises(Exception, match="forward"):
            make_valid_report(point_in_time_date=yesterday)

    def test_tier_counts_must_match_total(self):
        from models.simulation import TierSnapshot
        bad_snapshots = [
            TierSnapshot(
                tier=3, agent_count=5000,  # too few — won't match total_agents=10210
                bullish_pct=0.4, bearish_pct=0.4, neutral_pct=0.2, avg_conviction=0.5
            )
        ]
        with pytest.raises(Exception, match="tier agent counts"):
            make_valid_report(tier_snapshots=bad_snapshots)

    def test_tier_pcts_must_sum_to_one(self):
        from models.simulation import TierSnapshot
        with pytest.raises(Exception):
            TierSnapshot(
                tier=2, agent_count=200,
                bullish_pct=0.5, bearish_pct=0.5, neutral_pct=0.5,  # sums to 1.5
                avg_conviction=0.6
            )

    def test_has_cascade_risk_high(self):
        from models.simulation import CascadeRisk
        r = make_valid_report(cascade_risk=CascadeRisk.HIGH)
        assert r.has_cascade_risk is True

    def test_has_cascade_risk_low(self):
        from models.simulation import CascadeRisk
        r = make_valid_report(cascade_risk=CascadeRisk.LOW)
        assert r.has_cascade_risk is False

    def test_is_consensus_strong(self):
        r = make_valid_report(narrative_confidence=0.82, herd_index=0.75)
        assert r.is_consensus_strong is True

    def test_is_consensus_weak(self):
        r = make_valid_report(narrative_confidence=0.45, herd_index=0.30)
        assert r.is_consensus_strong is False

    def test_report_summary_string(self):
        r = make_valid_report()
        summary = r.report_summary
        assert "NIFTY50" in summary
        assert "accumulation" in summary


# ===========================================================================
# Config tests
# ===========================================================================

class TestConfig:
    def test_weights_sum_to_one(self):
        from config import cfg
        total = (
            cfg.WEIGHT_TECHNICAL + cfg.WEIGHT_MACRO +
            cfg.WEIGHT_CROWD_SIM + cfg.WEIGHT_RISK
        )
        assert abs(total - 1.0) < 0.01

    def test_persona_weights_sum_to_one(self):
        from config import cfg
        total = sum(cfg.TIER2_PERSONA_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01

    def test_actionable_threshold_above_log_only(self):
        from config import cfg
        assert cfg.SIGNAL_ACTIONABLE_THRESHOLD > cfg.SIGNAL_LOG_ONLY_THRESHOLD

    def test_broker_api_off_by_default(self):
        from config import cfg
        assert cfg.ENABLE_BROKER_API is False
        assert cfg.PAPER_TRADE is True


if __name__ == "__main__":
    # Run without pytest for quick local check
    import sys

    tests = [
        TestPriceZone(), TestRiskParameters(), TestCrowdMetrics(),
        TestTradingSignal(), TestSimulationReport(), TestConfig()
    ]
    passed = failed = 0
    for suite in tests:
        for name in [m for m in dir(suite) if m.startswith("test_")]:
            try:
                getattr(suite, name)()
                print(f"  ✓ {suite.__class__.__name__}.{name}")
                passed += 1
            except Exception as e:
                print(f"  ✗ {suite.__class__.__name__}.{name}: {e}")
                failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
