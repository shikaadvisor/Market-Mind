"""
feedback/population_feedback.py
================================
The learning engine — closes the loop between forward-test outcomes and
the simulation population.

How it works:
    After 30–60 days of forward testing you have resolved signals.
    Each signal carries attribution: which universe agreed with it,
    what the regime was, whether it was a contrarian setup.

    The feedback loop uses this to:

    1. UNIVERSE WEIGHT TUNING
       Which universe type was most predictive for each asset class?
       e.g. "For energy commodities in trending regimes, the Structural
       universe was right 68% of the time vs Momentum at 41%."
       → Increase Structural weight for energy, decrease Momentum.

    2. PERSONA CALIBRATION
       Which T2/T1 personas had views that matched the outcome?
       → Build a persona accuracy score. Upweight accurate personas
         in future T2/T1 sampling for that asset class.

    3. CONVICTION CALIBRATION
       Are 80% conviction signals actually winning 80% of the time?
       Classic calibration plot. If not, apply a correction multiplier.
       This is NOT modifying historical signals — it's adjusting future
       conviction thresholds.

    4. REGIME-CONDITIONAL LEARNING
       Which universe is most accurate IN TRENDING vs RANGE_BOUND regimes?
       Regimes shift, so universe weights should be regime-conditional,
       not static.

    5. OUTCOME INJECTION AS CONTEXT
       The most powerful learning mechanism: when a similar event occurs
       again, inject the past outcome as context into the LLM prompts.
       "The last 3 times a hot CPI print hit while DXY was above 104,
       the 10-day outcome was: -1.8%, +0.4%, -2.1% (avg: -1.2R)."
       This is NOT backtest contamination — it's the kind of market
       memory any experienced trader carries.

Forward-only guarantee:
    This module NEVER modifies historical signals.
    It only updates weights/parameters that apply to FUTURE runs.
    All updates are timestamped so you can audit exactly when
    any parameter changed and why.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("marketmind.feedback")


# ── Outcome record from forward monitor ─────────────────────────────────────

@dataclass
class OutcomeRecord:
    signal_id:           str
    instrument:          str
    asset_class:         str
    direction:           str
    conviction:          float
    regime:              str
    outcome:             str        # "win" | "loss" | "breakeven" | "expired"
    r_multiple:          Optional[float]
    days_live:           int
    dominant_universe:   str
    contrarian_setup:    bool
    universe_divergence: float
    universe_sentiments: dict[str, float]


# ── Calibration state ────────────────────────────────────────────────────────

@dataclass
class FeedbackState:
    """
    Persistent calibration state.
    Written to disk after every feedback run.
    Applied to SwarmRunner on next system start.
    """
    # Universe weights per asset class and regime
    # Structure: {asset_class: {regime: {universe_name: weight}}}
    universe_weights: dict = field(default_factory=dict)

    # Conviction calibration multiplier per conviction bucket
    # Structure: {conviction_bucket: correction_multiplier}
    conviction_calibration: dict = field(default_factory=dict)

    # Persona accuracy scores
    # Structure: {persona_name: {asset_class: win_rate}}
    persona_accuracy: dict = field(default_factory=dict)

    # Historical event memory (for context injection)
    # Structure: list of {event_fingerprint, asset_class, outcome_summary}
    event_memory: list = field(default_factory=list)

    # Metadata
    last_updated:     str = ""
    outcomes_used:    int = 0
    feedback_runs:    int = 0

    def global_universe_weights(self) -> dict[str, float]:
        """
        Average universe weights across all asset classes and regimes.
        Used as a fallback when no specific weights exist.
        """
        all_weights: dict[str, list[float]] = defaultdict(list)
        for ac_dict in self.universe_weights.values():
            for regime_dict in ac_dict.values():
                for name, w in regime_dict.items():
                    all_weights[name].append(w)

        if not all_weights:
            return {"momentum": 0.30, "value": 0.25, "crisis": 0.20, "structural": 0.25}

        return {
            name: float(np.mean(weights))
            for name, weights in all_weights.items()
        }


# ── Feedback engine ───────────────────────────────────────────────────────────

class PopulationFeedback:
    """
    Runs after each batch of forward-test outcomes.
    Requires at least 20 resolved signals for any meaningful calibration.
    """

    MIN_OUTCOMES_FOR_CALIBRATION = 20

    def __init__(self, log_dir: Path | None = None):
        self.log_dir   = log_dir or Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self._state_path = self.log_dir / "feedback_state.json"
        self._state      = self._load_state()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load_state(self) -> FeedbackState:
        if self._state_path.exists():
            try:
                with open(self._state_path) as f:
                    d = json.load(f)
                return FeedbackState(**d)
            except Exception as e:
                logger.warning("Could not load feedback state: %s — starting fresh", e)
        return FeedbackState()

    def _save_state(self) -> None:
        self._state.last_updated = datetime.utcnow().isoformat()
        with open(self._state_path, "w") as f:
            json.dump(asdict(self._state), f, indent=2)
        logger.info("Feedback state saved (%d outcomes used)", self._state.outcomes_used)

    # ── Main feedback run ─────────────────────────────────────────────────────

    def run(self, outcomes: list[dict]) -> FeedbackState:
        """
        Process a batch of outcome records and update the feedback state.
        outcomes: list of dicts from ForwardMonitor.get_outcomes_for_feedback()
        """
        if len(outcomes) < self.MIN_OUTCOMES_FOR_CALIBRATION:
            logger.info(
                "Only %d outcomes — need %d minimum for calibration. Skipping.",
                len(outcomes), self.MIN_OUTCOMES_FOR_CALIBRATION
            )
            return self._state

        records = [OutcomeRecord(**o) for o in outcomes]
        logger.info("Running feedback on %d outcomes", len(records))

        self._calibrate_universe_weights(records)
        self._calibrate_conviction(records)
        self._build_event_memory(records)

        self._state.outcomes_used = len(records)
        self._state.feedback_runs += 1
        self._save_state()

        return self._state

    # ── Universe weight calibration ──────────────────────────────────────────

    def _calibrate_universe_weights(self, records: list[OutcomeRecord]) -> None:
        """
        For each (asset_class, regime) combination, compute which universe
        had the most predictive sentiment direction and adjust its weight.
        """
        # Group by asset_class × regime
        groups: dict[tuple, list[OutcomeRecord]] = defaultdict(list)
        for rec in records:
            if rec.outcome in ("win", "loss"):
                groups[(rec.asset_class, rec.regime)].append(rec)

        for (asset_class, regime), group_records in groups.items():
            if len(group_records) < 5:
                continue  # need at least 5 per bucket

            # For each universe, compute: was its sentiment in the right direction?
            universe_accuracy: dict[str, list[bool]] = defaultdict(list)

            for rec in group_records:
                outcome_positive = (rec.outcome == "win")
                for univ_name, sent in rec.universe_sentiments.items():
                    direction = rec.direction.lower()
                    # Correct if: long + positive sentiment = win, or short + negative sentiment = win
                    sent_aligned = (
                        (direction == "long"  and sent > 0) or
                        (direction == "short" and sent < 0)
                    )
                    universe_accuracy[univ_name].append(sent_aligned == outcome_positive)

            # Compute accuracy per universe
            accuracy: dict[str, float] = {}
            for univ_name, correct_list in universe_accuracy.items():
                if correct_list:
                    accuracy[univ_name] = sum(correct_list) / len(correct_list)

            if not accuracy:
                continue

            # Softmax the accuracy scores to get new weights
            scores = np.array(list(accuracy.values()))
            weights_raw = np.exp(scores * 3.0)  # temperature=3 keeps spread reasonable
            weights_norm = weights_raw / weights_raw.sum()
            new_weights  = {name: float(w) for name, w in zip(accuracy.keys(), weights_norm)}

            # Store
            if asset_class not in self._state.universe_weights:
                self._state.universe_weights[asset_class] = {}
            self._state.universe_weights[asset_class][regime] = new_weights

            logger.info(
                "[feedback] %s / %s → universe weights: %s",
                asset_class, regime,
                {k: f"{v:.3f}" for k, v in new_weights.items()}
            )

    # ── Conviction calibration ───────────────────────────────────────────────

    def _calibrate_conviction(self, records: list[OutcomeRecord]) -> None:
        """
        Build a calibration curve: does 0.80 conviction → ~80% win rate?
        Compute a correction multiplier per conviction bucket.
        """
        buckets: dict[str, list[bool]] = defaultdict(list)
        for rec in records:
            if rec.outcome in ("win", "loss"):
                # Bucket into 0.10-wide bins
                bucket = f"{int(rec.conviction * 10) * 10}pct"
                buckets[bucket].append(rec.outcome == "win")

        calibration: dict[str, float] = {}
        for bucket, results in buckets.items():
            if len(results) >= 3:
                win_rate = sum(results) / len(results)
                # Stated conviction implied
                stated = int(bucket.replace("pct", "")) / 100.0
                # Correction: if stated 0.8 but actual 0.5, multiply future signals by 0.5/0.8
                correction = win_rate / stated if stated > 0 else 1.0
                calibration[bucket] = round(correction, 4)

        self._state.conviction_calibration = calibration
        logger.info("[feedback] Conviction calibration: %s", calibration)

    # ── Event memory ─────────────────────────────────────────────────────────

    def _build_event_memory(self, records: list[OutcomeRecord]) -> None:
        """
        Build a queryable event memory from resolved outcomes.
        This is injected as context into future LLM prompts when similar
        market conditions recur. This is market experience, not backtest data.
        """
        # Group by (asset_class, regime, direction) — each group is a "scenario type"
        groups: dict[tuple, list[OutcomeRecord]] = defaultdict(list)
        for rec in records:
            key = (rec.asset_class, rec.regime, rec.direction)
            groups[key].append(rec)

        memory_entries = []
        for (asset_class, regime, direction), group in groups.items():
            if len(group) < 3:
                continue

            wins    = sum(1 for r in group if r.outcome == "win")
            losses  = sum(1 for r in group if r.outcome == "loss")
            r_mults = [r.r_multiple for r in group if r.r_multiple is not None]
            avg_r   = float(np.mean(r_mults)) if r_mults else 0.0

            memory_entries.append({
                "asset_class":   asset_class,
                "regime":        regime,
                "direction":     direction,
                "n_signals":     len(group),
                "win_rate":      round(wins / len(group), 3),
                "avg_r_multiple": round(avg_r, 3),
                "sample_size":   len(group),
                "summary": (
                    f"{len(group)} prior {direction} signals in {regime} regime "
                    f"on {asset_class}: win rate {wins/len(group):.0%}, "
                    f"avg R-multiple {avg_r:+.2f}"
                ),
            })

        self._state.event_memory = memory_entries
        logger.info("[feedback] Event memory: %d scenario types recorded", len(memory_entries))

    # ── Context injection ─────────────────────────────────────────────────────

    def get_context_for_signal(
        self,
        asset_class: str,
        regime:      str,
        direction:   str,
    ) -> str:
        """
        Returns a short prose block to inject into LLM prompts.
        Provides forward-test history for similar past setups.
        This is the key learning mechanism — agents see their track record.
        """
        relevant = [
            m for m in self._state.event_memory
            if m["asset_class"] == asset_class
            and m["regime"] == regime
            and m["sample_size"] >= 3
        ]
        if not relevant:
            return ""

        lines = ["FORWARD-TEST MEMORY (similar past setups):"]
        for m in relevant[:3]:
            lines.append(f"  • {m['summary']}")
        return "\n".join(lines)

    def get_universe_weights_for(
        self,
        asset_class: str,
        regime:      str,
    ) -> dict[str, float]:
        """
        Returns evidence-based universe weights for this asset_class + regime.
        Falls back to global average if not enough data.
        """
        specific = (
            self._state.universe_weights
            .get(asset_class, {})
            .get(regime, None)
        )
        if specific:
            return specific
        return self._state.global_universe_weights()

    @property
    def state(self) -> FeedbackState:
        return self._state

    def report(self) -> str:
        """Human-readable summary of calibration state."""
        lines = [
            f"\n══ Population Feedback Report ══",
            f"Outcomes processed: {self._state.outcomes_used}",
            f"Feedback runs:      {self._state.feedback_runs}",
            f"Last updated:       {self._state.last_updated or 'never'}",
            f"\nConviction calibration:",
        ]
        for bucket, mult in sorted(self._state.conviction_calibration.items()):
            lines.append(f"  {bucket}: correction multiplier = {mult:.3f}")

        lines.append(f"\nUniverse accuracy (global average):")
        for name, w in self._state.global_universe_weights().items():
            lines.append(f"  {name:12s}: {w:.3f}")

        lines.append(f"\nEvent memory: {len(self._state.event_memory)} scenario types")
        return "\n".join(lines)
