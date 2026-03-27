"""
config.py  —  v3
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT     = Path(__file__).parent.resolve()
LOGS_DIR = ROOT / "logs"
DATA_DIR = ROOT / "data"
LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


@dataclass
class Config:
    # API keys
    OPENAI_API_KEY:    str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    ANTHROPIC_API_KEY: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    LANGSMITH_API_KEY: str = field(default_factory=lambda: os.environ.get("LANGSMITH_API_KEY", ""))

    # T3
    TIER3_AGENT_COUNT:          int   = 10_000
    TIER3_SIMULATION_STEPS:     int   = 50
    TIER3_MOMENTUM_SENSITIVITY: float = 0.6
    TIER3_NEWS_REACTIVITY:      float = 0.7
    TIER3_LOSS_AVERSION:        float = 2.5
    TIER3_HERDING_THRESHOLD:    float = 0.55
    TIER3_PANIC_THRESHOLD:      float = 0.65

    # T2
    TIER2_AGENT_COUNT:  int   = 200
    TIER2_BATCH_SIZE:   int   = 20
    TIER2_MAX_TOKENS:   int   = 300
    TIER2_TEMPERATURE:  float = 0.8
    TIER2_PERSONA_WEIGHTS: dict = field(default_factory=lambda: {
        "retail_analyst": 0.30, "options_trader": 0.20,
        "fin_twitter_influencer": 0.20, "sector_fund_manager": 0.15,
        "momentum_trader": 0.15,
    })

    # T1
    TIER1_AGENT_COUNT: int   = 10
    TIER1_MAX_TOKENS:  int   = 1200
    TIER1_TEMPERATURE: float = 0.4

    # Risk rules
    MAX_POSITION_SIZE_PCT: float = 0.05
    MIN_RISK_REWARD:       float = 1.5

    # Cost guard (Risk 10)
    DAILY_COST_CAP_USD: float = 20.0

    # Devil's advocate (Risk 1)
    DEVIL_TRIGGER_THRESHOLD: float = 0.70

    # Desk weights
    WEIGHT_TECHNICAL:  float = 0.30
    WEIGHT_MACRO:      float = 0.25
    WEIGHT_CROWD_SIM:  float = 0.25
    WEIGHT_RISK:       float = 0.20

    # Feedback
    FEEDBACK_MIN_OUTCOMES: int = 20

    # Logging
    LOG_DIR:       Path = field(default_factory=lambda: Path(__file__).parent / "logs")
    LOG_SIGNALS:   bool = True
    MAX_LOG_LINES: int  = 10_000

    # Model names — pinned via registry (Risk 3 / 9)
    @property
    def OPENAI_MODEL_TIER2(self) -> str:
        try: from utils.model_registry import get; return get("tier2").name
        except Exception: return "gpt-4o-mini-2024-07-18"

    @property
    def OPENAI_MODEL_TIER1(self) -> str:
        try: from utils.model_registry import get; return get("tier1").name
        except Exception: return "gpt-4o-2024-11-20"

    @property
    def OPENAI_MODEL_DESK(self) -> str:
        try: from utils.model_registry import get; return get("desk_macro").name
        except Exception: return "gpt-4o-2024-11-20"

    @property
    def ANTHROPIC_MODEL_COORD(self) -> str:
        try: from utils.model_registry import get; return get("coordinator").name
        except Exception: return "claude-opus-4-6"

    def validate(self) -> list[str]:
        errors = []
        w = round(self.WEIGHT_TECHNICAL + self.WEIGHT_MACRO + self.WEIGHT_CROWD_SIM + self.WEIGHT_RISK, 4)
        if abs(w - 1.0) > 0.001:
            errors.append(f"Desk weights must sum to 1.0, got {w}")
        return errors

cfg = Config()
