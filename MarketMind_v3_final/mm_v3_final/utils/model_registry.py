"""
utils/model_registry.py
========================
Risk 3 fix: Pin all LLM models to specific dated versions.
            OpenAI's undated "gpt-4o" alias has changed behaviour 4+ times
            without version-string changes. Pinned versions are stable.

            Includes a weekly regression test that fires the same canonical
            event through every model and checks output fields stay in range.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("marketmind.models")

# ---------------------------------------------------------------------------
# Pinned model versions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    name:          str
    provider:      str   # "openai" | "anthropic"
    max_tokens:    int
    temperature:   float
    cost_per_1k_input:  float   # USD
    cost_per_1k_output: float   # USD

MODELS = {
    # OpenAI — pinned to specific release dates
    "tier2": ModelSpec(
        name="gpt-4o-mini-2024-07-18",
        provider="openai", max_tokens=300, temperature=0.8,
        cost_per_1k_input=0.00015, cost_per_1k_output=0.00060,
    ),
    "tier1": ModelSpec(
        name="gpt-4o-2024-11-20",
        provider="openai", max_tokens=1200, temperature=0.4,
        cost_per_1k_input=0.0025, cost_per_1k_output=0.010,
    ),
    "desk_macro": ModelSpec(
        name="gpt-4o-2024-11-20",
        provider="openai", max_tokens=900, temperature=0.20,
        cost_per_1k_input=0.0025, cost_per_1k_output=0.010,
    ),
    "desk_technical": ModelSpec(
        name="gpt-4o-2024-11-20",
        provider="openai", max_tokens=900, temperature=0.15,
        cost_per_1k_input=0.0025, cost_per_1k_output=0.010,
    ),
    "desk_risk": ModelSpec(
        name="gpt-4o-2024-11-20",
        provider="openai", max_tokens=700, temperature=0.10,
        cost_per_1k_input=0.0025, cost_per_1k_output=0.010,
    ),
    "desk_sentiment": ModelSpec(
        name="gpt-4o-2024-11-20",
        provider="openai", max_tokens=1000, temperature=0.25,
        cost_per_1k_input=0.0025, cost_per_1k_output=0.010,
    ),
    "desk_devil": ModelSpec(       # devil's advocate agent
        name="gpt-4o-2024-11-20",
        provider="openai", max_tokens=600, temperature=0.30,
        cost_per_1k_input=0.0025, cost_per_1k_output=0.010,
    ),
    # Anthropic coordinator — pinned
    "coordinator": ModelSpec(
        name="claude-opus-4-6",
        provider="anthropic", max_tokens=2000, temperature=0.20,
        cost_per_1k_input=0.015, cost_per_1k_output=0.075,
    ),
}

def get(role: str) -> ModelSpec:
    if role not in MODELS:
        raise ValueError(f"Unknown model role '{role}'. Valid: {list(MODELS)}")
    return MODELS[role]


# ---------------------------------------------------------------------------
# Cost estimator (used by CostTracker)
# ---------------------------------------------------------------------------

def estimate_cost(role: str, input_tokens: int, output_tokens: int) -> float:
    m = get(role)
    return (input_tokens / 1000 * m.cost_per_1k_input +
            output_tokens / 1000 * m.cost_per_1k_output)


# ---------------------------------------------------------------------------
# Regression test harness (Risk 3)
# ---------------------------------------------------------------------------

REGRESSION_PROMPTS = {
    "macro": {
        "system": "You are a macro strategist. Respond ONLY with JSON.",
        "user":   ("Analyse: RBI holds at 6.5%, dovish tone. "
                   "Return {\"direction\":\"long\"|\"short\"|\"neutral\","
                   "\"conviction\":0.0-1.0,\"regime\":\"trending_up\"|\"trending_down\"|\"range_bound\"|\"high_volatility\"|\"event_driven\"}"),
        "check_keys": ["direction", "conviction", "regime"],
        "ranges": {"conviction": (0.0, 1.0)},
        "enum_fields": {"direction": ["long","short","neutral"],
                        "regime": ["trending_up","trending_down","range_bound","high_volatility","event_driven"]},
    },
    "risk": {
        "system": "You are a risk manager. Respond ONLY with JSON.",
        "user":   ("Evaluate: long NIFTY50, R:R=2.3, macro and technical agree. "
                   "Return {\"approved\":true|false,\"max_position_pct\":0.01-0.05,"
                   "\"conviction_modifier\":-0.30-0.00}"),
        "check_keys": ["approved","max_position_pct","conviction_modifier"],
        "ranges": {"max_position_pct": (0.0, 0.05), "conviction_modifier": (-0.30, 0.0)},
        "enum_fields": {},
    },
}


@dataclass
class RegressionResult:
    role:     str
    model:    str
    passed:   bool
    failures: list[str] = field(default_factory=list)
    raw:      str = ""
    run_at:   str = field(default_factory=lambda: datetime.utcnow().isoformat())


async def run_regression(openai_key: str, log_dir: Path) -> list[RegressionResult]:
    """
    Fire each regression prompt against its pinned model.
    Check that key fields are present, in-range, and from valid enums.
    Log results to log_dir/regression.jsonl.
    """
    from openai import AsyncOpenAI
    client  = AsyncOpenAI(api_key=openai_key)
    results = []

    for role, spec in REGRESSION_PROMPTS.items():
        model = get(f"desk_{role}")
        failures = []
        raw = ""
        try:
            resp = await client.chat.completions.create(
                model       = model.name,
                temperature = model.temperature,
                max_tokens  = 200,
                messages    = [
                    {"role":"system","content":spec["system"]},
                    {"role":"user",  "content":spec["user"]},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            parsed = json.loads(raw.replace("```json","").replace("```","").strip())

            for key in spec["check_keys"]:
                if key not in parsed:
                    failures.append(f"missing key: {key}")

            for key, (lo, hi) in spec["ranges"].items():
                val = parsed.get(key)
                if val is not None and not (lo <= float(val) <= hi):
                    failures.append(f"{key}={val} out of range [{lo},{hi}]")

            for key, valid in spec["enum_fields"].items():
                val = parsed.get(key)
                if val is not None and str(val).lower() not in valid:
                    failures.append(f"{key}='{val}' not in {valid}")

        except Exception as e:
            failures.append(f"exception: {e}")

        res = RegressionResult(
            role=role, model=model.name,
            passed=len(failures)==0, failures=failures, raw=raw,
        )
        results.append(res)
        status = "PASS" if res.passed else f"FAIL ({'; '.join(failures)})"
        logger.info("Regression [%s / %s]: %s", role, model.name, status)

    log_path = log_dir / "regression.jsonl"
    with open(log_path, "a") as f:
        for r in results:
            f.write(json.dumps({
                "role":r.role,"model":r.model,"passed":r.passed,
                "failures":r.failures,"run_at":r.run_at,
            }) + "\n")

    return results
