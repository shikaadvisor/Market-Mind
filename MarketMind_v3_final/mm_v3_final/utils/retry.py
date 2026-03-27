"""
utils/retry.py
==============
Risk 4 fix: Exponential backoff with jitter for all LLM calls.
            Model fallback: if primary model fails 3 times, drop to fallback.
            Circuit breaker: after 10 consecutive failures, pause for 60s.

Usage:
    from utils.retry import with_retry
    result = await with_retry(client.chat.completions.create, **kwargs)
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger("marketmind.retry")

# Model fallback chain: if primary fails, try these in order
MODEL_FALLBACK = {
    "gpt-4o-2024-11-20":      "gpt-4o-mini-2024-07-18",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini-2024-07-18",  # no fallback for mini
}


class CircuitBreaker:
    """Per-endpoint circuit breaker."""
    def __init__(self, threshold: int = 10, cooldown_sec: int = 60):
        self.threshold   = threshold
        self.cooldown    = cooldown_sec
        self._failures   = 0
        self._open_until = 0.0

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.threshold:
            self._open_until = time.monotonic() + self.cooldown
            logger.warning("Circuit breaker OPEN for %ds after %d failures",
                           self.cooldown, self._failures)

    def record_success(self) -> None:
        self._failures = 0

    def is_open(self) -> bool:
        if time.monotonic() < self._open_until:
            return True
        if self._open_until > 0 and time.monotonic() >= self._open_until:
            logger.info("Circuit breaker CLOSED — resuming")
            self._open_until = 0.0
        return False


_breakers: dict[str, CircuitBreaker] = {}

def _get_breaker(endpoint: str) -> CircuitBreaker:
    if endpoint not in _breakers:
        _breakers[endpoint] = CircuitBreaker()
    return _breakers[endpoint]


async def with_retry(
    fn:          Callable,
    *args,
    max_retries: int   = 3,
    base_delay:  float = 1.0,
    endpoint:    str   = "openai",
    **kwargs,
) -> Any:
    """
    Call fn(*args, **kwargs) with exponential backoff + jitter.
    Respects circuit breaker state for the given endpoint.
    On model errors, substitutes fallback model if available.
    """
    breaker = _get_breaker(endpoint)
    if breaker.is_open():
        raise RuntimeError(f"Circuit breaker open for {endpoint} — skipping call")

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            result = await fn(*args, **kwargs)
            breaker.record_success()
            return result
        except Exception as exc:
            last_exc = exc
            exc_str  = str(exc).lower()

            # Rate limit: back off longer
            if "rate limit" in exc_str or "429" in exc_str:
                delay = base_delay * (4 ** attempt) + random.uniform(0, 2)
                logger.warning("Rate limit hit (attempt %d/%d), backing off %.1fs",
                               attempt+1, max_retries+1, delay)
                await asyncio.sleep(delay)
                continue

            # Model not found: try fallback immediately
            if "model_not_found" in exc_str or "does not exist" in exc_str:
                current_model = kwargs.get("model", "")
                fallback      = MODEL_FALLBACK.get(current_model)
                if fallback and fallback != current_model:
                    logger.warning("Model '%s' unavailable, falling back to '%s'",
                                   current_model, fallback)
                    kwargs = {**kwargs, "model": fallback}
                    continue

            # Other transient errors: standard backoff
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning("LLM call failed (attempt %d/%d): %s — retry in %.1fs",
                               attempt+1, max_retries+1, exc, delay)
                await asyncio.sleep(delay)
            else:
                breaker.record_failure()
                logger.error("All %d retries exhausted: %s", max_retries+1, exc)

    raise last_exc  # type: ignore
