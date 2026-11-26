from __future__ import annotations

import contextlib
import os
import re
import signal
import time
from typing import Callable


class GracefulShutdown(BaseException):
    pass


class FatalConfigError(Exception):
    pass


class DailyQuotaExceededError(Exception):
    pass


class RateLimitError(Exception):
    pass


class OfferEvaluationTimeoutError(Exception):
    pass


class TransientEvaluationError(Exception):
    pass


class EvaluationFailedError(Exception):
    pass


class RetryableResponseError(Exception):
    pass


_STOP_REQUESTED = False
_SIGNAL_COUNT = 0
_CRITICAL_DEPTH = 0


def log(message: str) -> None:
    print(message, flush=True)


def should_stop() -> bool:
    return _STOP_REQUESTED


def reset_stop_state() -> None:
    global _STOP_REQUESTED, _SIGNAL_COUNT, _CRITICAL_DEPTH
    _STOP_REQUESTED = False
    _SIGNAL_COUNT = 0
    _CRITICAL_DEPTH = 0


@contextlib.contextmanager
def critical_section():
    global _CRITICAL_DEPTH
    _CRITICAL_DEPTH += 1
    try:
        yield
    finally:
        _CRITICAL_DEPTH -= 1


def _handle_stop_signal(signum: int, _frame) -> None:
    global _STOP_REQUESTED, _SIGNAL_COUNT
    name = signal.Signals(signum).name if signum in signal.Signals._value2member_map_ else str(signum)
    _SIGNAL_COUNT += 1
    _STOP_REQUESTED = True
    if _SIGNAL_COUNT == 1:
        log(f"\n[STOP] {name} reçu — arrêt propre à la prochaine étape sûre.")
        return
    if _CRITICAL_DEPTH > 0:
        log(f"\n[STOP] {name} reçu pendant une sauvegarde — patientez.")
        return
    raise GracefulShutdown(name)


def install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, _handle_stop_signal)
    signal.signal(signal.SIGTERM, _handle_stop_signal)


def interruptible_sleep(seconds: float) -> None:
    deadline = time.monotonic() + max(0.0, seconds)
    while True:
        if should_stop():
            raise GracefulShutdown("STOP")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.25, remaining))


def env_str(var_name: str, default: str = "") -> str:
    val = os.getenv(var_name)
    return val.strip() if isinstance(val, str) and val.strip() else default


_DAILY_QUOTA_RX = re.compile(
    r"(?:"
    r"\b(?:daily|per[\s_-]?day|requests[\s_-]?per[\s_-]?day|tokens[\s_-]?per[\s_-]?day)\b"
    r".{0,80}\b(?:quota|limit|exceeded|exhausted|cap)\b"
    r"|"
    r"\b(?:quota|limit|exceeded|exhausted)\b"
    r".{0,80}\b(?:daily|per[\s_-]?day|requests[\s_-]?per[\s_-]?day|tokens[\s_-]?per[\s_-]?day)\b"
    r"|"
    r"(?:requests|tokens|free_tier)[\s_-]*per[\s_-]*day"
    r")",
    re.IGNORECASE,
)


def is_daily_quota_error(message: str) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    low = text.lower()
    if "503" in low or "high demand" in low or "currently experiencing high demand" in low:
        return False
    return bool(_DAILY_QUOTA_RX.search(low))


def parse_retry_delay_s(message: str) -> float | None:
    low = (message or "").lower()
    for pattern in (r"retry in ([\d.]+)\s*s", r"retrydelay['\"]?\s*:\s*['\"]?([\d.]+)s"):
        if m := re.search(pattern, low):
            return float(m.group(1))
    return None


def status_code(exc: Exception) -> int | None:
    if isinstance(getattr(exc, "code", None), int):
        return exc.code
    match = re.search(r"\b(4\d\d|5\d\d)\b", str(exc))
    return int(match.group(1)) if match else None


def is_transient_network_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    return any(
        t in name
        for t in ("timeout", "connecterror", "connectionerror", "remoteprotocol", "readerror", "networkerror")
    )


class Throttle:
    def __init__(
        self,
        min_interval_s: float,
        clock: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] | None = None,
    ):
        self.min_interval_s = max(0.0, float(min_interval_s))
        self._clock = clock
        self._sleep = sleep_fn or interruptible_sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self.min_interval_s > 0 and self._last is not None:
            delay = self._last + self.min_interval_s - self._clock()
            if delay > 0:
                self._sleep(delay)
        self._last = self._clock()
