"""Общие вспомогательные функции: повторные попытки и безопасное логирование URL."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar
from urllib.parse import urlsplit, urlunsplit

from tenacity import RetryCallState, Retrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

T = TypeVar("T")

logger = logging.getLogger(__name__)


def redact_url(url: str) -> str:
    """Возвращает URL без логина/пароля, query-параметров и фрагмента — для логов."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if ":" in host:  # IPv6
            host = f"[{host}]"
        netloc = f"{host}:{parts.port}" if parts.port else host
        query = "<redacted>" if parts.query else ""
        return urlunsplit((parts.scheme, netloc, parts.path, query, ""))
    except ValueError:
        return "<invalid-url>"


def call_with_retry(
    func: Callable[[], T],
    *,
    stage: str,
    is_transient: Callable[[BaseException], bool],
    max_attempts: int,
    initial_delay: float,
    max_delay: float,
) -> tuple[T, int]:
    """Вызывает ``func`` с повторами при временных ошибках.

    Используется экспоненциальная задержка со случайным разбросом (jitter).
    Постоянные ошибки (``is_transient`` вернул False) пробрасываются сразу.
    Возвращает результат и число выполненных попыток.
    """

    def log_retry(state: RetryCallState) -> None:
        exc = state.outcome.exception() if state.outcome else None
        delay = state.next_action.sleep if state.next_action else 0
        logger.warning(
            "stage=%s event=retry attempt=%d/%d error_type=%s delay=%.2fs",
            stage,
            state.attempt_number,
            max_attempts,
            type(exc).__name__,
            delay,
        )

    retrying = Retrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(multiplier=initial_delay, max=max_delay, jitter=initial_delay),
        retry=retry_if_exception(is_transient),
        before_sleep=log_retry,
        reraise=True,
        sleep=time.sleep,
    )
    attempts = 0

    def counted() -> T:
        nonlocal attempts
        attempts += 1
        return func()

    try:
        return retrying(counted), attempts
    except BaseException as exc:
        exc.attempts = attempts  # type: ignore[attr-defined]  # для логирования на верхнем уровне
        raise
