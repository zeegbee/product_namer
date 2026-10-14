"""Загрузка и проверка конфигурации, настройка логирования."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

from app.errors import ConfigError

_PLACEHOLDER_KEYS = {"", "your_openai_api_key", "sk-...", "changeme"}
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


@dataclass(frozen=True)
class Settings:
    """Неизменяемые настройки приложения."""

    openai_api_key: str
    openai_model: str = "gpt-4.1-mini"
    request_timeout: float = 15.0
    openai_timeout: float = 60.0
    max_retries: int = 3
    max_page_size: int = 2_000_000
    max_text_length: int = 20_000
    max_redirects: int = 5
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    verify_model_on_startup: bool = True
    # Базовая и максимальная задержка экспоненциального backoff (секунды).
    retry_initial_delay: float = 1.0
    retry_max_delay: float = 10.0

    def __repr__(self) -> str:  # не допускаем попадания ключа в логи и трейсбеки
        return f"Settings(openai_model={self.openai_model!r}, log_level={self.log_level!r})"


def _read_number(name: str, default: float, cast: type, minimum: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return cast(default)
    try:
        value = cast(raw)
    except ValueError as exc:
        raise ConfigError(f"Переменная {name} должна быть числом, получено: {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"Переменная {name} должна быть не меньше {minimum}, получено: {value}")
    return value


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"Переменная {name} должна быть true или false, получено: {raw!r}")


def load_settings(env_file: str | None = ".env") -> Settings:
    """Читает настройки из окружения (и файла .env, если он есть) и проверяет их.

    Переменные окружения имеют приоритет над значениями из .env.
    """
    if env_file:
        load_dotenv(env_file, override=False)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key in _PLACEHOLDER_KEYS:
        raise ConfigError(
            "Не задан OPENAI_API_KEY. Скопируйте .env.example в .env и укажите реальный ключ OpenAI."
        )

    model = os.getenv("OPENAI_MODEL", "").strip() or Settings.openai_model
    log_level = (os.getenv("LOG_LEVEL", "").strip() or Settings.log_level).upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigError(f"Недопустимое значение LOG_LEVEL: {log_level!r}")

    return Settings(
        openai_api_key=api_key,
        openai_model=model,
        request_timeout=_read_number("REQUEST_TIMEOUT", Settings.request_timeout, float, 1),
        openai_timeout=_read_number("OPENAI_TIMEOUT", Settings.openai_timeout, float, 1),
        max_retries=_read_number("MAX_RETRIES", Settings.max_retries, int, 1),
        max_page_size=_read_number("MAX_PAGE_SIZE", Settings.max_page_size, int, 10_000),
        max_text_length=_read_number("MAX_TEXT_LENGTH", Settings.max_text_length, int, 500),
        max_redirects=_read_number("MAX_REDIRECTS", Settings.max_redirects, int, 0),
        log_level=log_level,
        log_file=os.getenv("LOG_FILE", "").strip() or Settings.log_file,
        verify_model_on_startup=_read_bool("VERIFY_MODEL_ON_STARTUP", Settings.verify_model_on_startup),
    )


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Настраивает вывод логов в консоль и в файл с ротацией.

    Повторный вызов безопасен: ранее добавленные обработчики заменяются.
    """
    formatter = logging.Formatter(_LOG_FORMAT)
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        if getattr(handler, "_product_namer", False):
            root.removeHandler(handler)
            handler.close()

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
        )
    for handler in handlers:
        handler.setFormatter(formatter)
        handler._product_namer = True  # type: ignore[attr-defined]
        root.addHandler(handler)

    # Библиотечные логгеры HTTP-клиентов могут писать заголовки запросов — приглушаем их.
    for noisy in ("httpx", "httpx2", "httpcore", "httpcore2", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
