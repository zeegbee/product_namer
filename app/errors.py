"""Исключения приложения.

Каждое исключение несёт сообщение для пользователя (на русском) и HTTP-статус,
который вернёт API. Подробности для диагностики пишутся только в серверные логи.
"""

from __future__ import annotations


class ConfigError(Exception):
    """Ошибка конфигурации, обнаруженная при запуске."""


class ProductNamerError(Exception):
    """Базовая ошибка обработки запроса."""

    status_code: int = 500
    error_type: str = "internal_error"
    default_message: str = "Внутренняя ошибка приложения."

    def __init__(self, user_message: str | None = None) -> None:
        self.user_message = user_message or self.default_message
        super().__init__(self.user_message)


class InvalidURLError(ProductNamerError):
    status_code = 422
    error_type = "invalid_url"
    default_message = "Некорректный URL. Укажите адрес вида https://example.com/page."


class BlockedURLError(InvalidURLError):
    """URL ведёт на запрещённый (внутренний или служебный) адрес."""

    error_type = "blocked_url"
    default_message = "Этот адрес недоступен для загрузки: разрешены только публичные сайты."


class ContentExtractionError(ProductNamerError):
    status_code = 422
    error_type = "insufficient_content"
    default_message = (
        "На странице недостаточно текста для анализа. "
        "Возможно, содержимое загружается через JavaScript — попробуйте другую страницу."
    )


class FetchError(ProductNamerError):
    status_code = 502
    error_type = "fetch_failed"
    default_message = "Не удалось загрузить страницу."


class FetchTimeoutError(FetchError):
    status_code = 504
    error_type = "fetch_timeout"
    default_message = "Сайт не ответил вовремя. Попробуйте позже."


class LLMError(ProductNamerError):
    status_code = 502
    error_type = "llm_error"
    default_message = "Сервис генерации названий временно недоступен. Попробуйте позже."


class LLMTimeoutError(LLMError):
    status_code = 504
    error_type = "llm_timeout"
    default_message = "Превышено время ожидания ответа модели. Попробуйте позже."


class LLMResponseError(LLMError):
    error_type = "llm_invalid_response"
    default_message = "Модель вернула некорректный результат. Попробуйте ещё раз."
