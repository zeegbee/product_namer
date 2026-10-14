"""Взаимодействие с OpenAI API.

Модуль ничего не знает о нейминге: он отправляет системный и пользовательский
промпты, запрашивает ответ в формате JSON Schema (Structured Outputs) и
возвращает проверенную pydantic-модель.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, TypeVar

import openai
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.errors import ConfigError, LLMError, LLMResponseError, LLMTimeoutError
from app.schemas import strict_json_schema
from app.utils import call_with_retry

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)


def is_transient_openai_error(exc: BaseException) -> bool:
    """Временные ошибки, которые имеет смысл повторить: сеть, тайм-аут, 429, 5xx.

    Исчерпанная квота (429 с кодом ``insufficient_quota``) — постоянная ошибка.
    """
    if isinstance(exc, openai.APIConnectionError):  # включает APITimeoutError
        return True
    if isinstance(exc, openai.RateLimitError):
        return getattr(exc, "code", None) != "insufficient_quota"
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500 or exc.status_code == 408
    return False


def _to_app_error(exc: Exception, model: str) -> LLMError:
    """Преобразует исключение SDK в ошибку приложения с понятным сообщением."""
    if isinstance(exc, openai.APITimeoutError):
        return LLMTimeoutError()
    if isinstance(exc, openai.APIConnectionError):
        return LLMError("Не удалось подключиться к OpenAI API. Попробуйте позже.")
    if isinstance(exc, openai.AuthenticationError):
        return LLMError("Ошибка авторизации в OpenAI API: проверьте OPENAI_API_KEY.")
    if isinstance(exc, openai.PermissionDeniedError):
        return LLMError(f"Нет доступа к модели {model}: проверьте права аккаунта OpenAI.")
    if isinstance(exc, openai.NotFoundError):
        return LLMError(f"Модель {model} не найдена или недоступна: проверьте OPENAI_MODEL.")
    if isinstance(exc, openai.RateLimitError):
        if getattr(exc, "code", None) == "insufficient_quota":
            return LLMError("Исчерпана квота OpenAI API: проверьте баланс аккаунта.")
        return LLMError("Превышен лимит запросов к OpenAI API. Попробуйте через минуту.")
    if isinstance(exc, openai.BadRequestError):
        return LLMError("OpenAI API отклонил запрос. Подробности — в логах сервера.")
    return LLMError()


def _is_response_format_unsupported(exc: openai.BadRequestError) -> bool:
    text = f"{exc.message} {getattr(exc, 'param', '') or ''}".lower()
    return "response_format" in text or "json_schema" in text


class OpenAIClient:
    """Тонкая обёртка над OpenAI SDK с повторами и проверкой структуры ответа."""

    def __init__(self, settings: Settings, client: openai.OpenAI | None = None) -> None:
        self._settings = settings
        # Встроенные повторы SDK отключены — повторами управляет tenacity,
        # чтобы число платных вызовов было предсказуемым.
        self._client = client or openai.OpenAI(
            api_key=settings.openai_api_key, timeout=settings.openai_timeout, max_retries=0
        )
        self._structured_outputs = True

    @property
    def model(self) -> str:
        return self._settings.openai_model

    def verify_model(self) -> None:
        """Проверяет, что модель из OPENAI_MODEL доступна аккаунту.

        Ошибки авторизации и отсутствие модели — фатальны (``ConfigError``).
        Сетевые сбои только логируются: приложение может стартовать без сети.
        """
        try:
            self._client.models.retrieve(self.model)
        except (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError) as exc:
            raise ConfigError(_to_app_error(exc, self.model).user_message) from exc
        except openai.OpenAIError as exc:
            logger.warning(
                "stage=startup event=model_check_skipped model=%s error_type=%s", self.model, type(exc).__name__
            )
            return
        logger.info("stage=startup event=model_available model=%s", self.model)

    def complete_structured(
        self, *, stage: str, system_prompt: str, user_prompt: str, response_model: type[ModelT]
    ) -> ModelT:
        """Запрашивает у модели JSON, соответствующий ``response_model``, и проверяет его."""
        started = time.monotonic()
        try:
            content, attempts = call_with_retry(
                lambda: self._request(system_prompt, user_prompt, response_model),
                stage=stage,
                is_transient=is_transient_openai_error,
                max_attempts=self._settings.max_retries,
                initial_delay=self._settings.retry_initial_delay,
                max_delay=self._settings.retry_max_delay,
            )
        except openai.OpenAIError as exc:
            logger.error(
                "stage=%s event=openai_failed model=%s error_type=%s status=%s attempts=%s duration_ms=%d detail=%s",
                stage,
                self.model,
                type(exc).__name__,
                getattr(exc, "status_code", "-"),
                getattr(exc, "attempts", "?"),
                (time.monotonic() - started) * 1000,
                getattr(exc, "message", str(exc))[:300],
            )
            raise _to_app_error(exc, self.model) from exc

        result = self._parse(content, response_model, stage)
        logger.info(
            "stage=%s event=openai_success model=%s attempts=%d duration_ms=%d",
            stage,
            self.model,
            attempts,
            (time.monotonic() - started) * 1000,
        )
        return result

    def _request(self, system_prompt: str, user_prompt: str, response_model: type[BaseModel]) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if self._structured_outputs:
            response_format: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": strict_json_schema(response_model),
                },
            }
            try:
                return self._create(messages, response_format)
            except openai.BadRequestError as exc:
                if not _is_response_format_unsupported(exc):
                    raise
                # Модель не поддерживает Structured Outputs — переходим на JSON mode,
                # схему передаём в промпте и проверяем ответ на стороне Python.
                logger.warning("event=structured_outputs_unsupported model=%s fallback=json_object", self.model)
                self._structured_outputs = False

        schema_hint = json.dumps(strict_json_schema(response_model), ensure_ascii=False)
        messages[0] = {
            "role": "system",
            "content": f"{system_prompt}\n\nОтвет — только JSON-объект по этой JSON Schema:\n{schema_hint}",
        }
        return self._create(messages, {"type": "json_object"})

    def _create(self, messages: list[dict[str, str]], response_format: dict[str, Any]) -> str:
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            response_format=response_format,  # type: ignore[arg-type]
        )
        if not completion.choices:
            raise LLMResponseError()
        choice = completion.choices[0]
        if getattr(choice.message, "refusal", None):
            logger.warning("event=openai_refusal model=%s", self.model)
            raise LLMResponseError("Модель отказалась обрабатывать эту страницу.")
        if choice.finish_reason == "length":
            raise LLMResponseError("Ответ модели оборвался из-за ограничения длины. Попробуйте ещё раз.")
        if not choice.message.content:
            raise LLMResponseError()
        return choice.message.content

    @staticmethod
    def _parse(content: str, response_model: type[ModelT], stage: str) -> ModelT:
        try:
            return response_model.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.error(
                "stage=%s event=invalid_llm_json error_type=%s length=%d", stage, type(exc).__name__, len(content)
            )
            raise LLMResponseError() from exc
