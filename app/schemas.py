"""Типизированные модели входных и выходных данных."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

NAMES_COUNT = 5
MAX_URL_LENGTH = 2048


# --- REST API ---------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Тело запроса POST /api/generate."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(..., description="Адрес страницы с описанием продукта")

    @field_validator("url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        # Здесь только синтаксическая проверка; проверка DNS и IP-адресов (SSRF)
        # выполняется в scraper непосредственно перед загрузкой.
        value = value.strip()
        if not value:
            raise ValueError("URL не может быть пустым")
        if len(value) > MAX_URL_LENGTH:
            raise ValueError(f"URL длиннее {MAX_URL_LENGTH} символов")
        try:
            parts = urlsplit(value)
        except ValueError as exc:
            raise ValueError("Некорректный URL") from exc
        if parts.scheme.lower() not in {"http", "https"}:
            raise ValueError("Поддерживаются только адреса http:// и https://")
        if not parts.hostname:
            raise ValueError("В URL не указан домен")
        return value


class NameSuggestion(BaseModel):
    """Один вариант названия."""

    name: str = Field(..., description="Название продукта")
    reason: str = Field(..., description="Краткое объяснение выбора")


class GenerateResponse(BaseModel):
    """Успешный ответ POST /api/generate."""

    source_url: str
    names: list[NameSuggestion] = Field(..., min_length=NAMES_COUNT, max_length=NAMES_COUNT)


class ErrorResponse(BaseModel):
    detail: str
    error_type: str


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str


# --- Внутренние модели агента ------------------------------------------------


class PageContent(BaseModel):
    """Содержимое страницы, извлечённое из HTML."""

    url: str
    title: str = ""
    description: str = ""
    text: str


class ProductContext(BaseModel):
    """Структурированный контекст продукта (результат шага анализа)."""

    model_config = ConfigDict(extra="forbid")

    product_name_in_source: str = Field(
        ..., description="Текущее название продукта в источнике или пустая строка"
    )
    purpose: str = Field(..., description="Назначение продукта")
    target_audience: str = Field(..., description="Целевая аудитория")
    key_benefits: list[str] = Field(..., description="Основные преимущества")
    key_features: list[str] = Field(..., description="Ключевые характеристики")
    brand_style: str = Field(..., description="Стиль и позиционирование бренда")


class NamesResult(BaseModel):
    """Ответ модели на шаге генерации названий."""

    model_config = ConfigDict(extra="forbid")

    names: list[NameSuggestion]


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Строит JSON Schema, совместимую со strict-режимом Structured Outputs OpenAI.

    Strict-режим требует, чтобы у каждого объекта были перечислены все свойства
    в ``required`` и было задано ``additionalProperties: false``.
    """
    schema = deepcopy(model.model_json_schema())

    def walk(node: dict[str, Any]) -> None:
        node.pop("title", None)
        if "$ref" in node:  # strict-режим не допускает соседних с $ref ключей
            node.pop("description", None)
        properties = node.get("properties")
        if isinstance(properties, dict):
            node["required"] = list(properties.keys())
            node["additionalProperties"] = False
            for child in properties.values():
                walk(child)
        if isinstance(node.get("items"), dict):
            walk(node["items"])
        for key in ("anyOf", "allOf"):
            for child in node.get(key, []):
                walk(child)
        for child in node.get("$defs", {}).values():
            walk(child)

    walk(schema)
    return schema
