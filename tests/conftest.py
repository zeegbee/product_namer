"""Общие фикстуры. Тесты не обращаются к сети и не вызывают OpenAI."""

from __future__ import annotations

import io
import socket
from collections.abc import Callable

import pytest
import requests
from requests.structures import CaseInsensitiveDict

from app.config import Settings

PUBLIC_IP = "93.184.215.14"


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        openai_api_key="sk-test",
        openai_model="gpt-test",
        request_timeout=5,
        openai_timeout=5,
        max_retries=3,
        max_page_size=50_000,
        max_text_length=5_000,
        log_file=str(tmp_path / "app.log"),
        verify_model_on_startup=False,
        retry_initial_delay=0,  # без реальных пауз в тестах
        retry_max_delay=0,
    )


@pytest.fixture
def fake_dns(monkeypatch) -> Callable[[dict[str, list[str]]], None]:
    """Подменяет DNS: имя -> список IP. Неизвестные имена разрешаются в публичный IP."""

    def install(mapping: dict[str, list[str]] | None = None) -> None:
        mapping = mapping or {}

        def getaddrinfo(host, port, *args, **kwargs):  # noqa: ANN001
            ips = mapping.get(host)
            if ips is None:
                try:
                    socket.inet_pton(socket.AF_INET6 if ":" in host else socket.AF_INET, host)
                    ips = [host]  # IP-литерал разрешается сам в себя
                except OSError:
                    ips = [PUBLIC_IP]
            if ips == []:
                raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
            return [
                (socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))
                for ip in ips
            ]

        monkeypatch.setattr("app.scraper.socket.getaddrinfo", getaddrinfo)

    install()
    return install


def make_response(
    url: str,
    status: int = 200,
    body: bytes | str = b"",
    headers: dict[str, str] | None = None,
) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.url = url
    raw = body.encode("utf-8") if isinstance(body, str) else body
    response.raw = io.BytesIO(raw)
    default_headers = {"Content-Type": "text/html; charset=utf-8"}
    response.headers = CaseInsensitiveDict({**default_headers, **(headers or {})})
    return response


PRODUCT_HTML = """
<!doctype html>
<html><head>
  <title>AquaPure Home — умный фильтр для воды</title>
  <meta name="description" content="Компактный фильтр с мониторингом качества воды через приложение.">
  <script>window.tracking = "secret";</script>
  <style>.x { color: red; }</style>
</head>
<body>
  <nav class="main-menu"><a href="/">Главная</a><a href="/shop">Магазин</a></nav>
  <header class="site-header">Логотип магазина</header>
  <div class="ad-banner">Купите рекламу прямо сейчас!</div>
  <article>
    <h1>AquaPure Home</h1>
    <p>AquaPure Home — компактный фильтр для воды, который устанавливается под мойку за десять минут.
       Он очищает воду от хлора, тяжёлых металлов и микропластика.</p>
    <p>Встроенные датчики отслеживают качество воды и состояние картриджа, а мобильное приложение
       напоминает о замене фильтра. Продукт создан для семей с детьми и людей, которые следят за здоровьем.</p>
    <noscript>Включите JavaScript</noscript>
    <svg><text>svg-text</text></svg>
  </article>
  <aside class="sidebar">Популярные статьи</aside>
  <footer>© 2026 Магазин. Все права защищены.</footer>
</body></html>
"""
