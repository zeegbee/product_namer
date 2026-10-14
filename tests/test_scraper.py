from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from app.errors import BlockedURLError, ContentExtractionError, FetchError, FetchTimeoutError, InvalidURLError
from app.scraper import PageFetcher, create_safe_session, extract_content, truncate_text, validate_url
from tests.conftest import PRODUCT_HTML, make_response

# --- Извлечение текста ---------------------------------------------------------


def test_extract_content_returns_title_description_and_main_text():
    content = extract_content(PRODUCT_HTML, "https://shop.example/aquapure", max_text_length=5000)

    assert content.title == "AquaPure Home — умный фильтр для воды"
    assert content.description == "Компактный фильтр с мониторингом качества воды через приложение."
    assert "очищает воду от хлора" in content.text
    assert "семей с детьми" in content.text


@pytest.mark.parametrize(
    "noise",
    ["window.tracking", "color: red", "Главная", "Логотип магазина", "Купите рекламу",
     "Включите JavaScript", "svg-text", "Популярные статьи", "Все права защищены"],
)  # fmt: skip
def test_extract_content_removes_noise(noise):
    content = extract_content(PRODUCT_HTML, "https://shop.example/", max_text_length=5000)
    assert noise not in content.text


def test_extract_content_collapses_whitespace():
    lines = "".join(f"Строка   \t  номер {i}\n\n\n\n" for i in range(30))
    html = f"<html><body><main><p>{lines}</p></main></body></html>"
    content = extract_content(html, "https://x.example/", max_text_length=5000)
    assert "   " not in content.text
    assert "\n\n" not in content.text


def test_extract_content_keeps_inline_links_in_sentence():
    html = "<html><body><p>" + "Фильтр <a href='#'>очищает</a> <b>воду</b> за минуту. " * 10 + "</p></body></html>"
    content = extract_content(html, "https://x.example/", max_text_length=5000)
    assert content.text.startswith("Фильтр очищает воду за минуту.")


def test_extract_content_uses_body_when_no_article():
    html = "<html><body><div><p>" + "Описание продукта и его возможностей. " * 10 + "</p></div></body></html>"
    content = extract_content(html, "https://x.example/", max_text_length=5000)
    assert content.text.startswith("Описание продукта")


def test_extract_content_detects_charset_from_meta_for_bytes():
    html = (
        '<html><head><meta charset="windows-1251"></head><body><p>'
        + "Умная кофеварка готовит эспрессо и капучино одним нажатием. " * 5
        + "</p></body></html>"
    ).encode("cp1251")
    content = extract_content(html, "https://x.example/", max_text_length=5000)
    assert "Умная кофеварка" in content.text


@pytest.mark.parametrize(
    "html",
    [
        "",
        "<html><body></body></html>",
        "<html><head><script>var a = 1;</script></head><body><div id='root'></div></body></html>",
        "<html><body><p>Слишком коротко.</p></body></html>",
    ],
)
def test_extract_content_rejects_empty_page(html):
    with pytest.raises(ContentExtractionError):
        extract_content(html, "https://x.example/", max_text_length=5000)


def test_extract_content_limits_text_length():
    html = "<html><body><p>" + "длинный текст " * 2000 + "</p></body></html>"
    content = extract_content(html, "https://x.example/", max_text_length=1000)
    assert len(content.text) <= 1002


def test_truncate_text_cuts_on_word_boundary():
    assert truncate_text("один два три четыре", 14) == "один два три …"
    assert truncate_text("коротко", 100) == "коротко"


# --- Проверка URL и защита от SSRF ------------------------------------------------


@pytest.mark.parametrize(
    "url",
    ["", "not a url", "ftp://example.com/file", "file:///etc/passwd", "javascript:alert(1)",
     "https://", "http://user:pass@example.com/", "https://example.com:99999/"],
)  # fmt: skip
def test_validate_url_rejects_invalid_urls(url, fake_dns):
    with pytest.raises(InvalidURLError):
        validate_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/",
        "http://LOCALHOST./admin",
        "http://app.localhost/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://127.0.0.1/",
        "http://10.0.0.5/",
        "http://172.16.3.4/",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://100.100.100.200/",
        "http://0.0.0.0/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://[fd00:ec2::254]/",
        "http://example.com:22/",
    ],
)
def test_validate_url_blocks_internal_addresses(url, fake_dns):
    with pytest.raises(BlockedURLError):
        validate_url(url)


def test_validate_url_blocks_hostname_resolving_to_private_ip(fake_dns):
    fake_dns({"evil.example": ["10.1.2.3"]})
    with pytest.raises(BlockedURLError):
        validate_url("https://evil.example/")


def test_validate_url_blocks_mixed_public_and_private_resolution(fake_dns):
    fake_dns({"mixed.example": ["93.184.215.14", "127.0.0.1"]})
    with pytest.raises(BlockedURLError):
        validate_url("https://mixed.example/")


def test_validate_url_accepts_public_host(fake_dns):
    assert validate_url(" https://shop.example/page?id=1 ") == "https://shop.example/page?id=1"


def test_validate_url_reports_dns_failure(fake_dns):
    fake_dns({"missing.example": []})
    with pytest.raises(FetchError, match="DNS"):
        validate_url("https://missing.example/")


def test_safe_session_blocks_private_ip_at_connection_level(fake_dns):
    """Даже если предварительную проверку обойти, сокет не откроется на внутренний IP
    (защита от DNS rebinding: имя разрешается заново при подключении)."""
    fake_dns({"rebind.example": ["127.0.0.1"]})
    with create_safe_session() as session, pytest.raises(BlockedURLError):
        session.get("http://rebind.example/", timeout=2)


# --- Загрузка страницы -----------------------------------------------------------


def fetcher_with(settings, *responses_or_errors) -> tuple[PageFetcher, MagicMock]:
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = list(responses_or_errors)
    return PageFetcher(settings, session=session), session


def test_fetch_returns_page(settings, fake_dns):
    fetcher, session = fetcher_with(settings, make_response("https://shop.example/", body=PRODUCT_HTML))
    page = fetcher.fetch("https://shop.example/")
    assert b"AquaPure" in page.body
    assert page.encoding == "utf-8"
    _, kwargs = session.get.call_args
    assert kwargs["allow_redirects"] is False
    assert kwargs["stream"] is True


def test_fetch_follows_safe_redirect(settings, fake_dns):
    fetcher, session = fetcher_with(
        settings,
        make_response("https://shop.example/old", status=301, headers={"Location": "/new"}),
        make_response("https://shop.example/new", body=PRODUCT_HTML),
    )
    page = fetcher.fetch("https://shop.example/old")
    assert page.url == "https://shop.example/new"
    assert session.get.call_args_list[1].args[0] == "https://shop.example/new"


@pytest.mark.parametrize(
    "location",
    ["http://127.0.0.1/admin", "http://169.254.169.254/latest/meta-data/", "http://localhost:8000/health"],
)
def test_fetch_blocks_redirect_to_internal_address(settings, fake_dns, location):
    fetcher, session = fetcher_with(
        settings, make_response("https://shop.example/", status=302, headers={"Location": location})
    )
    with pytest.raises(BlockedURLError):
        fetcher.fetch("https://shop.example/")
    assert session.get.call_count == 1  # внутренний адрес так и не запрошен


def test_fetch_limits_redirect_count(settings, fake_dns):
    loop = [make_response("https://shop.example/a", status=302, headers={"Location": "/a"}) for _ in range(10)]
    fetcher, _ = fetcher_with(settings, *loop)
    with pytest.raises(FetchError, match="перенаправлений"):
        fetcher.fetch("https://shop.example/a")


def test_fetch_rejects_too_large_page_by_header(settings, fake_dns):
    fetcher, _ = fetcher_with(
        settings, make_response("https://shop.example/", body="x", headers={"Content-Length": "10000000"})
    )
    with pytest.raises(FetchError, match="слишком большая"):
        fetcher.fetch("https://shop.example/")


def test_fetch_rejects_too_large_page_while_streaming(settings, fake_dns):
    fetcher, _ = fetcher_with(settings, make_response("https://shop.example/", body=b"a" * 200_000))
    with pytest.raises(FetchError, match="слишком большая"):
        fetcher.fetch("https://shop.example/")


def test_fetch_rejects_non_html(settings, fake_dns):
    fetcher, _ = fetcher_with(
        settings, make_response("https://shop.example/f.pdf", body=b"%PDF", headers={"Content-Type": "application/pdf"})
    )
    with pytest.raises(FetchError, match="HTML"):
        fetcher.fetch("https://shop.example/f.pdf")


def test_fetch_does_not_retry_client_errors(settings, fake_dns):
    fetcher, session = fetcher_with(settings, make_response("https://shop.example/", status=404))
    with pytest.raises(FetchError, match="404"):
        fetcher.fetch("https://shop.example/")
    assert session.get.call_count == 1


def test_fetch_retries_network_errors_then_succeeds(settings, fake_dns):
    fetcher, session = fetcher_with(
        settings,
        requests.exceptions.ConnectionError("reset"),
        make_response("https://shop.example/", status=503),
        make_response("https://shop.example/", body=PRODUCT_HTML),
    )
    page = fetcher.fetch("https://shop.example/")
    assert b"AquaPure" in page.body
    assert session.get.call_count == 3


def test_fetch_network_error_after_retries(settings, fake_dns):
    fetcher, session = fetcher_with(settings, *[requests.exceptions.ConnectionError("down")] * 3)
    with pytest.raises(FetchError) as exc_info:
        fetcher.fetch("https://shop.example/")
    assert exc_info.value.status_code == 502
    assert session.get.call_count == settings.max_retries


def test_fetch_timeout_maps_to_504(settings, fake_dns):
    fetcher, session = fetcher_with(settings, *[requests.exceptions.ReadTimeout("slow")] * 3)
    with pytest.raises(FetchTimeoutError) as exc_info:
        fetcher.fetch("https://shop.example/")
    assert exc_info.value.status_code == 504
    assert session.get.call_count == 3


def test_fetch_does_not_retry_ssl_errors(settings, fake_dns):
    fetcher, session = fetcher_with(settings, requests.exceptions.SSLError("bad cert"))
    with pytest.raises(FetchError, match="SSL"):
        fetcher.fetch("https://shop.example/")
    assert session.get.call_count == 1
