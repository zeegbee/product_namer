"""Безопасная загрузка веб-страниц и извлечение основного текста.

Защита от SSRF устроена в два уровня:

1. ``validate_url`` — до запроса проверяет схему, порт, имя хоста и все IP-адреса,
   в которые разрешается имя. Вызывается для исходного URL и для каждого редиректа
   (редиректы обрабатываются вручную, ``allow_redirects=False``).
2. ``_SafeHTTPConnection`` — на уровне сокета urllib3 повторно разрешает имя и
   подключается только к проверенному публичному IP. Это закрывает атаку DNS
   rebinding, при которой имя между проверкой и подключением начинает указывать
   на внутренний адрес. Сертификат HTTPS при этом проверяется по исходному
   имени хоста, а не по IP.

Переменные окружения прокси игнорируются (``trust_env=False``), чтобы запрос
нельзя было увести через прокси во внутреннюю сеть.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
import time
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from requests.utils import get_encoding_from_headers
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError
from urllib3.util.connection import create_connection

from app.config import Settings
from app.errors import BlockedURLError, ContentExtractionError, FetchError, FetchTimeoutError, InvalidURLError
from app.schemas import PageContent
from app.utils import call_with_retry, redact_url

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES = frozenset({"http", "https"})
ALLOWED_PORTS = frozenset({80, 443, 8080, 8443})
ALLOWED_CONTENT_TYPES = frozenset({"text/html", "application/xhtml+xml", "text/plain"})
BLOCKED_HOSTNAMES = frozenset({"localhost", "metadata", "metadata.google.internal", "instance-data"})
BLOCKED_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".localdomain", ".home.arpa", ".lan", ".intranet")
MIN_TEXT_LENGTH = 200
USER_AGENT = "Mozilla/5.0 (compatible; ProductNamerBot/1.0; +https://github.com/)"


# --- Проверка адресов ---------------------------------------------------------


def is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True, если адрес публичный и маршрутизируемый в интернете.

    ``is_global`` отсекает частные сети, loopback, link-local (включая
    169.254.169.254 — metadata облаков), CGNAT 100.64.0.0/10 (metadata Alibaba Cloud),
    зарезервированные и документационные диапазоны.
    """
    if isinstance(ip, ipaddress.IPv6Address):
        # IPv4-mapped (::ffff:127.0.0.1), 6to4 и Teredo могут скрывать внутренний IPv4.
        embedded = ip.ipv4_mapped or ip.sixtofour or (ip.teredo[1] if ip.teredo else None)
        if embedded is not None and not is_public_ip(embedded):
            return False
    return ip.is_global and not ip.is_multicast and not ip.is_reserved


def _check_hostname(hostname: str) -> str:
    host = hostname.strip().rstrip(".").lower()
    if not host or len(host) > 253:
        raise InvalidURLError("В URL указан некорректный домен.")
    if host in BLOCKED_HOSTNAMES or host.endswith(BLOCKED_HOST_SUFFIXES):
        raise BlockedURLError()
    return host


def resolve_public_addresses(hostname: str, port: int) -> list[str]:
    """Разрешает имя хоста и возвращает IP-адреса, если ВСЕ они публичные.

    Если хотя бы один адрес внутренний, запрос блокируется целиком: иначе
    злоумышленник мог бы вернуть смесь публичного и внутреннего адресов.
    """
    host = _check_hostname(hostname)
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError) as exc:
        raise FetchError("Не удалось найти сайт по указанному адресу (ошибка DNS).") from exc

    addresses: list[str] = []
    for info in infos:
        ip_text = str(info[4][0]).split("%", 1)[0]  # убираем zone id у IPv6
        ip = ipaddress.ip_address(ip_text)
        if not is_public_ip(ip):
            logger.warning("stage=ssrf_check event=blocked host=%s ip=%s", host, ip)
            raise BlockedURLError()
        if ip_text not in addresses:
            addresses.append(ip_text)
    if not addresses:
        raise FetchError("Не удалось найти сайт по указанному адресу (ошибка DNS).")
    return addresses


def validate_url(url: str) -> str:
    """Проверяет URL перед загрузкой (схема, учётные данные, порт, хост, DNS)."""
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError as exc:
        raise InvalidURLError() from exc
    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise InvalidURLError("Поддерживаются только адреса http:// и https://.")
    if not parts.hostname:
        raise InvalidURLError("В URL не указан домен.")
    if parts.username or parts.password:
        raise InvalidURLError("URL не должен содержать логин и пароль.")
    effective_port = port or (443 if parts.scheme.lower() == "https" else 80)
    if effective_port not in ALLOWED_PORTS:
        raise BlockedURLError("Разрешены только стандартные веб-порты (80, 443, 8080, 8443).")
    resolve_public_addresses(parts.hostname, effective_port)
    return url.strip()


# --- Соединения urllib3 с проверкой IP ----------------------------------------


class _SafeConnectionMixin:
    """Подключается только к проверенным публичным IP-адресам."""

    def _new_conn(self):  # type: ignore[no-untyped-def]
        addresses = resolve_public_addresses(self._dns_host, self.port)  # type: ignore[attr-defined]
        last_error: OSError | None = None
        timeout_error: ConnectTimeoutError | None = None
        for address in addresses:
            try:
                return create_connection(
                    (address, self.port),  # type: ignore[attr-defined]
                    self.timeout,  # type: ignore[attr-defined]
                    source_address=self.source_address,  # type: ignore[attr-defined]
                    socket_options=self.socket_options,  # type: ignore[attr-defined]
                )
            except socket.timeout as exc:
                last_error = exc
                timeout_error = ConnectTimeoutError(self, f"Connection to {self.host} timed out.")  # type: ignore[attr-defined]
            except OSError as exc:
                last_error = exc
        if timeout_error is not None:
            raise timeout_error from last_error
        raise NewConnectionError(self, f"Failed to establish a new connection: {last_error}") from last_error  # type: ignore[arg-type]


class _SafeHTTPConnection(_SafeConnectionMixin, HTTPConnection):
    pass


class _SafeHTTPSConnection(_SafeConnectionMixin, HTTPSConnection):
    pass


class _SafeHTTPConnectionPool(HTTPConnectionPool):
    ConnectionCls = _SafeHTTPConnection


class _SafeHTTPSConnectionPool(HTTPSConnectionPool):
    ConnectionCls = _SafeHTTPSConnection


class SafeHTTPAdapter(HTTPAdapter):
    """HTTPAdapter, все соединения которого проходят проверку IP-адреса."""

    def init_poolmanager(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = {
            "http": _SafeHTTPConnectionPool,
            "https": _SafeHTTPSConnectionPool,
        }


def create_safe_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False  # не использовать прокси и .netrc из окружения
    adapter = SafeHTTPAdapter(max_retries=0)  # повторы выполняет tenacity
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml;q=0.9,text/plain;q=0.8",
            "Accept-Language": "ru,en;q=0.8",
        }
    )
    return session


# --- Загрузка страницы ---------------------------------------------------------


@dataclass(frozen=True)
class FetchedPage:
    url: str
    body: bytes
    encoding: str | None


@dataclass(frozen=True)
class _Redirect:
    location: str


class _TransientHTTPStatus(Exception):
    """HTTP 429/5xx — временная ошибка сервера, запрос стоит повторить."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _is_transient_fetch_error(exc: BaseException) -> bool:
    if isinstance(exc, requests.exceptions.SSLError):
        return False  # ошибка сертификата не исправится повтором
    return isinstance(
        exc,
        (
            _TransientHTTPStatus,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ),
    )


class PageFetcher:
    """Загружает HTML-страницу с ограничениями по размеру, времени и адресам."""

    def __init__(self, settings: Settings, session: requests.Session | None = None) -> None:
        self._settings = settings
        self._session = session

    def fetch(self, url: str) -> FetchedPage:
        current_url = validate_url(url)
        session = self._session or create_safe_session()
        try:
            for _ in range(self._settings.max_redirects + 1):
                result, attempts = self._fetch_with_retry(session, current_url)
                if isinstance(result, FetchedPage):
                    logger.info(
                        "stage=fetch event=downloaded url=%s bytes=%d attempts=%d",
                        redact_url(result.url),
                        len(result.body),
                        attempts,
                    )
                    return result
                next_url = urljoin(current_url, result.location)
                logger.info("stage=fetch event=redirect from=%s to=%s", redact_url(current_url), redact_url(next_url))
                current_url = validate_url(next_url)
            raise FetchError("Слишком много перенаправлений при загрузке страницы.")
        finally:
            if self._session is None:
                session.close()

    def _fetch_with_retry(self, session: requests.Session, url: str) -> tuple[FetchedPage | _Redirect, int]:
        try:
            return call_with_retry(
                lambda: self._fetch_once(session, url),
                stage="fetch",
                is_transient=_is_transient_fetch_error,
                max_attempts=self._settings.max_retries,
                initial_delay=self._settings.retry_initial_delay,
                max_delay=self._settings.retry_max_delay,
            )
        except _TransientHTTPStatus as exc:
            raise FetchError(f"Сайт вернул ошибку HTTP {exc.status_code}. Попробуйте позже.") from exc
        except requests.exceptions.Timeout as exc:
            raise FetchTimeoutError() from exc
        except requests.exceptions.SSLError as exc:
            raise FetchError("Не удалось установить защищённое соединение с сайтом (ошибка SSL).") from exc
        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as exc:
            raise FetchError("Не удалось подключиться к сайту. Проверьте адрес и попробуйте снова.") from exc
        except requests.exceptions.RequestException as exc:
            raise FetchError() from exc

    def _fetch_once(self, session: requests.Session, url: str) -> FetchedPage | _Redirect:
        timeout = self._settings.request_timeout
        response = session.get(url, timeout=(timeout, timeout), allow_redirects=False, stream=True)
        try:
            if response.is_redirect:
                return _Redirect(location=response.headers["Location"])
            status = response.status_code
            if status == 429 or status >= 500:
                raise _TransientHTTPStatus(status)
            if status >= 400:
                raise FetchError(f"Сайт вернул ошибку HTTP {status}. Проверьте адрес страницы.")
            if not 200 <= status < 300:
                raise FetchError(f"Неожиданный ответ сайта: HTTP {status}.")

            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type and content_type not in ALLOWED_CONTENT_TYPES:
                raise FetchError(f"Страница должна быть HTML-документом, получен тип {content_type!r}.")

            body = self._read_limited(response, deadline=time.monotonic() + timeout)
            # Без charset в заголовке requests подставляет ISO-8859-1; в этом случае
            # пусть BeautifulSoup определит кодировку по <meta charset> и содержимому.
            has_charset = "charset=" in response.headers.get("Content-Type", "").lower()
            encoding = get_encoding_from_headers(response.headers) if has_charset else None
            return FetchedPage(url=url, body=body, encoding=encoding)
        finally:
            response.close()

    def _read_limited(self, response: requests.Response, deadline: float) -> bytes:
        """Читает тело ответа потоково, не превышая MAX_PAGE_SIZE и общий тайм-аут."""
        limit = self._settings.max_page_size
        declared = response.headers.get("Content-Length")
        if declared and declared.isdigit() and int(declared) > limit:
            raise FetchError(f"Страница слишком большая (более {limit // 1000} КБ).")

        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            size += len(chunk)
            if size > limit:
                raise FetchError(f"Страница слишком большая (более {limit // 1000} КБ).")
            if time.monotonic() > deadline:
                raise FetchTimeoutError()
            chunks.append(chunk)
        return b"".join(chunks)


# --- Извлечение текста -----------------------------------------------------------

_REMOVE_TAGS = (
    "script", "style", "noscript", "svg", "iframe", "template", "canvas", "object", "embed",
    "button", "input", "select", "textarea", "nav", "footer", "aside", "dialog", "link", "meta",
)  # fmt: skip
_NOISE_ROLES = frozenset({"navigation", "banner", "contentinfo", "complementary", "search", "dialog", "alert"})
_NOISE_TOKENS = frozenset(
    {
        "ad", "ads", "advert", "advertisement", "adsbygoogle", "banner", "promo", "sponsored",
        "cookie", "cookies", "consent", "gdpr", "popup", "modal", "overlay",
        "nav", "navbar", "navigation", "menu", "breadcrumb", "breadcrumbs", "sidebar",
        "footer", "share", "sharing", "social", "subscribe", "newsletter",
        "comments", "comment", "related", "recommended", "widget",
    }
)  # fmt: skip
_PROTECTED_TAGS = frozenset({"html", "body", "main", "article"})
_TOKEN_SPLIT = re.compile(r"[\s\-_]+")
_WHITESPACE = re.compile(r"[ \t\r\f\v\u00a0\u200b]+")
_BLOCK_TAGS = (
    "p", "div", "section", "article", "main", "header", "li", "ul", "ol", "dl", "dt", "dd",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "table", "tr", "td", "th",
    "figure", "figcaption", "br", "hr",
)  # fmt: skip


def _attr_tokens(tag: Tag) -> Iterable[str]:
    classes = tag.get("class") or []
    if isinstance(classes, str):
        classes = [classes]
    values = [*classes, str(tag.get("id") or "")]
    for value in values:
        for token in _TOKEN_SPLIT.split(value.lower()):
            if token:
                yield token


def _is_noise(tag: Tag) -> bool:
    if tag.name in _PROTECTED_TAGS:
        return False
    if str(tag.get("role", "")).lower() in _NOISE_ROLES:
        return True
    if tag.has_attr("hidden") or str(tag.get("aria-hidden", "")).lower() == "true":
        return True
    style = str(tag.get("style", "")).replace(" ", "").lower()
    if "display:none" in style or "visibility:hidden" in style:
        return True
    return any(token in _NOISE_TOKENS for token in _attr_tokens(tag))


def _clean_text(raw: str) -> str:
    lines: list[str] = []
    for line in raw.splitlines():
        line = _WHITESPACE.sub(" ", line).strip()
        if line and (not lines or lines[-1] != line):
            lines.append(line)
    return "\n".join(lines)


def _meta_content(soup: BeautifulSoup, *selectors: dict[str, str]) -> str:
    for attrs in selectors:
        tag = soup.find("meta", attrs=attrs)
        if isinstance(tag, Tag) and tag.get("content"):
            return _clean_text(str(tag["content"])).replace("\n", " ")
    return ""


def truncate_text(text: str, max_length: int) -> str:
    """Обрезает текст до ``max_length`` символов по границе слова."""
    if len(text) <= max_length:
        return text
    cut = text[:max_length]
    boundary = max(cut.rfind(" "), cut.rfind("\n"))
    if boundary > max_length * 0.8:
        cut = cut[:boundary]
    return cut.rstrip() + " …"


def extract_content(html: str | bytes, url: str, max_text_length: int, encoding: str | None = None) -> PageContent:
    """Извлекает заголовок, метаописание и основной текст страницы.

    JavaScript не выполняется. Если содержательного текста меньше
    ``MIN_TEXT_LENGTH`` символов, выбрасывается ``ContentExtractionError``.
    """
    soup = BeautifulSoup(html, "html.parser", from_encoding=encoding if isinstance(html, bytes) else None)

    title = _meta_content(soup, {"property": "og:title"}, {"name": "twitter:title"})
    if not title and soup.title and soup.title.string:
        title = _clean_text(soup.title.string).replace("\n", " ")
    description = _meta_content(
        soup, {"name": "description"}, {"property": "og:description"}, {"name": "twitter:description"}
    )

    for tag in soup.find_all(list(_REMOVE_TAGS)):
        tag.decompose()
    for tag in [t for t in soup.find_all(True) if _is_noise(t)]:
        if not tag.decomposed:
            tag.decompose()
    # Шапка сайта — шум, но <header> внутри статьи обычно содержит её заголовок.
    for tag in soup.find_all("header"):
        if not tag.decomposed and tag.find_parent("article") is None:
            tag.decompose()
    # Переносы строк — только на границах блочных элементов, чтобы ссылки и
    # выделения внутри абзаца не разрывали предложения.
    for tag in soup.find_all(list(_BLOCK_TAGS)):
        tag.insert_before("\n")
        tag.insert_after("\n")

    text = _main_text(soup)
    if len(text) < MIN_TEXT_LENGTH:
        raise ContentExtractionError()

    return PageContent(
        url=url,
        title=title[:300],
        description=description[:1000],
        text=truncate_text(text, max_text_length),
    )


def _main_text(soup: BeautifulSoup) -> str:
    """Выбирает основной блок страницы: самую длинную <article>, затем <main>, затем <body>."""
    candidates: list[Tag] = [t for t in soup.find_all("article") if isinstance(t, Tag)]
    candidates += [t for t in soup.find_all(["main"]) if isinstance(t, Tag)]
    candidates += [t for t in soup.find_all(attrs={"role": "main"}) if isinstance(t, Tag)]

    best = ""
    for candidate in candidates:
        text = _clean_text(candidate.get_text())
        if len(text) > len(best):
            best = text
    if len(best) >= MIN_TEXT_LENGTH:
        return best
    root = soup.body or soup
    return _clean_text(root.get_text())
