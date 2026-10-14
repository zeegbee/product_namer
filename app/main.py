"""FastAPI-приложение: REST API и веб-интерфейс Product Namer."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse

from app import __version__
from app.agent import ProductNamerAgent
from app.config import Settings, load_settings, setup_logging
from app.errors import ConfigError, ProductNamerError
from app.schemas import ErrorResponse, GenerateRequest, GenerateResponse, HealthResponse

logger = logging.getLogger(__name__)

TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}

_ERROR_RESPONSES = {
    422: {"model": ErrorResponse, "description": "Некорректный URL или входные данные"},
    500: {"model": ErrorResponse, "description": "Внутренняя ошибка приложения"},
    502: {"model": ErrorResponse, "description": "Ошибка внешнего сервиса или недопустимый ответ модели"},
    504: {"model": ErrorResponse, "description": "Превышено время ожидания"},
}


def _error(status_code: int, detail: str, error_type: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail, "error_type": error_type})


def create_app(settings: Settings | None = None, agent: ProductNamerAgent | None = None) -> FastAPI:
    """Создаёт приложение. Параметры позволяют подменить настройки и агента в тестах."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            app_settings = settings or load_settings()
        except ConfigError as exc:
            setup_logging()
            logger.critical("stage=startup event=config_error message=%s", exc)
            raise
        setup_logging(app_settings.log_level, app_settings.log_file)

        app_agent = agent or ProductNamerAgent(app_settings)
        if agent is None and app_settings.verify_model_on_startup:
            try:
                app_agent.llm.verify_model()
            except ConfigError as exc:
                logger.critical("stage=startup event=config_error message=%s", exc)
                raise

        app.state.settings = app_settings
        app.state.agent = app_agent
        logger.info("stage=startup event=ready version=%s model=%s", __version__, app_settings.openai_model)
        yield
        logger.info("stage=shutdown event=stopped")

    app = FastAPI(
        title="Product Namer",
        version=__version__,
        description="AI-агент, который подбирает 5 названий продукта по описанию на веб-странице.",
        lifespan=lifespan,
    )
    index_html = TEMPLATE_PATH.read_text(encoding="utf-8")

    @app.middleware("http")
    async def security_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        for header, value in _SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        return response

    @app.exception_handler(ProductNamerError)
    async def handle_app_error(request: Request, exc: ProductNamerError) -> JSONResponse:
        return _error(exc.status_code, exc.user_message, exc.error_type)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        messages = []
        for error in exc.errors():
            message = str(error.get("msg", "")).removeprefix("Value error, ")
            messages.append(message)
        detail = "; ".join(messages) or "Некорректные входные данные."
        if any(error.get("type") == "missing" for error in exc.errors()):
            detail = "Передайте JSON вида {\"url\": \"https://example.com/product\"}."
        return _error(422, detail, "validation_error")

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("stage=request event=unhandled_error path=%s error_type=%s", request.url.path, type(exc).__name__)
        return _error(500, "Внутренняя ошибка приложения. Подробности — в логах сервера.", "internal_error")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        return HTMLResponse(index_html)

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    def health(request: Request) -> HealthResponse:
        return HealthResponse(status="ok", version=__version__, model=request.app.state.settings.openai_model)

    @app.post("/api/generate", response_model=GenerateResponse, responses=_ERROR_RESPONSES, tags=["naming"])
    def generate(payload: GenerateRequest, request: Request) -> GenerateResponse:
        # Синхронный обработчик: FastAPI выполняет его в пуле потоков,
        # поэтому блокирующие requests/OpenAI-вызовы не останавливают event loop.
        return request.app.state.agent.run(payload.url)

    return app


app = create_app()
