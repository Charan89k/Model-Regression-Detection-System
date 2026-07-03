import time
import uuid
from typing import Awaitable, Callable

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from mrds.core.logging.setup import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle structlog context binding for every HTTP request.
    Injects request IDs, correlation IDs, and logs the request/response lifecycle.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Extract or generate Request and Correlation IDs
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))

        # Clear context from previous requests in this async task context
        structlog.contextvars.clear_contextvars()

        # Bind context variables for this specific request globally across the async flow
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            correlation_id=correlation_id,
            path=request.url.path,
            method=request.method,
            client_ip=request.client.host if request.client else "unknown",
        )

        start_time = time.perf_counter()

        try:
            logger.info("HTTP Request started")
            response = await call_next(request)

            process_time = time.perf_counter() - start_time

            # Inject headers back to the client
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Correlation-ID"] = correlation_id

            logger.info(
                "HTTP Request completed",
                status_code=response.status_code,
                duration_ms=round(process_time * 1000, 2),
            )
            return response

        except Exception as exc:
            process_time = time.perf_counter() - start_time
            logger.exception(
                "HTTP Request failed with unhandled exception",
                duration_ms=round(process_time * 1000, 2),
                error=str(exc),
            )
            raise
