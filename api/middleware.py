"""
Operational and Security Middlewares for Document Verification API.

Includes:
- SecurityHeadersMiddleware: Adds defensive HTTP headers (nosniff, DENY, no-referrer, no-store).
- CorrelationAndLoggingMiddleware: Injects X-Request-ID, logs safe metadata, measures latency.
- RateLimitMiddleware: Enforces per-client IP sliding window rate limits with standard 429 response.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from api.config import api_config
from api.rate_limiter import rate_limiter

logger = logging.getLogger("api.operations")


def get_client_identifier(request: Request) -> str:
    """Extracts a sanitized client identifier from IP or forwarding headers."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take the leftmost client IP
        first_ip = forwarded.split(",")[0].strip()
        if re.match(r"^[0-9a-fA-F:.]+$", first_ip):
            return first_ip
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"


class SecurityAndObservabilityMiddleware(BaseHTTPMiddleware):
    """
    Combined high-performance middleware applying:
    1. Correlation ID management (X-Request-ID)
    2. Rate limiting enforcement (HTTP 429)
    3. Security headers injection
    4. Safe operational timing and metadata logging
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()

        # 1. Determine or generate Correlation / Request ID
        incoming_id = request.headers.get("X-Request-ID")
        if incoming_id and re.match(r"^[a-zA-Z0-9_-]{1,64}$", incoming_id):
            req_id = incoming_id
        else:
            req_id = uuid.uuid4().hex

        request.state.request_id = req_id
        client_id = get_client_identifier(request)

        # 2. Rate Limiting Check (bypass for docs/openapi/health/ready if needed, or apply uniformly)
        # Apply rate limiting to all requests
        allowed, retry_after = rate_limiter.is_allowed(
            client_id=client_id,
            max_requests=api_config.RATE_LIMIT_REQUESTS,
            window_seconds=api_config.RATE_LIMIT_WINDOW_SECONDS,
        )

        if not allowed:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.warning(
                "[%s] Rate limit exceeded for client %s on %s %s (%d requests in %ds)",
                req_id, client_id, request.method, request.url.path,
                api_config.RATE_LIMIT_REQUESTS, api_config.RATE_LIMIT_WINDOW_SECONDS
            )
            response = JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": f"Rate limit exceeded. Please retry after {retry_after} seconds.",
                    },
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-Request-ID": req_id,
                    "X-Content-Type-Options": "nosniff",
                    "X-Frame-Options": "DENY",
                    "Referrer-Policy": "no-referrer",
                    "Cache-Control": "no-store",
                },
            )
            return response

        # 3. Log incoming request metadata (zero PII)
        content_length = request.headers.get("content-length", "0")
        logger.info(
            "[%s] Incoming %s %s - Client: %s - Content-Length: %s bytes",
            req_id, request.method, request.url.path, client_id, content_length
        )

        # 4. Dispatch downstream request handler
        try:
            response: Response = await call_next(request)
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "[%s] Unhandled exception processing %s %s after %.2f ms",
                req_id, request.method, request.url.path, duration_ms, exc_info=True
            )
            response = JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "Document verification could not be completed.",
                    },
                },
            )

        # 5. Inject Security Headers and Correlation ID
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"

        # 6. Log operational summary metadata (strictly no document content or PII)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "[%s] Completed %s %s - Status %d in %.2f ms [client=%s]",
            req_id, request.method, request.url.path, response.status_code, duration_ms, client_id
        )

        return response
