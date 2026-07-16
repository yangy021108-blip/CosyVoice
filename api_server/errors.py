"""Stable error types exposed by the HTTP API."""

from __future__ import annotations


class ServiceError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        code: str,
        error_type: str = "invalid_request_error",
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code
        self.error_type = error_type
        self.param = param
