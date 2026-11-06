"""One error shape for every failure: {"error": {"code", "message", "request_id"}}. Fixed text only: submitted values, exception
messages and paths are never echoed."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.observability.context import request_id_var


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict[str, str] | None = None):
        super().__init__(code)
        self.status, self.code, self.message, self.headers = status, code, message, headers or {}


def error_response(status: int, code: str, message: str, headers: dict[str, str] | None = None, details: list[dict] | None = None):
    error: dict[str, object] = {"code": code, "message": message, "request_id": request_id_var.get()}
    if details:
        error["details"] = details
    return JSONResponse({"error": error}, status_code=status, headers=headers)


_HTTP = {404: ("not_found", "request could not be served"), 405: ("method_not_allowed", "method not allowed")}


def safe_details(errors: list) -> list[dict[str, str]]:
    """Which field is wrong and why (pydantic's error TYPE only). Submitted values and client-chosen field names are never echoed."""
    out = []
    for e in errors[:10]:
        kind = str(e.get("type", "invalid"))
        loc = [str(p) for p in e.get("loc", ()) if p != "body"]
        if kind == "extra_forbidden" and loc:
            loc[-1] = "<unknown field>"
        out.append({"field": ".".join(x[:40] for x in loc) or "body", "problem": kind})
    return out


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api(_: Request, exc: ApiError):
        return error_response(exc.status, exc.code, exc.message, exc.headers)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        code, message = _HTTP.get(exc.status_code, ("http_error", "request could not be served"))
        return error_response(exc.status_code, code, message)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        return error_response(422, "invalid_request", "the request is not valid", details=safe_details(exc.errors()))


def internal_error_response():
    return error_response(500, "internal_error", "an internal error occurred")
