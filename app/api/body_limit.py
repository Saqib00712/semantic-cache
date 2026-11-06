"""Refuse oversized request bodies before they are read into memory: the declared Content-Length, and a running count of the bytes
actually received (a client can lie or stream chunked). Either gives 413."""

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.errors import error_response


class BodyLimitMiddleware:
    """Pure ASGI middleware (the protocol needs a callable object that holds the app and the limit)."""

    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = Headers(scope=scope).get("content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > self.max_bytes):
            await error_response(413, "payload_too_large", "request body is too large")(scope, receive, send)
            return
        received, started, exceeded = 0, False, False

        async def counting_receive() -> Message:
            nonlocal received, exceeded
            if exceeded:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    exceeded = True
                    return {"type": "http.disconnect"}
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if exceeded and not started:
                return  # the app is answering a cut-off body; the 413 below replaces that answer
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except Exception:
            if not exceeded:
                raise
        if exceeded and not started:
            await error_response(413, "payload_too_large", "request body is too large")(scope, receive, send)
