"""A minimal OpenAI-compatible mock LLM gateway served over real HTTP. It records the headers it receives and enforces a toy policy
(a deny-list of models and a require-key check) so the tests can show the cache never routes around it."""

import socket
import threading
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class MockGateway:
    def __init__(self, key="gw-test-key", denied=("gpt-denied",)):
        self.key, self.denied, self.seen = key, set(denied), []
        self.app = FastAPI()

        @self.app.get("/health")
        def health():
            return {"status": "ok"}

        @self.app.post("/v1/chat/completions")
        async def chat(request: Request):
            body = await request.json()
            self.seen.append({"headers": dict(request.headers), "model": body.get("model")})
            if request.headers.get("authorization") != f"Bearer {self.key}":
                return JSONResponse({"error": {"message": "bad key"}}, status_code=401)
            if body.get("model") in self.denied:
                return JSONResponse({"error": {"message": "policy: model denied"}}, status_code=403)
            q = body["messages"][-1]["content"]
            return {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": f"GW[{q}]"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}  # fmt: skip

    def start(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="critical"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                return self
            time.sleep(0.05)
        raise RuntimeError("mock gateway did not start")

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=5)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/v1"
