"""Helpers for mocking urllib HTTP calls without touching the network."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Callable
from unittest.mock import patch


class FakeResponse:
    """Mimics the context-manager response returned by ``opener.open(...)``."""

    def __init__(self, payload: Any = None, *, raw: bytes | None = None) -> None:
        if raw is None:
            raw = json.dumps(payload).encode("utf-8")
        self._raw = raw

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return self._raw


@contextmanager
def mock_opener(
    *,
    respond: Callable[[Any], Any] | None = None,
    raise_exception: BaseException | None = None,
):
    """Patch ``analyst_agent.build_opener`` so no real HTTP request is made.

    ``respond`` receives the JSON payload that ``call_qwen`` would have sent and
    must return the bytes that the fake server sends back. Use ``raise_exception``
    to make ``opener.open`` raise (HTTPError, URLError, TimeoutError, ...).
    """

    captured: dict[str, Any] = {}

    def fake_build_opener(*handlers, **kwargs):  # noqa: ARG001
        class Opener:
            def open(self_inner, request, timeout=None):  # noqa: N805
                captured["request"] = request
                captured["timeout"] = timeout
                try:
                    body = json.loads(request.data.decode("utf-8"))
                except Exception:
                    body = None
                captured["payload"] = body
                if raise_exception is not None:
                    raise raise_exception
                if respond is None:
                    raise AssertionError("mock_opener used without respond/raise_exception")
                raw = respond(body)
                if not isinstance(raw, (bytes, bytearray)):
                    raw = json.dumps(raw).encode("utf-8")
                return FakeResponse(raw=bytes(raw))

        return Opener()

    with patch("analyst_agent.build_opener", side_effect=fake_build_opener):
        yield captured
