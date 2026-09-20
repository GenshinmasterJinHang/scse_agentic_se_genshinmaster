"""Tests for ``run_analyst`` and ``call_qwen``: payload contract, retries, and errors.

All HTTP calls are mocked; nothing here contacts a real Ollama server.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from analyst_agent import (
    OllamaError,
    RequirementsValidationError,
    REQUIREMENTS_SCHEMA,
    call_qwen,
    get_model_name,
    run_analyst,
    validate_requirements,
)
from urllib.error import HTTPError, URLError

from tests._http_mock import mock_opener


GOOD_REQUIREMENTS = {
    "goal": "navigate toward the goal when safe",
    "allowed_actions": ["FORWARD", "LEFT", "RIGHT", "STOP"],
    "safe_stop": True,
    "avoid_obstacles": True,
}


def _envelope(content: str) -> dict:
    """Build a fake but well-formed Ollama /api/chat response envelope."""
    return {
        "model": "qwen3:8b",
        "created_at": "2026-09-20T00:00:00Z",
        "done": True,
        "done_reason": "stop",
        "message": {"role": "assistant", "content": content},
    }


BRIEF = "The robot must prefer the goal direction when safe and stop otherwise."


class PayloadContractTests(unittest.TestCase):
    """The HTTP request body must satisfy the project contract."""

    def test_structured_request_uses_qwen3_8b(self) -> None:
        with mock_opener(respond=lambda _payload: _envelope("{}")) as captured:
            call_qwen("sys", "user", structured=True)
        self.assertEqual(captured["payload"]["model"], "qwen3:8b")

    def test_structured_request_sets_stream_false(self) -> None:
        with mock_opener(respond=lambda _payload: _envelope("{}")) as captured:
            call_qwen("sys", "user", structured=True)
        self.assertEqual(captured["payload"]["stream"], False)

    def test_structured_request_sets_think_false(self) -> None:
        with mock_opener(respond=lambda _payload: _envelope("{}")) as captured:
            call_qwen("sys", "user", structured=True)
        self.assertEqual(captured["payload"]["think"], False)

    def test_structured_request_includes_schema_format(self) -> None:
        with mock_opener(respond=lambda _payload: _envelope("{}")) as captured:
            call_qwen("sys", "user", structured=True)
        self.assertEqual(captured["payload"]["format"], REQUIREMENTS_SCHEMA)

    def test_unstructured_request_omits_schema_format(self) -> None:
        # Stage A must NOT force the Stage B schema or the model cannot emit free text.
        with mock_opener(respond=lambda _payload: _envelope("hello")) as captured:
            call_qwen("sys", "user", structured=False)
        self.assertNotIn("format", captured["payload"])

    def test_unstructured_request_also_sets_think_false_by_default(self) -> None:
        with mock_opener(respond=lambda _payload: _envelope("hello")) as captured:
            call_qwen("sys", "user", structured=False)
        self.assertEqual(captured["payload"]["think"], False)

    def test_unstructured_request_respects_qwen_think_true(self) -> None:
        with patch.dict("os.environ", {"QWEN_THINK": "true"}):
            with mock_opener(respond=lambda _payload: _envelope("hi")) as captured:
                call_qwen("sys", "user", structured=False)
        self.assertEqual(captured["payload"]["think"], True)

    def test_request_uses_localhost_endpoint(self) -> None:
        with mock_opener(respond=lambda _payload: _envelope("{}")) as captured:
            call_qwen("sys", "user", structured=True)
        self.assertEqual(captured["request"].full_url, "http://127.0.0.1:11434/api/chat")
        self.assertEqual(captured["request"].headers.get("Content-type"), "application/json")

    def test_messages_include_full_brief_in_user_role(self) -> None:
        with mock_opener(respond=lambda _payload: _envelope(json.dumps(GOOD_REQUIREMENTS))) as captured:
            call_qwen("system text", "USER BRIEF PAYLOAD", structured=True)
        messages = captured["payload"]["messages"]
        self.assertEqual(messages[0], {"role": "system", "content": "system text"})
        self.assertEqual(messages[1], {"role": "user", "content": "USER BRIEF PAYLOAD"})

    def test_options_temperature_and_num_predict_match_contract(self) -> None:
        with mock_opener(respond=lambda _payload: _envelope("{}")) as captured:
            call_qwen("sys", "user", structured=True)
        options = captured["payload"]["options"]
        self.assertEqual(options["temperature"], 0.2)
        self.assertEqual(options["num_predict"], 1024)

    def test_qwen_model_env_override_is_respected(self) -> None:
        with patch.dict("os.environ", {"QWEN_MODEL": "qwen3:8b"}):
            self.assertEqual(get_model_name(), "qwen3:8b")


class InfrastructureErrorTests(unittest.TestCase):
    """Network, timeout, and HTTP-level failures must surface clearly."""

    def test_url_error_becomes_ollama_error(self) -> None:
        with mock_opener(raise_exception=URLError("connection refused")):
            with self.assertRaises(OllamaError) as ctx:
                call_qwen("s", "u")
        self.assertIn("Cannot reach local Ollama", str(ctx.exception))

    def test_timeout_becomes_ollama_error(self) -> None:
        with mock_opener(raise_exception=TimeoutError("slow")):
            with self.assertRaises(OllamaError) as ctx:
                call_qwen("s", "u")
        self.assertIn("timed out", str(ctx.exception))

    def test_http_404_suggests_pull(self) -> None:
        err = HTTPError(
            url="http://127.0.0.1:11434/api/chat",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )
        with mock_opener(raise_exception=err):
            with self.assertRaises(OllamaError) as ctx:
                call_qwen("s", "u")
        self.assertIn("ollama pull", str(ctx.exception))

    def test_http_500_passes_status_through(self) -> None:
        err = HTTPError(
            url="http://127.0.0.1:11434/api/chat",
            code=500,
            msg="Internal",
            hdrs=None,
            fp=None,
        )
        with mock_opener(raise_exception=err):
            with self.assertRaises(OllamaError) as ctx:
                call_qwen("s", "u")
        self.assertIn("HTTP 500", str(ctx.exception))

    def test_envelope_not_json(self) -> None:
        with mock_opener(respond=lambda _p: b"not json"):
            with self.assertRaises(OllamaError):
                call_qwen("s", "u")

    def test_envelope_error_field(self) -> None:
        with mock_opener(respond=lambda _p: {"error": "boom"}):
            with self.assertRaises(OllamaError):
                call_qwen("s", "u")

    def test_envelope_not_done(self) -> None:
        with mock_opener(respond=lambda _p: {"done": False, "message": {"content": "x"}}):
            with self.assertRaises(OllamaError):
                call_qwen("s", "u")

    def test_envelope_done_reason_length(self) -> None:
        with mock_opener(
            respond=lambda _p: {
                "done": True,
                "done_reason": "length",
                "message": {"content": "x"},
            }
        ):
            with self.assertRaises(OllamaError):
                call_qwen("s", "u")

    def test_empty_content_rejected(self) -> None:
        with mock_opener(respond=lambda _p: {"done": True, "message": {"content": "  "}}):
            with self.assertRaises(OllamaError):
                call_qwen("s", "u")

    def test_message_with_only_thinking_does_not_pass_through(self) -> None:
        # Per the contract: thinking must NOT be treated as the final answer.
        with mock_opener(
            respond=lambda _p: {
                "done": True,
                "message": {"thinking": "long internal monologue", "content": ""},
            }
        ):
            with self.assertRaises(OllamaError):
                call_qwen("s", "u")


class RunAnalystTests(unittest.TestCase):
    """End-to-end behaviour of ``run_analyst`` with mocked HTTP."""

    def test_returns_validated_dict_on_first_try(self) -> None:
        with mock_opener(respond=lambda _p: _envelope(json.dumps(GOOD_REQUIREMENTS))):
            result = run_analyst(BRIEF)
        self.assertEqual(result, GOOD_REQUIREMENTS)
        # validate_requirements returns the validated dict; equality is what matters.
        self.assertEqual(result, validate_requirements(GOOD_REQUIREMENTS))

    def test_first_invalid_second_valid_retries_then_succeeds(self) -> None:
        responses = iter(
            [
                _envelope("```json\n" + json.dumps(GOOD_REQUIREMENTS) + "\n```"),  # fences
                _envelope(json.dumps(GOOD_REQUIREMENTS)),
            ]
        )
        attempts = {"n": 0}
        user_contents: list[str] = []

        def respond(_payload):
            attempts["n"] += 1
            user_contents.append(_payload["messages"][1]["content"])
            return next(responses)

        with mock_opener(respond=respond):
            result = run_analyst(BRIEF)
        self.assertEqual(result, GOOD_REQUIREMENTS)
        self.assertEqual(attempts["n"], 2, msg="expected exactly 2 attempts")
        # First attempt carries only the brief; the retry must include feedback.
        self.assertIn("failed validation", user_contents[1])

    def test_two_invalid_responses_raises_after_two_attempts(self) -> None:
        bad = _envelope("not json at all")
        attempts = {"n": 0}

        def respond(_payload):
            attempts["n"] += 1
            return bad

        with mock_opener(respond=respond):
            with self.assertRaises(RequirementsValidationError) as ctx:
                run_analyst(BRIEF)
        self.assertEqual(attempts["n"], 2)
        self.assertIn("twice", str(ctx.exception))

    def test_infrastructure_failure_does_not_return_canned_answer(self) -> None:
        # A network error must NOT silently produce a valid result.
        with mock_opener(raise_exception=URLError("offline")):
            with self.assertRaises(OllamaError):
                run_analyst(BRIEF)

    def test_full_brief_passed_through_to_model(self) -> None:
        seen_briefs: list[str] = []
        custom_brief = "CUSTOM_BRIEF_TOKEN_42\nmulti-line\nstill brief"

        def respond(_payload):
            seen_briefs.append(_payload["messages"][1]["content"])
            return _envelope(json.dumps(GOOD_REQUIREMENTS))

        with mock_opener(respond=respond):
            run_analyst(custom_brief)
        self.assertTrue(any("CUSTOM_BRIEF_TOKEN_42" in s for s in seen_briefs))
        self.assertTrue(all(custom_brief in s for s in seen_briefs))

    def test_empty_brief_rejected_before_http(self) -> None:
        with self.assertRaises(ValueError):
            run_analyst("   ")

    def test_non_string_brief_rejected_before_http(self) -> None:
        with self.assertRaises(ValueError):
            run_analyst(123)  # type: ignore[arg-type]


class DuplicateJsonKeyTests(unittest.TestCase):
    """The JSON parser must reject duplicate keys, not silently drop them."""

    def test_duplicate_key_in_model_response_is_rejected(self) -> None:
        # Qwen accidentally writes goal twice in the same object.
        dup = (
            '{"goal": "x", "goal": "y", '
            '"allowed_actions": ["FORWARD", "LEFT", "RIGHT", "STOP"], '
            '"safe_stop": true, "avoid_obstacles": true}'
        )
        with mock_opener(respond=lambda _p: _envelope(dup)):
            with self.assertRaises(RequirementsValidationError) as ctx:
                run_analyst(BRIEF)
        self.assertIn("Duplicate JSON key", str(ctx.exception))

    def test_nonstandard_constant_nan_is_rejected(self) -> None:
        bad = '{"goal": NaN, "allowed_actions": ["FORWARD","LEFT","RIGHT","STOP"], "safe_stop": true, "avoid_obstacles": true}'
        with mock_opener(respond=lambda _p: _envelope(bad)):
            with self.assertRaises(RequirementsValidationError) as ctx:
                run_analyst(BRIEF)
        self.assertIn("Non-standard JSON constant", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
