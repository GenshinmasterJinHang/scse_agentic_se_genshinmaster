"""Tests for developer_agent: validate_developer_output + run_developer."""

from __future__ import annotations

import json
import textwrap
import unittest
from urllib.error import URLError

from analyst_agent import OllamaError
from developer_agent import (
    DEVELOPER_SCHEMA,
    DeveloperValidationError,
    run_developer,
    validate_developer_output,
)

from tests._http_mock import mock_opener


def _envelope(content: str) -> dict:
    return {
        "model": "qwen3:8b",
        "done": True,
        "done_reason": "stop",
        "message": {"role": "assistant", "content": content},
    }


GOOD_PLAN = {
    "strategy": "Prefer goal direction when safe; stop when no direction is safe.",
    "decisions": [
        {"condition": "goal ahead and forward is clear", "action": "FORWARD"},
        {"condition": "no safe direction", "action": "STOP"},
    ],
    "stop_condition": "When forward, left, and right are all blocked.",
}


def _good_code() -> str:
    return textwrap.dedent(
        """
        def decide_next_move(state):
            if state.get("goal_ahead") and not state.get("front_blocked"):
                return "FORWARD"
            if state.get("goal_on_left") and not state.get("left_blocked"):
                return "LEFT"
            if state.get("goal_on_right") and not state.get("right_blocked"):
                return "RIGHT"
            return "STOP"
        """
    ).strip()


def _good_envelope() -> dict:
    return {"description": "Decides which of FORWARD/LEFT/RIGHT/STOP to take.", "code": _good_code()}


class ValidateDeveloperOutputTests(unittest.TestCase):
    def test_legal_envelope_passes(self) -> None:
        env = _good_envelope()
        self.assertIs(validate_developer_output(env), env)

    def test_non_dict_rejected(self) -> None:
        with self.assertRaises(DeveloperValidationError):
            validate_developer_output("code")  # type: ignore[arg-type]

    def test_missing_field_rejected(self) -> None:
        env = _good_envelope()
        env.pop("code")
        with self.assertRaises(DeveloperValidationError) as ctx:
            validate_developer_output(env)
        self.assertIn("code", str(ctx.exception))

    def test_extra_field_rejected(self) -> None:
        env = {**_good_envelope(), "extra": "x"}
        with self.assertRaises(DeveloperValidationError) as ctx:
            validate_developer_output(env)
        self.assertIn("extra", str(ctx.exception))

    def test_empty_description_rejected(self) -> None:
        env = _good_envelope()
        env["description"] = "  "
        with self.assertRaises(DeveloperValidationError):
            validate_developer_output(env)

    def test_empty_code_rejected(self) -> None:
        env = _good_envelope()
        env["code"] = ""
        with self.assertRaises(DeveloperValidationError):
            validate_developer_output(env)

    def test_code_must_be_parseable_python(self) -> None:
        env = _good_envelope()
        # Drop the trailing ':' so the function header becomes a syntax error.
        env["code"] = "def decide_next_move(state)\n    return 'STOP'\n"
        with self.assertRaises(DeveloperValidationError) as ctx:
            validate_developer_output(env)
        self.assertIn("not valid Python", str(ctx.exception))

    def test_code_without_decide_next_move_rejected(self) -> None:
        env = _good_envelope()
        env["code"] = "x = 1\ny = 2\n"
        with self.assertRaises(DeveloperValidationError) as ctx:
            validate_developer_output(env)
        self.assertIn("decide_next_move", str(ctx.exception))

    def test_wrong_entry_function_name_rejected(self) -> None:
        env = _good_envelope()
        env["code"] = "def decide_action(state):\n    return 'STOP'\n"
        with self.assertRaises(DeveloperValidationError) as ctx:
            validate_developer_output(env)
        self.assertIn("decide_next_move", str(ctx.exception))

    def test_wrong_parameter_name_rejected(self) -> None:
        env = _good_envelope()
        env["code"] = "def decide_next_move(sensor_data):\n    return 'STOP'\n"
        with self.assertRaises(DeveloperValidationError) as ctx:
            validate_developer_output(env)
        self.assertIn("state", str(ctx.exception))

    def test_wrong_parameter_count_rejected(self) -> None:
        env = _good_envelope()
        env["code"] = (
            "def decide_next_move(state, goal_direction):\n    return 'STOP'\n"
        )
        with self.assertRaises(DeveloperValidationError) as ctx:
            validate_developer_output(env)
        self.assertIn("state", str(ctx.exception))


class RunDeveloperTests(unittest.TestCase):
    def test_returns_validated_envelope(self) -> None:
        with mock_opener(respond=lambda _p: _envelope(json.dumps(_good_envelope()))):
            result = run_developer(GOOD_PLAN)
        self.assertEqual(result, _good_envelope())

    def test_uses_qwen3_8b_stream_false_think_false_and_schema(self) -> None:
        with mock_opener(respond=lambda _p: _envelope(json.dumps(_good_envelope()))) as captured:
            run_developer(GOOD_PLAN)
        payload = captured["payload"]
        self.assertEqual(payload["model"], "qwen3:8b")
        self.assertEqual(payload["stream"], False)
        self.assertEqual(payload["think"], False)
        schema = payload["format"]
        self.assertEqual(schema["type"], "object")
        self.assertEqual(set(schema["required"]), {"description", "code"})
        self.assertEqual(schema["additionalProperties"], False)
        self.assertEqual(payload["options"]["temperature"], 0.2)

    def test_sends_only_plan_not_requirements(self) -> None:
        user_contents: list[str] = []

        def respond(payload):
            user_contents.append(payload["messages"][1]["content"])
            return _envelope(json.dumps(_good_envelope()))

        with mock_opener(respond=respond):
            run_developer(GOOD_PLAN)
        sent = user_contents[0]
        # Plan fields present; requirements provenance not present.
        self.assertIn("strategy", sent)
        self.assertIn("decisions", sent)
        self.assertIn("stop_condition", sent)
        self.assertNotIn("safe_stop", sent)
        self.assertNotIn("avoid_obstacles", sent)

    def test_first_invalid_then_valid_retries(self) -> None:
        responses = iter([
            _envelope("```json\n" + json.dumps(_good_envelope()) + "\n```"),
            _envelope(json.dumps(_good_envelope())),
        ])
        attempts = {"n": 0}

        def respond(_payload):
            attempts["n"] += 1
            return next(responses)

        with mock_opener(respond=respond):
            env = run_developer(GOOD_PLAN)
        self.assertEqual(env, _good_envelope())
        self.assertEqual(attempts["n"], 2)

    def test_two_invalid_attempts_raises(self) -> None:
        with mock_opener(respond=lambda _p: _envelope("garbage")):
            with self.assertRaises(DeveloperValidationError) as ctx:
                run_developer(GOOD_PLAN)
        self.assertIn("twice", str(ctx.exception))

    def test_network_error_does_not_return_canned_envelope(self) -> None:
        with mock_opener(raise_exception=URLError("offline")):
            with self.assertRaises(OllamaError):
                run_developer(GOOD_PLAN)

    def test_non_dict_plan_rejected_before_http(self) -> None:
        with self.assertRaises(ValueError):
            run_developer("not a dict")  # type: ignore[arg-type]

    def test_duplicate_key_in_response_rejected(self) -> None:
        dup = '{"description": "d", "description": "d", "code": "def decide_next_move(state): return \\"STOP\\""}'
        with mock_opener(respond=lambda _p: _envelope(dup)):
            with self.assertRaises(DeveloperValidationError):
                run_developer(GOOD_PLAN)


if __name__ == "__main__":
    unittest.main()