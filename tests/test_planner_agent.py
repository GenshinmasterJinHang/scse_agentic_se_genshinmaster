"""Tests for planner_agent: validate_plan + run_planner contract + retries."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from analyst_agent import OllamaError
from planner_agent import (
    PLANNER_SCHEMA,
    PlanValidationError,
    run_planner,
    validate_plan,
)
from urllib.error import URLError

from tests._http_mock import mock_opener


GOOD_REQUIREMENTS = {
    "goal": "navigate toward the goal when safe",
    "allowed_actions": ["FORWARD", "LEFT", "RIGHT", "STOP"],
    "safe_stop": True,
    "avoid_obstacles": True,
}


def _envelope(content: str) -> dict:
    return {
        "model": "qwen3:8b",
        "done": True,
        "done_reason": "stop",
        "message": {"role": "assistant", "content": content},
    }


def _good_plan() -> dict:
    return {
        "strategy": "Prefer goal direction when safe; stop when no direction is safe.",
        "decisions": [
            {"condition": "goal ahead and forward is clear", "action": "FORWARD"},
            {"condition": "goal to the left and left is clear", "action": "LEFT"},
            {"condition": "no safe direction", "action": "STOP"},
        ],
        "stop_condition": "When forward, left, and right are all blocked.",
    }


class ValidatePlanTests(unittest.TestCase):
    def test_legal_plan_returns_unchanged(self) -> None:
        plan = _good_plan()
        result = validate_plan(plan)
        self.assertIs(result, plan)

    def test_non_dict_rejected(self) -> None:
        with self.assertRaises(PlanValidationError):
            validate_plan(["not", "a", "dict"])  # type: ignore[arg-type]

    def test_missing_field_rejected(self) -> None:
        bad = _good_plan()
        bad.pop("stop_condition")
        with self.assertRaises(PlanValidationError) as ctx:
            validate_plan(bad)
        self.assertIn("stop_condition", str(ctx.exception))

    def test_extra_field_rejected(self) -> None:
        bad = {**_good_plan(), "extra": "x"}
        with self.assertRaises(PlanValidationError) as ctx:
            validate_plan(bad)
        self.assertIn("extra", str(ctx.exception))

    def test_empty_strategy_rejected(self) -> None:
        bad = _good_plan()
        bad["strategy"] = "   "
        with self.assertRaises(PlanValidationError):
            validate_plan(bad)

    def test_empty_decisions_rejected(self) -> None:
        bad = _good_plan()
        bad["decisions"] = []
        with self.assertRaises(PlanValidationError):
            validate_plan(bad)

    def test_decisions_must_be_list(self) -> None:
        bad = _good_plan()
        bad["decisions"] = "FORWARD"
        with self.assertRaises(PlanValidationError):
            validate_plan(bad)

    def test_decision_item_must_be_dict(self) -> None:
        bad = _good_plan()
        bad["decisions"] = ["FORWARD"]
        with self.assertRaises(PlanValidationError):
            validate_plan(bad)

    def test_decision_item_extra_keys_rejected(self) -> None:
        bad = _good_plan()
        bad["decisions"] = [{"condition": "x", "action": "FORWARD", "weight": 0.5}]
        with self.assertRaises(PlanValidationError) as ctx:
            validate_plan(bad)
        self.assertIn("weight", str(ctx.exception))

    def test_decision_action_unknown_rejected(self) -> None:
        bad = _good_plan()
        bad["decisions"][0]["action"] = "BACKWARD"
        with self.assertRaises(PlanValidationError) as ctx:
            validate_plan(bad)
        self.assertIn("BACKWARD", str(ctx.exception))

    def test_decision_condition_empty_rejected(self) -> None:
        bad = _good_plan()
        bad["decisions"][0]["condition"] = ""
        with self.assertRaises(PlanValidationError):
            validate_plan(bad)

    def test_stop_condition_empty_rejected(self) -> None:
        bad = _good_plan()
        bad["stop_condition"] = ""
        with self.assertRaises(PlanValidationError):
            validate_plan(bad)

    def test_does_not_mutate_input(self) -> None:
        plan = _good_plan()
        snapshot = json.loads(json.dumps(plan))
        with self.assertRaises(PlanValidationError):
            validate_plan({**plan, "extra": 1})
        self.assertEqual(plan, snapshot)


class RunPlannerTests(unittest.TestCase):
    def test_returns_validated_plan(self) -> None:
        with mock_opener(respond=lambda _p: _envelope(json.dumps(_good_plan()))):
            plan = run_planner(GOOD_REQUIREMENTS)
        self.assertEqual(plan, _good_plan())

    def test_uses_qwen3_8b_stream_false_think_false_and_schema(self) -> None:
        with mock_opener(respond=lambda _p: _envelope(json.dumps(_good_plan()))) as captured:
            run_planner(GOOD_REQUIREMENTS)
        payload = captured["payload"]
        self.assertEqual(payload["model"], "qwen3:8b")
        self.assertEqual(payload["stream"], False)
        self.assertEqual(payload["think"], False)
        # Schema round-tripped through JSON: assert structural essentials instead of
        # full dict equality (which can fail on cosmetic key ordering).
        schema = payload["format"]
        self.assertEqual(schema["type"], "object")
        self.assertEqual(set(schema["required"]), {"strategy", "decisions", "stop_condition"})
        self.assertEqual(schema["additionalProperties"], False)
        self.assertEqual(payload["options"]["temperature"], 0.2)
        self.assertEqual(payload["options"]["num_predict"], 1024)

    def test_sends_only_requirements_not_brief(self) -> None:
        # Context isolation: only the validated requirements dict are sent.
        user_contents: list[str] = []

        def respond(payload):
            user_contents.append(payload["messages"][1]["content"])
            return _envelope(json.dumps(_good_plan()))

        with mock_opener(respond=respond):
            run_planner(GOOD_REQUIREMENTS)
        sent = user_contents[0]
        self.assertIn("goal", sent)
        self.assertIn("safe_stop", sent)
        self.assertIn("avoid_obstacles", sent)
        # The user message is the requirements as JSON; no "ROBOT NAVIGATION" prose.
        self.assertNotIn("ROBOT NAVIGATION", sent)

    def test_first_invalid_then_valid_retries(self) -> None:
        responses = iter([
            _envelope("```json\n" + json.dumps(_good_plan()) + "\n```"),
            _envelope(json.dumps(_good_plan())),
        ])
        attempts = {"n": 0}

        def respond(_payload):
            attempts["n"] += 1
            return next(responses)

        with mock_opener(respond=respond):
            plan = run_planner(GOOD_REQUIREMENTS)
        self.assertEqual(plan, _good_plan())
        self.assertEqual(attempts["n"], 2)

    def test_two_invalid_attempts_raises(self) -> None:
        with mock_opener(respond=lambda _p: _envelope("not json")):
            with self.assertRaises(PlanValidationError) as ctx:
                run_planner(GOOD_REQUIREMENTS)
        self.assertIn("twice", str(ctx.exception))

    def test_network_error_does_not_return_canned_plan(self) -> None:
        with mock_opener(raise_exception=URLError("offline")):
            with self.assertRaises(OllamaError):
                run_planner(GOOD_REQUIREMENTS)

    def test_duplicate_key_in_response_rejected(self) -> None:
        dup = (
            '{"strategy": "s", "strategy": "t", '
            '"decisions": [{"condition": "c", "action": "FORWARD"}], '
            '"stop_condition": "stop"}'
        )
        with mock_opener(respond=lambda _p: _envelope(dup)):
            with self.assertRaises(PlanValidationError) as ctx:
                run_planner(GOOD_REQUIREMENTS)
        self.assertIn("Duplicate JSON key", str(ctx.exception))

    def test_non_dict_requirement_rejected_before_http(self) -> None:
        with self.assertRaises(ValueError):
            run_planner("not a dict")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()