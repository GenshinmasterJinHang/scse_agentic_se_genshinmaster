"""Validation rules for the Stage B JSON contract.

These tests cover the schema-level checks performed by ``validate_requirements``
without ever touching the model or the network.
"""

from __future__ import annotations

import unittest

from analyst_agent import (
    ALLOWED_ACTIONS,
    RequirementsValidationError,
    validate_requirements,
)


def _legal() -> dict:
    return {
        "goal": "navigate toward the goal when safe",
        "allowed_actions": ["FORWARD", "LEFT", "RIGHT", "STOP"],
        "safe_stop": True,
        "avoid_obstacles": True,
    }


class ValidateRequirementsTests(unittest.TestCase):
    def test_legal_dict_returns_unchanged(self) -> None:
        data = _legal()
        result = validate_requirements(data)
        self.assertIs(result, data)

    def test_non_dict_rejected(self) -> None:
        with self.assertRaises(RequirementsValidationError):
            validate_requirements(["not", "a", "dict"])  # type: ignore[arg-type]

    def test_missing_field_rejected(self) -> None:
        bad = _legal()
        bad.pop("safe_stop")
        with self.assertRaises(RequirementsValidationError) as ctx:
            validate_requirements(bad)
        self.assertIn("safe_stop", str(ctx.exception))

    def test_extra_field_rejected(self) -> None:
        bad = {**_legal(), "reasoning": "extra"}
        with self.assertRaises(RequirementsValidationError) as ctx:
            validate_requirements(bad)
        self.assertIn("reasoning", str(ctx.exception))

    def test_empty_goal_rejected(self) -> None:
        bad = _legal()
        bad["goal"] = ""
        with self.assertRaises(RequirementsValidationError):
            validate_requirements(bad)

    def test_whitespace_only_goal_rejected(self) -> None:
        bad = _legal()
        bad["goal"] = "   \n\t  "
        with self.assertRaises(RequirementsValidationError):
            validate_requirements(bad)

    def test_non_string_goal_rejected(self) -> None:
        bad = _legal()
        bad["goal"] = 123
        with self.assertRaises(RequirementsValidationError):
            validate_requirements(bad)

    def test_actions_not_list_rejected(self) -> None:
        bad = _legal()
        bad["allowed_actions"] = "FORWARD"
        with self.assertRaises(RequirementsValidationError):
            validate_requirements(bad)

    def test_actions_with_non_string_element_rejected(self) -> None:
        bad = _legal()
        bad["allowed_actions"] = ["FORWARD", "LEFT", 1, "STOP"]
        with self.assertRaises(RequirementsValidationError):
            validate_requirements(bad)

    def test_actions_unknown_value_rejected(self) -> None:
        bad = _legal()
        bad["allowed_actions"] = ["FORWARD", "BACKWARD", "LEFT", "STOP"]
        with self.assertRaises(RequirementsValidationError):
            validate_requirements(bad)

    def test_actions_duplicate_rejected(self) -> None:
        bad = _legal()
        bad["allowed_actions"] = ["FORWARD", "FORWARD", "LEFT", "STOP"]
        with self.assertRaises(RequirementsValidationError):
            validate_requirements(bad)

    def test_actions_missing_one_rejected(self) -> None:
        bad = _legal()
        bad["allowed_actions"] = ["FORWARD", "LEFT", "RIGHT"]
        with self.assertRaises(RequirementsValidationError):
            validate_requirements(bad)

    def test_string_true_does_not_count_as_bool(self) -> None:
        bad = _legal()
        bad["safe_stop"] = "true"
        with self.assertRaises(RequirementsValidationError):
            validate_requirements(bad)

    def test_int_one_does_not_count_as_bool(self) -> None:
        bad = _legal()
        bad["avoid_obstacles"] = 1
        with self.assertRaises(RequirementsValidationError):
            validate_requirements(bad)

    def test_none_does_not_count_as_bool(self) -> None:
        bad = _legal()
        bad["safe_stop"] = None
        with self.assertRaises(RequirementsValidationError):
            validate_requirements(bad)

    def test_safe_stop_false_rejected_for_this_brief(self) -> None:
        bad = _legal()
        bad["safe_stop"] = False
        with self.assertRaises(RequirementsValidationError):
            validate_requirements(bad)

    def test_avoid_obstacles_false_rejected_for_this_brief(self) -> None:
        bad = _legal()
        bad["avoid_obstacles"] = False
        with self.assertRaises(RequirementsValidationError):
            validate_requirements(bad)

    def test_does_not_mutate_input(self) -> None:
        original = _legal()
        snapshot = {k: (list(v) if isinstance(v, list) else v) for k, v in original.items()}
        with self.assertRaises(RequirementsValidationError):
            validate_requirements({**original, "extra_field": "x"})
        self.assertEqual(original, snapshot)


class AllowedActionsConstantTests(unittest.TestCase):
    def test_constant_includes_four_required_actions(self) -> None:
        self.assertEqual(set(ALLOWED_ACTIONS), {"FORWARD", "LEFT", "RIGHT", "STOP"})
        self.assertEqual(len(ALLOWED_ACTIONS), 4)


if __name__ == "__main__":
    unittest.main()
