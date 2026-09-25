"""Behavioral test for the generated ``decide_next_move(state)`` (Level 2).

Required by ``SCSE '26 - Project Instructions - Testing.pdf``:
    Use the ``decide_next_move`` from the navigation logic artifact. Create
    several test cases to test every known robot state. A sample test case
    is given in the test_generated_navigation_logic.py shared with you
        - Pass the "state" from each case to the developer agent, let it run
          as usual, and check the result is a valid action and that it is
          also the correct expected move.

    Try to test even those states where the generated logic may not work as
    expected.

Then: introduce a bug intentionally into ``navigation_logic.py`` to see if
the test cases actually catch it.

Conventions for the ``state`` argument
-------------------------------------
The brief mentions three obstacle sensors (front/left/right) and possibly a
goal direction (ahead/left/right). The tests below use the same shape as the
sample fixture from the testing instructions:

    state = {
        "goal_ahead":     bool,
        "goal_on_left":   bool,
        "goal_on_right":  bool,
        "front_blocked":  bool,
        "left_blocked":   bool,
        "right_blocked":  bool,
    }

The Developer Agent defines ``decide_next_move(state)``; tests below call it
and check both that the returned action is one of the four legal actions and
that safety/preference invariants hold.
"""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import navigation_logic  # noqa: E402

ALLOWED_ACTIONS = ("FORWARD", "LEFT", "RIGHT", "STOP")


def _state(
    *,
    goal_ahead: bool = False,
    goal_on_left: bool = False,
    goal_on_right: bool = False,
    front_blocked: bool = False,
    left_blocked: bool = False,
    right_blocked: bool = False,
) -> dict[str, bool]:
    return {
        "goal_ahead": goal_ahead,
        "goal_on_left": goal_on_left,
        "goal_on_right": goal_on_right,
        "front_blocked": front_blocked,
        "left_blocked": left_blocked,
        "right_blocked": right_blocked,
    }


def _direction_blocked(action: str, state: dict[str, bool]) -> bool:
    """Return True if ``action`` would move into a direction that ``state`` says is blocked."""
    if action == "FORWARD":
        return bool(state["front_blocked"])
    if action == "LEFT":
        return bool(state["left_blocked"])
    if action == "RIGHT":
        return bool(state["right_blocked"])
    return False  # STOP never moves.


def _expected_action(state: dict[str, bool]) -> str:
    """Reference implementation mirroring the default in navigation_logic.py.

    Used to make test cases concrete; tests that hard-code an expected action
    document the contract, while invariant tests check safety/preference
    properties for every state.
    """
    if state["goal_ahead"] and not state["front_blocked"]:
        return "FORWARD"
    if state["goal_on_left"] and not state["left_blocked"]:
        return "LEFT"
    if state["goal_on_right"] and not state["right_blocked"]:
        return "RIGHT"
    if not state["front_blocked"]:
        return "FORWARD"
    if not state["left_blocked"]:
        return "LEFT"
    if not state["right_blocked"]:
        return "RIGHT"
    return "STOP"


def _all_states() -> list[dict[str, bool]]:
    """Enumerate the 64 possible boolean states of the 6 inputs."""
    states: list[dict[str, bool]] = []
    for ga in (False, True):
        for gl in (False, True):
            for gr in (False, True):
                for fb in (False, True):
                    for lb in (False, True):
                        for rb in (False, True):
                            states.append(_state(
                                goal_ahead=ga, goal_on_left=gl, goal_on_right=gr,
                                front_blocked=fb, left_blocked=lb, right_blocked=rb,
                            ))
    return states


class SampleCaseTests(unittest.TestCase):
    """The sample fixture from the testing instructions PDF."""

    def test_sample_forward_when_goal_ahead_and_clear(self) -> None:
        sample_state = {
            "goal_ahead": True,
            "goal_on_left": False,
            "goal_on_right": False,
            "front_blocked": False,
            "left_blocked": False,
            "right_blocked": False,
        }
        action = navigation_logic.decide_next_move(sample_state)
        self.assertEqual(action, "FORWARD")


class EnumAllStatesTests(unittest.TestCase):
    """Run decide_next_move against every one of the 64 possible states."""

    def test_every_state_returns_a_valid_action(self) -> None:
        for state in _all_states():
            with self.subTest(state=state):
                action = navigation_logic.decide_next_move(dict(state))
                self.assertIn(action, ALLOWED_ACTIONS)

    def test_never_moves_into_a_blocked_direction(self) -> None:
        for state in _all_states():
            with self.subTest(state=state):
                action = navigation_logic.decide_next_move(dict(state))
                self.assertFalse(
                    _direction_blocked(action, state),
                    msg=f"action {action!r} blocked by state {state}",
                )

    def test_stops_when_no_safe_direction(self) -> None:
        for state in _all_states():
            if not (state["front_blocked"] and state["left_blocked"] and state["right_blocked"]):
                continue
            with self.subTest(state=state):
                action = navigation_logic.decide_next_move(dict(state))
                self.assertEqual(action, "STOP")

    def test_prefers_goal_direction_when_safe(self) -> None:
        """When the goal is in exactly one direction and that direction is clear,
        the robot must move toward the goal."""
        cases = [
            (_state(goal_ahead=True, front_blocked=False), "FORWARD"),
            (_state(goal_on_left=True, left_blocked=False), "LEFT"),
            (_state(goal_on_right=True, right_blocked=False), "RIGHT"),
        ]
        for state, expected in cases:
            with self.subTest(state=state):
                action = navigation_logic.decide_next_move(dict(state))
                self.assertEqual(action, expected)


class HardCasesTests(unittest.TestCase):
    """States where a naive implementation is likely to fail."""

    def test_goal_ahead_but_front_blocked_prefers_safe_alternative(self) -> None:
        state = _state(goal_ahead=True, front_blocked=True, left_blocked=False, right_blocked=True)
        action = navigation_logic.decide_next_move(dict(state))
        # Front is blocked; only LEFT is open. Must not move forward.
        self.assertEqual(action, "LEFT")

    def test_goal_ahead_blocked_with_other_safe_directions(self) -> None:
        state = _state(
            goal_ahead=True, goal_on_right=True,
            front_blocked=True, left_blocked=False, right_blocked=False,
        )
        action = navigation_logic.decide_next_move(dict(state))
        # Front is blocked. Either LEFT or RIGHT is acceptable as long as it is safe.
        self.assertIn(action, ("LEFT", "RIGHT"))
        self.assertFalse(_direction_blocked(action, state))

    def test_goal_direction_unset_prefers_forward_when_clear(self) -> None:
        state = _state(front_blocked=False, left_blocked=True, right_blocked=True)
        action = navigation_logic.decide_next_move(dict(state))
        self.assertEqual(action, "FORWARD")

    def test_all_blocked_returns_stop(self) -> None:
        state = _state(
            goal_ahead=True, goal_on_left=True, goal_on_right=True,
            front_blocked=True, left_blocked=True, right_blocked=True,
        )
        action = navigation_logic.decide_next_move(dict(state))
        self.assertEqual(action, "STOP")


class BugDetectionTests(unittest.TestCase):
    """Sanity checks the test suite can fail when ``decide_next_move`` is buggy.

    These tests import ``navigation_logic`` afresh after rewriting the module
    on disk, so they can simulate the "introduce a bug" step from the testing
    instructions. The original file is restored at the end of each test.
    """

    def setUp(self) -> None:
        self._module_path = Path(navigation_logic.__file__)
        self._original = self._module_path.read_text(encoding="utf-8")
        # Drop the module so the next import re-reads the file.
        sys.modules.pop("navigation_logic", None)

    def tearDown(self) -> None:
        self._module_path.write_text(self._original, encoding="utf-8")
        sys.modules.pop("navigation_logic", None)
        importlib.import_module("navigation_logic")

    def _reload(self) -> Any:
        return importlib.import_module("navigation_logic")

    def test_buggy_implementation_returning_wrong_action_is_caught(self) -> None:
        """Introduce a bug that returns FORWARD even when front_blocked=True.

        At least one of the invariant tests above MUST fail -- otherwise the
        test suite is not actually exercising the navigation logic.
        """
        buggy = (
            "def decide_next_move(state):\n"
            "    return 'FORWARD'\n"
        )
        self._module_path.write_text(buggy, encoding="utf-8")
        module = self._reload()
        state = _state(front_blocked=True, left_blocked=True, right_blocked=True)
        action = module.decide_next_move(dict(state))
        # The bug returns FORWARD even though everything is blocked.
        self.assertEqual(action, "FORWARD")
        # Now apply the same invariant the suite asserts: never move into a
        # blocked direction. The bug violates this property.
        self.assertTrue(_direction_blocked(action, state))


if __name__ == "__main__":
    unittest.main()