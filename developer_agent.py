"""Convert the Planner's validated plan into Python navigation code with Qwen.

The Developer Agent receives the validated plan dict (context isolation) and
emits a JSON object containing a description plus a ``code`` field. The
``code`` field must be syntactically valid Python that defines exactly one
top-level function named ``decide_next_move(state)``. The runner saves that
string to ``navigation_logic.py`` for the simulator to import.

Reuses ``call_qwen`` and ``OllamaError`` from ``analyst_agent``.
"""

from __future__ import annotations

import ast
import json
from typing import Any

from analyst_agent import OllamaError, call_qwen


REQUIRED_CODE_KEYS = {"description", "code"}
REQUIRED_ENTRY_FUNCTION = "decide_next_move"
ALLOWED_ACTIONS = ("FORWARD", "LEFT", "RIGHT", "STOP")


DEVELOPER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "description": {"type": "string", "minLength": 1},
        "code": {"type": "string", "minLength": 1},
    },
    "required": ["description", "code"],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """You are a software developer for a mobile robot.
You receive the Planner's validated navigation plan and translate it into a
small, self-contained Python module that a simulator can import.

Constraints:
- Output exactly one JSON object with two keys and no others:
    * description: a nonempty English string explaining what the function does.
    * code: a nonempty Python source string that compiles cleanly.
- The code MUST define exactly one top-level function named
  ``decide_next_move`` with exactly one parameter named ``state``. The
  simulator only calls this entry point -- do not invent other names
  (decide_action, navigation_step, choose_move, ...). Helper functions are
  allowed but the entry point must be ``decide_next_move(state)``.
- ``state`` is the sensor/goal snapshot the robot receives; you may read any
  keys out of it (use ``state.get(key, default)`` for safety). Use plain
  Python types (numbers, strings, tuples, dicts) -- do not depend on
  third-party packages.
- The function must return one of the strings ``"FORWARD"``, ``"LEFT"``,
  ``"RIGHT"``, or ``"STOP"``.
- Do not include imports beyond the Python standard library. Do not write to
  files or print to stdout. Do not include markdown fences or any prose.
- The code must reflect the Planner's strategy and decisions verbatim. Do not
  invent extra logic the plan did not specify.

The JSON schema is:
""" + json.dumps(DEVELOPER_SCHEMA)


class DeveloperValidationError(ValueError):
    """Qwen's developer output is not acceptable under the contract."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DeveloperValidationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise DeveloperValidationError(f"Non-standard JSON constant: {value}")


def _looks_like_navigation(code: str) -> tuple[bool, str]:
    """Return (ok, function_name) when ``code`` defines ``decide_next_move(state)``.

    The Developer Agent must define a top-level function named
    ``decide_next_move`` whose single parameter is named ``state``. Other
    helper functions are allowed but the simulator only calls this one entry
    point.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"code is not valid Python: {exc}"

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name != REQUIRED_ENTRY_FUNCTION:
                continue
            args = node.args.args
            if len(args) != 1:
                return False, (
                    f"{REQUIRED_ENTRY_FUNCTION} must take exactly one "
                    f"parameter named 'state'; got {len(args)} parameters."
                )
            if args[0].arg != "state":
                return False, (
                    f"{REQUIRED_ENTRY_FUNCTION} parameter must be named "
                    f"'state'; got {args[0].arg!r}."
                )
            return True, REQUIRED_ENTRY_FUNCTION
    return False, (
        f"code does not define a top-level function named "
        f"{REQUIRED_ENTRY_FUNCTION!r}"
    )


def validate_developer_output(data: Any) -> dict[str, Any]:
    """Validate the JSON envelope and that the code field is parseable Python."""
    if not isinstance(data, dict):
        raise DeveloperValidationError("Developer output must be a dictionary.")
    keys = set(data)
    if keys != REQUIRED_CODE_KEYS:
        missing = REQUIRED_CODE_KEYS - keys
        extra = keys - REQUIRED_CODE_KEYS
        raise DeveloperValidationError(
            f"Wrong keys. Missing: {missing}; extra: {extra}"
        )
    if not isinstance(data["description"], str) or not data["description"].strip():
        raise DeveloperValidationError("description must be a nonempty string.")
    if not isinstance(data["code"], str) or not data["code"].strip():
        raise DeveloperValidationError("code must be a nonempty string.")
    ok, message = _looks_like_navigation(data["code"])
    if not ok:
        raise DeveloperValidationError(message)
    return data


def run_developer(plan: Any) -> dict[str, Any]:
    """Ask Qwen, parse JSON, validate, and return the developer output."""
    if not isinstance(plan, dict):
        raise ValueError("plan must be the validated plan dictionary.")
    # Context isolation: send the plan -- nothing else.
    user_payload = json.dumps(plan, ensure_ascii=False, sort_keys=True)

    feedback = ""
    for attempt in range(2):
        raw_text = call_qwen(
            SYSTEM_PROMPT,
            "Generate Python navigation code for this plan:\n<plan>\n"
            + user_payload
            + "\n</plan>"
            + feedback,
            structured=True,
            temperature=0.2,
            schema=DEVELOPER_SCHEMA,
        )
        try:
            data = json.loads(
                raw_text,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
            return validate_developer_output(data)
        except (json.JSONDecodeError, DeveloperValidationError) as exc:
            if attempt == 1:
                raise DeveloperValidationError(
                    f"Qwen returned an invalid developer output twice. Last error: {exc}"
                ) from exc
            feedback = (
                "\nYour previous response failed validation: " + str(exc)
                + "\nRead the plan again and return a new, complete JSON object."
            )
    raise AssertionError("Unreachable")