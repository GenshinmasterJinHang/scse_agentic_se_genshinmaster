"""Convert the Planner's validated plan into Python navigation code with Qwen.

The Developer Agent receives the validated plan dict (context isolation) and
emits a JSON object containing a description plus a ``code`` field. The
``code`` field must be syntactically valid Python that defines at least one
function related to robot navigation. The runner saves that string to
``navigation_logic.py`` for the simulator to import.

Reuses ``call_qwen`` and ``OllamaError`` from ``analyst_agent``.
"""

from __future__ import annotations

import ast
import json
import keyword
from typing import Any

from analyst_agent import OllamaError, call_qwen


REQUIRED_CODE_KEYS = {"description", "code"}
NAV_HINTS = (
    "navigation",
    "navigate",
    "decide",
    "act",
    "action",
    "move",
    "step",
    "plan",
)


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
- The code must define at least one top-level function whose name relates to
  robot navigation (e.g. decide_action, navigation_step, choose_move, ...).
- The function may take whatever sensor/state arguments you need; the planner
  did not fix them. Use plain types (numbers, strings, tuples, dicts) -- do not
  depend on third-party packages.
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
    """Return (ok, function_name) for the first navigation-looking function.

    The Developer Agent is judged on architecture (does the code parse? does it
    expose a navigation function?), not on whether the function is semantically
    correct -- that is tested by the simulator.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"code is not valid Python: {exc}"

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            if keyword.iskeyword(name):
                return False, f"function name {name!r} is a Python keyword"
            lname = name.lower()
            if any(hint in lname for hint in NAV_HINTS):
                return True, name
    return False, "code does not define a top-level function related to navigation"


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