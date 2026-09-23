"""Convert the Analyst's validated requirements into a navigation plan with Qwen.

Context isolation: this agent receives ONLY the validated requirements dict, never
the Analyst conversation. The Planner's output contract is a fixed JSON object
with three required keys; Python re-validates before persisting plan.json.

Reuses ``call_qwen`` and ``OllamaError`` from ``analyst_agent`` so all three
agents speak to Ollama through one HTTP layer.
"""

from __future__ import annotations

import json
from typing import Any

from analyst_agent import OllamaError, call_qwen


ALLOWED_PLAN_ACTIONS = ("FORWARD", "LEFT", "RIGHT", "STOP")
REQUIRED_PLAN_KEYS = {"strategy", "decisions", "stop_condition"}


PLANNER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "strategy": {"type": "string", "minLength": 1},
        "decisions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string", "minLength": 1},
                    "action": {"type": "string", "enum": list(ALLOWED_PLAN_ACTIONS)},
                },
                "required": ["condition", "action"],
                "additionalProperties": False,
            },
        },
        "stop_condition": {"type": "string", "minLength": 1},
    },
    "required": ["strategy", "decisions", "stop_condition"],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """You are a software planner for a mobile robot.
You receive the validated requirements produced by the Analyst Agent and turn
them into an actionable navigation plan.

Treat the requirements as the only source of truth. Do not invent extra sensors,
extra actions, or a tie-breaking order that the brief did not specify. If the
brief is silent on a case, write the corresponding decision as "no safe action"
or skip it -- never add rules the requirements do not justify.

Return exactly one JSON object with these three keys and no others:
- strategy: a nonempty English string describing the overall navigation
  strategy in plain prose.
- decisions: a non-empty array of objects. Each object has exactly two keys:
    * condition: a nonempty English string describing when this decision fires.
    * action: one of "FORWARD", "LEFT", "RIGHT", "STOP".
  Decisions must cover (1) safe_progress, (2) prefer_goal_direction when safe,
  and (3) stop when no direction is safe. Do not list duplicate conditions.
- stop_condition: a nonempty English string stating exactly when the robot
  must issue STOP.

Do not invent extra keys, markdown fences, code blocks, or thinking prose.
Return JSON only. The JSON schema is:
""" + json.dumps(PLANNER_SCHEMA)


class PlanValidationError(ValueError):
    """Qwen's plan is not acceptable under the Planner Agent contract."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys (mirrors analyst_agent)."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PlanValidationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise PlanValidationError(f"Non-standard JSON constant: {value}")


def validate_plan(data: Any) -> dict[str, Any]:
    """Return the same object if valid; otherwise raise without repairing.

    Schema-level checks only. Does not prove the plan is semantically correct.
    """
    if not isinstance(data, dict):
        raise PlanValidationError("Plan must be a dictionary.")
    keys = set(data)
    if keys != REQUIRED_PLAN_KEYS:
        missing = REQUIRED_PLAN_KEYS - keys
        extra = keys - REQUIRED_PLAN_KEYS
        raise PlanValidationError(f"Wrong keys. Missing: {missing}; extra: {extra}")

    if not isinstance(data["strategy"], str) or not data["strategy"].strip():
        raise PlanValidationError("strategy must be a nonempty string.")

    decisions = data["decisions"]
    if not isinstance(decisions, list):
        raise PlanValidationError("decisions must be a list.")
    if not decisions:
        raise PlanValidationError("decisions must contain at least one entry.")
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            raise PlanValidationError(f"decisions[{index}] must be a dictionary.")
        if set(decision) != {"condition", "action"}:
            extra = set(decision) - {"condition", "action"}
            missing = {"condition", "action"} - set(decision)
            raise PlanValidationError(
                f"decisions[{index}] must have exactly the keys condition and action; "
                f"extra={sorted(extra)}, missing={sorted(missing)}."
            )
        if not isinstance(decision["condition"], str) or not decision["condition"].strip():
            raise PlanValidationError(f"decisions[{index}].condition must be a nonempty string.")
        if decision["action"] not in ALLOWED_PLAN_ACTIONS:
            raise PlanValidationError(
                f"decisions[{index}].action must be one of {ALLOWED_PLAN_ACTIONS}, "
                f"got {decision['action']!r}."
            )

    if not isinstance(data["stop_condition"], str) or not data["stop_condition"].strip():
        raise PlanValidationError("stop_condition must be a nonempty string.")
    return data


def run_planner(requirement: Any) -> dict[str, Any]:
    """Ask Qwen, parse JSON, validate, and return the validated plan.

    ``requirement`` must be the validated requirements dict produced by
    ``run_analyst``. Only that dict is sent to Qwen -- not the original brief.
    """
    if not isinstance(requirement, dict):
        raise ValueError("requirement must be the validated requirements dictionary.")
    # Context isolation: serialize only the validated requirements object.
    user_payload = json.dumps(requirement, ensure_ascii=False, sort_keys=True)

    feedback = ""
    for attempt in range(2):
        raw_text = call_qwen(
            SYSTEM_PROMPT,
            "Plan the navigation for these validated requirements:\n<requirements>\n"
            + user_payload
            + "\n</requirements>"
            + feedback,
            structured=True,
            temperature=0.2,
            schema=PLANNER_SCHEMA,
        )
        try:
            data = json.loads(
                raw_text,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
            return validate_plan(data)
        except (json.JSONDecodeError, PlanValidationError) as exc:
            if attempt == 1:
                raise PlanValidationError(
                    f"Qwen returned an invalid plan twice. Last error: {exc}"
                ) from exc
            feedback = (
                "\nYour previous response failed validation: " + str(exc)
                + "\nRead the requirements again and return a new, complete JSON object."
            )
    raise AssertionError("Unreachable")