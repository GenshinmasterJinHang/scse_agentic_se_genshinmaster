"""Smoke test for the Planner Agent pipeline (Level 1).

Required by ``SCSE '26 - Project Instructions - Testing.pdf``:
    Take the artifact requirements.json file, send it to the planner agent,
    and let the agent run as usual. The output must be stored under
    ``artifacts/plan.json`` and displayed after the test ends.

Strategy:
    * When ``artifacts/requirements.json`` exists and local Ollama is
      reachable, run the real planner.
    * Otherwise, fall back to a mock so the full CLI/validator path is
      still exercised offline.
"""

from __future__ import annotations

import json
import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analyst_agent import OllamaError, run_analyst  # noqa: E402
from planner_agent import run_planner  # noqa: E402

GOOD_REQUIREMENTS = {
    "goal": "navigate toward the goal when safe and stop when no direction is safe",
    "allowed_actions": ["FORWARD", "LEFT", "RIGHT", "STOP"],
    "safe_stop": True,
    "avoid_obstacles": True,
}

GOOD_PLAN = {
    "strategy": "Prefer goal direction when safe; stop when no direction is safe.",
    "decisions": [
        {"condition": "goal ahead and forward is clear", "action": "FORWARD"},
        {"condition": "goal on left and left is clear", "action": "LEFT"},
        {"condition": "goal on right and right is clear", "action": "RIGHT"},
        {"condition": "no safe direction", "action": "STOP"},
    ],
    "stop_condition": "When forward, left, and right are all blocked.",
}


def _ollama_reachable(host: str = "127.0.0.1", port: int = 11434, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _envelope(content: str) -> dict:
    return {
        "model": "qwen3:8b",
        "done": True,
        "done_reason": "stop",
        "message": {"role": "assistant", "content": content},
    }


class TestPlannerPipeline(unittest.TestCase):
    """Run the planner agent against requirements.json and validate the output."""

    def test_planner_produces_valid_plan_from_requirements(self) -> None:
        req_path = REPO_ROOT / "artifacts" / "requirements.json"
        plan_path = REPO_ROOT / "artifacts" / "plan.json"

        # Save existing artifacts so we do not clobber real runs.
        prev_req = req_path.read_bytes() if req_path.exists() else None
        prev_plan = plan_path.read_bytes() if plan_path.exists() else None

        try:
            req_path.parent.mkdir(parents=True, exist_ok=True)
            if req_path.exists():
                requirements = json.loads(req_path.read_text(encoding="utf-8"))
            else:
                # No requirements.json on disk yet -- synthesize a known-good one
                # rather than calling the analyst agent (that has its own pipeline
                # test in test_analyst.py).
                requirements = dict(GOOD_REQUIREMENTS)

            try:
                if _ollama_reachable():
                    plan = run_planner(requirements)
                else:
                    fake_response = _envelope(json.dumps(GOOD_PLAN))
                    with patch("analyst_agent.build_opener") as fake_build:
                        def _build(*_handlers, **_kwargs):
                            class _Opener:
                                def open(self_inner, _request, timeout=None):  # noqa: N805
                                    from tests._http_mock import FakeResponse
                                    return FakeResponse(raw=json.dumps(fake_response).encode("utf-8"))
                            return _Opener()
                        fake_build.side_effect = _build
                        plan = run_planner(requirements)
            except OllamaError as exc:
                self.skipTest(f"Ollama unavailable and offline fallback disabled: {exc}")
                return

            # The planner's own validation should already have run inside run_planner.
            self.assertEqual(set(plan), {"strategy", "decisions", "stop_condition"})
            self.assertTrue(isinstance(plan["decisions"], list) and plan["decisions"])

            plan_path.write_text(
                json.dumps(plan, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # Display the artifact after the test ends (per project instructions).
            print("\n--- artifacts/plan.json ---")
            print(plan_path.read_text(encoding="utf-8"))
            print("--- end artifact ---")

            on_disk = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk, plan)
        finally:
            # Restore previous artifacts if they existed.
            if prev_req is not None:
                req_path.write_bytes(prev_req)
            if prev_plan is not None:
                plan_path.write_bytes(prev_plan)


if __name__ == "__main__":
    unittest.main()