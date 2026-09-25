"""Smoke test for the Developer Agent pipeline (Level 1).

Required by ``SCSE '26 - Project Instructions - Testing.pdf``:
    Take the artifact plan.json file, send it to the developer agent, and
    let the agent run it as usual. The output must be stored under
    ``artifacts/developer_output.json`` and ``navigation_logic.py``, and the
    latter must define ``decide_next_move(state)``.

Strategy:
    * When ``artifacts/plan.json`` exists and local Ollama is reachable, run
      the real developer.
    * Otherwise fall back to a mock that returns a known-good developer
      envelope, so this test still exercises the full validator/CLI path.
"""

from __future__ import annotations

import ast
import json
import socket
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analyst_agent import OllamaError  # noqa: E402
from developer_agent import run_developer  # noqa: E402

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
            if state.get('goal_ahead') and not state.get('front_blocked'):
                return 'FORWARD'
            if state.get('goal_on_left') and not state.get('left_blocked'):
                return 'LEFT'
            if state.get('goal_on_right') and not state.get('right_blocked'):
                return 'RIGHT'
            return 'STOP'
        """
    ).strip()


def _good_envelope() -> dict:
    return {"description": "Decides which of FORWARD/LEFT/RIGHT/STOP to take.", "code": _good_code()}


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


class TestDeveloperPipeline(unittest.TestCase):
    """Run the developer agent against plan.json and validate the output."""

    def test_developer_produces_decide_next_move_module(self) -> None:
        plan_path = REPO_ROOT / "artifacts" / "plan.json"
        envelope_path = REPO_ROOT / "artifacts" / "developer_output.json"
        module_path = REPO_ROOT / "navigation_logic.py"

        prev_plan = plan_path.read_bytes() if plan_path.exists() else None
        prev_envelope = envelope_path.read_bytes() if envelope_path.exists() else None
        prev_module = module_path.read_bytes() if module_path.exists() else None

        try:
            if plan_path.exists():
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
            else:
                plan = dict(GOOD_PLAN)

            try:
                if _ollama_reachable():
                    envelope = run_developer(plan)
                else:
                    fake_response = _envelope(json.dumps(_good_envelope()))
                    with patch("analyst_agent.build_opener") as fake_build:
                        def _build(*_handlers, **_kwargs):
                            class _Opener:
                                def open(self_inner, _request, timeout=None):  # noqa: N805
                                    from tests._http_mock import FakeResponse
                                    return FakeResponse(raw=json.dumps(fake_response).encode("utf-8"))
                            return _Opener()
                        fake_build.side_effect = _build
                        envelope = run_developer(plan)
            except OllamaError as exc:
                self.skipTest(f"Ollama unavailable and offline fallback disabled: {exc}")
                return

            self.assertEqual(set(envelope), {"description", "code"})
            self.assertTrue(envelope["code"].strip())

            # Code must parse and define decide_next_move(state).
            tree = ast.parse(envelope["code"])
            entry = next(
                (n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.name == "decide_next_move"),
                None,
            )
            self.assertIsNotNone(entry, msg="code must define decide_next_move")
            self.assertEqual(len(entry.args.args), 1)
            self.assertEqual(entry.args.args[0].arg, "state")

            envelope_path.parent.mkdir(parents=True, exist_ok=True)
            envelope_path.write_text(
                json.dumps(envelope, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            module_path.write_text(
                envelope["code"] if envelope["code"].endswith("\n") else envelope["code"] + "\n",
                encoding="utf-8",
            )

            # Display artifacts after the test ends (per project instructions).
            print("\n--- artifacts/developer_output.json ---")
            print(envelope_path.read_text(encoding="utf-8"))
            print("\n--- navigation_logic.py ---")
            print(module_path.read_text(encoding="utf-8"))
            print("--- end artifacts ---")

            on_disk_env = json.loads(envelope_path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk_env, envelope)
        finally:
            if prev_plan is not None:
                plan_path.write_bytes(prev_plan)
            if prev_envelope is not None:
                envelope_path.write_bytes(prev_envelope)
            if prev_module is not None:
                module_path.write_bytes(prev_module)


if __name__ == "__main__":
    unittest.main()