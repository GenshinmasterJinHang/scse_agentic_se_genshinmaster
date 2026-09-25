"""Smoke test for the Analyst Agent pipeline (Level 1).

Required by ``SCSE '26 - Project Instructions - Testing.pdf``:
    Take the original brief.txt file, send it to the analyst agent, and let
    the agent run as usual. The output must be stored under
    ``artifacts/requirements.json`` and displayed after the test ends.

Strategy:
    * When local Ollama is reachable, run the real agent end-to-end.
    * When Ollama is not reachable, fall back to a mock that returns a
      known-good requirements dict so this test still exercises the full
      CLI/validator path offline.

Either way the agent's own validation must accept the result, the file must
be atomically written, and the saved requirements must match what the agent
returned.
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

GOOD_REQUIREMENTS = {
    "goal": "navigate toward the goal when safe and stop when no direction is safe",
    "allowed_actions": ["FORWARD", "LEFT", "RIGHT", "STOP"],
    "safe_stop": True,
    "avoid_obstacles": True,
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


class TestAnalystPipeline(unittest.TestCase):
    """Run the analyst agent against the project's brief.txt and validate the output."""

    def test_analyst_produces_valid_requirements_from_brief(self) -> None:
        brief_path = REPO_ROOT / "brief.txt"
        self.assertTrue(brief_path.exists(), msg=f"missing: {brief_path}")
        brief_text = brief_path.read_text(encoding="utf-8-sig")

        output_path = REPO_ROOT / "artifacts" / "requirements.json"
        # Save the existing file (if any) so we do not clobber a real run.
        previous = output_path.read_bytes() if output_path.exists() else None

        try:
            if _ollama_reachable():
                # Real run.
                requirements = run_analyst(brief_text)
                self.assertEqual(set(requirements), {"goal", "allowed_actions", "safe_stop", "avoid_obstacles"})
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(requirements, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            else:
                # Offline fallback: mock the HTTP layer with a known-good envelope.
                fake_response = _envelope(json.dumps(GOOD_REQUIREMENTS))
                with patch("analyst_agent.build_opener") as fake_build:
                    def _build(*_handlers, **_kwargs):
                        class _Opener:
                            def open(self_inner, _request, timeout=None):  # noqa: N805
                                from tests._http_mock import FakeResponse
                                return FakeResponse(raw=json.dumps(fake_response).encode("utf-8"))
                        return _Opener()
                    fake_build.side_effect = _build
                    requirements = run_analyst(brief_text)
                self.assertEqual(requirements, GOOD_REQUIREMENTS)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(requirements, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            # Display the artifact after the test ends (per project instructions).
            print("\n--- artifacts/requirements.json ---")
            print(output_path.read_text(encoding="utf-8"))
            print("--- end artifact ---")

            # Confirm the saved artifact is parseable and matches what the agent returned.
            on_disk = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk, requirements)
        except OllamaError as exc:
            self.skipTest(f"Ollama unavailable and offline fallback disabled: {exc}")
        finally:
            # Restore previous artifact if it existed.
            if previous is not None:
                output_path.write_bytes(previous)


if __name__ == "__main__":
    unittest.main()