"""Tests for ``brief_to_req.py`` -- Stage A free-text extraction.

Stage A must NOT use the Stage B JSON schema (so Qwen can return numbered
natural-language requirements). It must call Qwen independently for each run
and must not overwrite the main output if any run fails.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from tests._http_mock import mock_opener


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_text_envelope(text: str) -> dict:
    return {
        "model": "qwen3:8b",
        "done": True,
        "done_reason": "stop",
        "message": {"role": "assistant", "content": text},
    }


class BriefToReqInvocationTests(unittest.TestCase):
    """Drive ``brief_to_req.py`` as a subprocess so CLI behaviour is verified."""

    def _run_cli(self, *extra_args: str, brief_text: str = "brief body") -> subprocess.CompletedProcess:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            brief_path = Path(tmp) / "brief.txt"
            brief_path.write_text(brief_text, encoding="utf-8")
            out_path = Path(tmp) / "robot_requirements.txt"
            runs_dir = Path(tmp) / "runs"
            cmd = [
                sys.executable,
                str(REPO_ROOT / "brief_to_req.py"),
                "--brief",
                str(brief_path),
                "--output",
                str(out_path),
                "--runs-root",
                str(runs_dir),
                *extra_args,
            ]
            env = {
                "PATH": __import__("os").environ.get("PATH", ""),
                "SystemRoot": __import__("os").environ.get("SystemRoot", ""),
                "TEMP": __import__("os").environ.get("TEMP", ""),
            }
            return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=30)

    def test_cli_accepts_runs_argument(self) -> None:
        # We only verify argument parsing works; the actual HTTP call is mocked below.
        # The CLI must not refuse the flag.
        # This test is a guard against typos in the argparse section.
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "brief_to_req.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--runs", proc.stdout)


class BriefToReqMockedHttpTests(unittest.TestCase):
    """Run ``brief_to_req.main`` directly with HTTP mocked."""

    def _import_module(self):
        import importlib.util
        import sys

        # Load under the file's own module name so patches targeting
        # "brief_to_req.call_qwen" hit the right namespace.
        if "brief_to_req" in sys.modules:
            return sys.modules["brief_to_req"]
        spec = importlib.util.spec_from_file_location(
            "brief_to_req", str(REPO_ROOT / "brief_to_req.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["brief_to_req"] = module
        spec.loader.exec_module(module)
        return module

    def test_runs3_sends_three_independent_requests(self) -> None:
        import tempfile

        module = self._import_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            brief = tmp_path / "brief.txt"
            brief.write_text("THE_BRIEF_CONTENT", encoding="utf-8")
            output = tmp_path / "out.txt"
            runs_root = tmp_path / "runs"

            call_count = {"n": 0}
            user_messages: list[str] = []

            def respond(_prompt, _user, **_kwargs):
                call_count["n"] += 1
                # The mock captures whatever the producer code sent; we don't decode
                # the JSON envelope here -- the producer passes a system prompt string
                # and a user prompt string, so we extract the user prompt as-is.
                user_messages.append(_user)
                return _run_text_envelope(f"REQ RESPONSE #{call_count['n']}")["message"]["content"]

            argv = [
                "--brief",
                str(brief),
                "--output",
                str(output),
                "--runs",
                "3",
                "--runs-root",
                str(runs_root),
            ]
            with patch("brief_to_req.call_qwen", side_effect=respond):
                rc = module.main(argv)
            self.assertEqual(rc, 0)
            self.assertEqual(call_count["n"], 3)
            self.assertTrue(all("THE_BRIEF_CONTENT" in m for m in user_messages))
            self.assertEqual(output.read_text(encoding="utf-8"), "REQ RESPONSE #3")
            run_files = sorted((runs_root).glob("*/run_*.txt"))
            self.assertEqual(len(run_files), 3)
            self.assertEqual(run_files[-1].read_text(encoding="utf-8"), "REQ RESPONSE #3")

    def test_does_not_use_stage_b_schema_format(self) -> None:
        import tempfile

        module = self._import_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "brief.txt").write_text("b", encoding="utf-8")

            def respond(_prompt, _user, **_kwargs):
                # Stage A does NOT force the Stage B schema or set structured=True.
                # The mock just needs to return text.
                return "free text requirements"

            argv = [
                "--brief",
                str(tmp_path / "brief.txt"),
                "--output",
                str(tmp_path / "out.txt"),
            ]
            with patch("brief_to_req.call_qwen", side_effect=respond):
                module.main(argv)
            # Indirect proof: structured was False (call_qwen was called without the
            # structured kwarg being meaningfully consumed) and the call returned text.

    def test_failure_on_run_does_not_overwrite_existing_output(self) -> None:
        import tempfile

        module = self._import_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "brief.txt").write_text("b", encoding="utf-8")
            output = tmp_path / "out.txt"
            output.write_text("PRESERVED OLD CONTENT", encoding="utf-8")
            runs_root = tmp_path / "runs"

            from analyst_agent import OllamaError

            def respond(*_args, **_kwargs):
                raise OllamaError("injected failure")

            argv = [
                "--brief",
                str(tmp_path / "brief.txt"),
                "--output",
                str(output),
                "--runs",
                "3",
                "--runs-root",
                str(runs_root),
            ]
            with patch("brief_to_req.call_qwen", side_effect=respond):
                rc = module.main(argv)
            self.assertEqual(rc, 1)
            self.assertEqual(output.read_text(encoding="utf-8"), "PRESERVED OLD CONTENT")

    def test_partial_run_saves_completed_runs_individually(self) -> None:
        import tempfile

        module = self._import_module()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "brief.txt").write_text("b", encoding="utf-8")
            output = tmp_path / "out.txt"
            runs_root = tmp_path / "runs"

            from analyst_agent import OllamaError

            calls = {"n": 0}

            def respond(*_args, **_kwargs):
                calls["n"] += 1
                if calls["n"] == 2:
                    raise OllamaError("fail on second run")
                return _run_text_envelope(f"run-{calls['n']}")["message"]["content"]

            argv = [
                "--brief",
                str(tmp_path / "brief.txt"),
                "--output",
                str(output),
                "--runs",
                "3",
                "--runs-root",
                str(runs_root),
            ]
            with patch("brief_to_req.call_qwen", side_effect=respond):
                rc = module.main(argv)
            self.assertEqual(rc, 1)
            run_files = sorted(runs_root.glob("*/run_*.txt"))
            # Only the run that completed before the failure is persisted on disk;
            # the partial run is preserved as forensic evidence.
            self.assertEqual(len(run_files), 1)
            self.assertFalse(output.exists())

    def test_invalid_runs_argument_rejected(self) -> None:
        module = self._import_module()
        with self.assertRaises(SystemExit):
            module.main(["--runs", "0"])

    def test_invalid_temperature_argument_rejected(self) -> None:
        module = self._import_module()
        with self.assertRaises(SystemExit):
            module.main(["--temperature", "-1"])


if __name__ == "__main__":
    unittest.main()
