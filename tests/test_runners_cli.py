"""Tests for run_planner.py and run_developer.py CLIs.

Pattern mirrors test_run_analyst_cli.py: mock the agent inside a wrapper
subprocess so ``patch(...)`` takes effect.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


GOOD_REQUIREMENTS = {
    "goal": "navigate toward the goal when safe",
    "allowed_actions": ["FORWARD", "LEFT", "RIGHT", "STOP"],
    "safe_stop": True,
    "avoid_obstacles": True,
}


GOOD_PLAN = {
    "strategy": "Prefer goal direction when safe; stop when no direction is safe.",
    "decisions": [
        {"condition": "goal ahead and forward is clear", "action": "FORWARD"},
        {"condition": "no safe direction", "action": "STOP"},
    ],
    "stop_condition": "When forward, left, and right are all blocked.",
}


GOOD_DEVELOPER = {
    "description": "Decides which of FORWARD/LEFT/RIGHT/STOP to take.",
    "code": "def decide_action():\n    return 'STOP'\n",
}


def _run_with_patch(
    program: str,
    argv_after_program: list[str],
    *,
    patch_target: str,
    patch_body: str,
    cwd: str | None = None,
) -> subprocess.CompletedProcess:
    repo_root = str(REPO_ROOT)
    wrapper = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {repo_root!r})
        from unittest.mock import patch
        sys.argv = {argv_after_program!r}
        with patch({patch_target!r}, {patch_body}):
            import {program}
            sys.exit({program}.main())
        """
    ).strip()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(wrapper)
        wrapper_path = handle.name
    try:
        return subprocess.run(
            [sys.executable, wrapper_path],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=30,
        )
    finally:
        Path(wrapper_path).unlink(missing_ok=True)


class RunPlannerCliTests(unittest.TestCase):
    def test_default_paths_relative_to_script_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            req_file = tmp_path / "req.json"
            req_file.write_text(json.dumps(GOOD_REQUIREMENTS), encoding="utf-8")
            plan_file = tmp_path / "plan.json"
            proc = _run_with_patch(
                "run_planner",
                [
                    "run_planner.py",
                    "--requirements",
                    str(req_file),
                    "--output",
                    str(plan_file),
                ],
                patch_target="run_planner.run_planner",
                patch_body=f"return_value={GOOD_PLAN!r}",
                cwd=str(tmp_path),
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(plan_file.exists())
            self.assertEqual(json.loads(plan_file.read_text(encoding="utf-8")), GOOD_PLAN)

    def test_creates_missing_parent_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            req_file = tmp_path / "req.json"
            req_file.write_text(json.dumps(GOOD_REQUIREMENTS), encoding="utf-8")
            plan_file = tmp_path / "deep" / "nested" / "plan.json"
            proc = _run_with_patch(
                "run_planner",
                [
                    "run_planner.py",
                    "--requirements",
                    str(req_file),
                    "--output",
                    str(plan_file),
                ],
                patch_target="run_planner.run_planner",
                patch_body=f"return_value={GOOD_PLAN!r}",
                cwd=str(tmp_path),
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(plan_file.exists())

    def test_failure_does_not_overwrite_existing_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            req_file = tmp_path / "req.json"
            req_file.write_text(json.dumps(GOOD_REQUIREMENTS), encoding="utf-8")
            plan_file = tmp_path / "plan.json"
            plan_file.write_text(json.dumps({"old": True}), encoding="utf-8")
            proc = _run_with_patch(
                "run_planner",
                [
                    "run_planner.py",
                    "--requirements",
                    str(req_file),
                    "--output",
                    str(plan_file),
                ],
                patch_target="run_planner.run_planner",
                patch_body='side_effect=RuntimeError("injected")',
                cwd=str(tmp_path),
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("No new result was generated", proc.stderr)
            self.assertEqual(
                json.loads(plan_file.read_text(encoding="utf-8")), {"old": True}
            )

    def test_requirements_must_be_a_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            req_file = tmp_path / "req.json"
            req_file.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
            plan_file = tmp_path / "plan.json"
            proc = _run_with_patch(
                "run_planner",
                [
                    "run_planner.py",
                    "--requirements",
                    str(req_file),
                    "--output",
                    str(plan_file),
                ],
                patch_target="run_planner.run_planner",
                patch_body="return_value={}",
                cwd=str(tmp_path),
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertFalse(plan_file.exists())

    def test_requirements_keys_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            req_file = tmp_path / "req.json"
            req_file.write_text(json.dumps({"goal": "x"}), encoding="utf-8")  # missing fields
            plan_file = tmp_path / "plan.json"
            proc = _run_with_patch(
                "run_planner",
                [
                    "run_planner.py",
                    "--requirements",
                    str(req_file),
                    "--output",
                    str(plan_file),
                ],
                patch_target="run_planner.run_planner",
                patch_body="return_value={}",
                cwd=str(tmp_path),
            )
            self.assertNotEqual(proc.returncode, 0)


class RunDeveloperCliTests(unittest.TestCase):
    def test_writes_module_and_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plan_file = tmp_path / "plan.json"
            plan_file.write_text(json.dumps(GOOD_PLAN), encoding="utf-8")
            module_file = tmp_path / "nav.py"
            envelope_file = tmp_path / "env.json"
            proc = _run_with_patch(
                "run_developer",
                [
                    "run_developer.py",
                    "--plan",
                    str(plan_file),
                    "--module-output",
                    str(module_file),
                    "--envelope-output",
                    str(envelope_file),
                ],
                patch_target="run_developer.run_developer",
                patch_body=f"return_value={GOOD_DEVELOPER!r}",
                cwd=str(tmp_path),
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(module_file.exists())
            self.assertTrue(envelope_file.exists())
            # The Python file contains the code exactly as Qwen wrote it.
            self.assertIn("def decide_action", module_file.read_text(encoding="utf-8"))
            on_disk = json.loads(envelope_file.read_text(encoding="utf-8"))
            self.assertEqual(on_disk, GOOD_DEVELOPER)

    def test_failure_does_not_overwrite_existing_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plan_file = tmp_path / "plan.json"
            plan_file.write_text(json.dumps(GOOD_PLAN), encoding="utf-8")
            module_file = tmp_path / "nav.py"
            module_file.write_text("# KEEP_ME\n", encoding="utf-8")
            envelope_file = tmp_path / "env.json"
            proc = _run_with_patch(
                "run_developer",
                [
                    "run_developer.py",
                    "--plan",
                    str(plan_file),
                    "--module-output",
                    str(module_file),
                    "--envelope-output",
                    str(envelope_file),
                ],
                patch_target="run_developer.run_developer",
                patch_body='side_effect=RuntimeError("injected")',
                cwd=str(tmp_path),
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("No new result was generated", proc.stderr)
            self.assertEqual(module_file.read_text(encoding="utf-8"), "# KEEP_ME\n")
            self.assertFalse(envelope_file.exists())

    def test_plan_must_be_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plan_file = tmp_path / "plan.json"
            plan_file.write_text(json.dumps(["nope"]), encoding="utf-8")
            proc = _run_with_patch(
                "run_developer",
                [
                    "run_developer.py",
                    "--plan",
                    str(plan_file),
                ],
                patch_target="run_developer.run_developer",
                patch_body="return_value={}",
                cwd=str(tmp_path),
            )
            self.assertNotEqual(proc.returncode, 0)

    def test_plan_keys_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plan_file = tmp_path / "plan.json"
            plan_file.write_text(json.dumps({"strategy": "only"}), encoding="utf-8")
            proc = _run_with_patch(
                "run_developer",
                [
                    "run_developer.py",
                    "--plan",
                    str(plan_file),
                ],
                patch_target="run_developer.run_developer",
                patch_body="return_value={}",
                cwd=str(tmp_path),
            )
            self.assertNotEqual(proc.returncode, 0)


class AtomicWriteTests(unittest.TestCase):
    """save_plan, save_envelope, write_python_module must write atomically."""

    def test_save_plan_atomic(self) -> None:
        from run_planner import save_plan

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "p.json"
            save_plan({"a": 1}, out)
            self.assertTrue(out.exists())
            leftovers = list(out.parent.glob(".plan_*"))
            self.assertEqual(leftovers, [])

    def test_save_envelope_atomic(self) -> None:
        from run_developer import save_envelope

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "e.json"
            save_envelope({"a": 1}, out)
            leftovers = list(out.parent.glob(".developer_*"))
            self.assertEqual(leftovers, [])

    def test_write_python_module_atomic(self) -> None:
        from run_developer import write_python_module

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "n.py"
            write_python_module("def decide_action(): pass\n", out)
            self.assertTrue(out.exists())
            leftovers = list(out.parent.glob(".navigation_*"))
            self.assertEqual(leftovers, [])
            self.assertEqual(out.read_text(encoding="utf-8"), "def decide_action(): pass\n")


if __name__ == "__main__":
    unittest.main()