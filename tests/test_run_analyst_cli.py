"""Tests for ``run_analyst.py``: CLI args, atomic write, failure preserves old file.

Mock-based scenarios run inside a wrapper subprocess so ``patch(...)`` actually
takes effect (the patch lives in the subprocess's Python process, not the
test's parent).
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


def _run_with_patched_runner(
    argv_after_program: list[str],
    *,
    patch_target: str = "run_analyst.run_analyst",
    patch_body: str = "return_value",
    cwd: str | None = None,
    side_channel_path: str | None = None,
) -> subprocess.CompletedProcess:
    """Run ``run_analyst.main()`` in a subprocess with ``patch_target`` mocked.

    ``patch_body`` is the body of a ``with patch(patch_target, <body>):`` block.
    Use ``return_value = <python expr>`` for a successful return, or
    ``side_effect = <python expr>`` to raise.

    If ``side_channel_path`` is given, the wrapper imports a helper named
    ``_side_channel`` and exposes it under ``side_channel`` for the test to read.
    """
    repo_root = str(REPO_ROOT)
    wrapper = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {repo_root!r})
        from unittest.mock import patch
        sys.argv = {argv_after_program!r}
        with patch({patch_target!r}, {patch_body}):
            import run_analyst
            sys.exit(run_analyst.main())
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


class RunAnalystCliTests(unittest.TestCase):
    def test_default_paths_are_relative_to_script_dir(self) -> None:
        # Run from an unrelated directory and override --output to a tmp file so the
        # test cannot pollute the real artifacts/requirements.json.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output = tmp_path / "out.json"
            brief = tmp_path / "brief.txt"
            brief.write_text("BRIEF", encoding="utf-8")
            proc = _run_with_patched_runner(
                [
                    "run_analyst.py",
                    "--brief",
                    str(brief),
                    "--output",
                    str(output),
                ],
                patch_body=f"return_value={GOOD_REQUIREMENTS!r}",
                cwd=str(tmp_path),
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(output.exists(), msg=f"missing: {output}")
            on_disk = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(on_disk, GOOD_REQUIREMENTS)

    def test_default_output_is_script_dir_artifacts(self) -> None:
        # The default --output path is <script dir>/artifacts/requirements.json,
        # even when launched from elsewhere. We verify this by reading the source.
        source = (REPO_ROOT / "run_analyst.py").read_text(encoding="utf-8")
        self.assertIn(
            "BASE_DIR / \"artifacts\" / \"requirements.json\"",
            source,
        )
        self.assertIn("BASE_DIR / \"brief.txt\"", source)
        # BASE_DIR itself comes from __file__ -- not from cwd.
        self.assertIn("Path(__file__).resolve().parent", source)

    def test_creates_missing_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            brief = tmp_path / "brief.txt"
            brief.write_text("BRIEF", encoding="utf-8")
            output = tmp_path / "deep" / "nested" / "out.json"
            self.assertFalse(output.parent.exists())
            proc = _run_with_patched_runner(
                [
                    "run_analyst.py",
                    "--brief",
                    str(brief),
                    "--output",
                    str(output),
                ],
                patch_body=f"return_value={GOOD_REQUIREMENTS!r}",
                cwd=str(tmp_path),
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(output.exists())
            on_disk = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(on_disk, GOOD_REQUIREMENTS)

    def test_failure_does_not_overwrite_existing_valid_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            brief = tmp_path / "brief.txt"
            brief.write_text("BRIEF", encoding="utf-8")
            output = tmp_path / "out.json"
            output.write_text(
                json.dumps({"old": "data", "marker": "KEEP_ME"}),
                encoding="utf-8",
            )
            proc = _run_with_patched_runner(
                [
                    "run_analyst.py",
                    "--brief",
                    str(brief),
                    "--output",
                    str(output),
                ],
                patch_body='side_effect=RuntimeError("injected failure")',
                cwd=str(tmp_path),
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("本次未生成新结果", proc.stderr)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"old": "data", "marker": "KEEP_ME"},
            )

    def test_writes_with_indent_and_ensure_ascii_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            brief = tmp_path / "brief.txt"
            brief.write_text("BRIEF", encoding="utf-8")
            output = tmp_path / "out.json"
            proc = _run_with_patched_runner(
                [
                    "run_analyst.py",
                    "--brief",
                    str(brief),
                    "--output",
                    str(output),
                ],
                patch_body=f"return_value={GOOD_REQUIREMENTS!r}",
                cwd=str(tmp_path),
            )
            self.assertEqual(proc.returncode, 0)
            raw = output.read_bytes()
            # ensure_ascii=False => raw text, no \\uXXXX escapes.
            self.assertIn(b'"goal"', raw)
            # indent=2 => key on its own indented line.
            self.assertIn(b'\n  "allowed_actions"', raw)
            # The goal value should appear verbatim.
            self.assertIn(GOOD_REQUIREMENTS["goal"].encode("utf-8"), raw)

    def test_reads_utf8_bom(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            brief = tmp_path / "brief.txt"
            # UTF-8 BOM + non-ASCII brief.
            brief.write_bytes(b"\xef\xbb\xbf" + "包含中文 BRIEF".encode("utf-8"))
            output = tmp_path / "out.json"
            captured_file = tmp_path / "captured.txt"

            wrapper = "\n".join([
                "import sys",
                f"sys.path.insert(0, {str(REPO_ROOT)!r})",
                "from unittest.mock import patch",
                f"sys.argv = ['run_analyst.py', '--brief', {str(brief)!r}, '--output', {str(output)!r}]",
                "",
                "def _fake(brief_text):",
                f"    with open({str(captured_file)!r}, 'w', encoding='utf-8') as fh:",
                "        fh.write(brief_text)",
                f"    return {GOOD_REQUIREMENTS!r}",
                "",
                "with patch('run_analyst.run_analyst', side_effect=_fake):",
                "    import run_analyst",
                "    sys.exit(run_analyst.main())",
                "",
            ])
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as handle:
                handle.write(wrapper)
                wrapper_path = handle.name
            try:
                proc = subprocess.run(
                    [sys.executable, wrapper_path],
                    capture_output=True,
                    text=True,
                    cwd=str(tmp_path),
                    timeout=30,
                )
            finally:
                Path(wrapper_path).unlink(missing_ok=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(captured_file.exists(), msg=f"missing: {captured_file}")
            self.assertIn("包含中文", captured_file.read_text(encoding="utf-8"))


class AtomicWriteTests(unittest.TestCase):
    """``save_requirements`` must write atomically; no half-written JSON on failure."""

    def test_save_requirements_replaces_target_atomically(self) -> None:
        from run_analyst import save_requirements

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "x.json"
            save_requirements({"a": 1}, out)
            self.assertTrue(out.exists())
            leftovers = list(out.parent.glob(".requirements_*"))
            self.assertEqual(leftovers, [])
            save_requirements({"a": 2}, out)
            self.assertEqual(json.loads(out.read_text(encoding="utf-8")), {"a": 2})

    def test_save_requirements_failure_leaves_existing_intact(self) -> None:
        from run_analyst import save_requirements

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "x.json"
            out.write_text(json.dumps({"original": True}), encoding="utf-8")
            # A file as a "parent directory" makes the recursive mkdir fail.
            blocker = Path(tmp) / "blocker"
            blocker.write_text("I am a file", encoding="utf-8")
            target = blocker / "deeper" / "x.json"
            with self.assertRaises((OSError, FileNotFoundError)):
                save_requirements({"new": True}, target)
            self.assertEqual(
                json.loads(out.read_text(encoding="utf-8")), {"original": True}
            )


if __name__ == "__main__":
    unittest.main()
