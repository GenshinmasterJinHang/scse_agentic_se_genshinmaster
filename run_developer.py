"""Read plan.json, call the Developer Agent, and save the validated Python code.

The Developer Agent returns a JSON envelope ``{description, code}``. The runner
extracts ``code`` and writes it verbatim to ``navigation_logic.py`` so the
simulator can ``import`` it. The full JSON envelope is also persisted to
``artifacts/developer_output.json`` for review.

Context isolation: the Developer receives ONLY the validated plan -- never the
Planner conversation or the original requirements dict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

from analyst_agent import OllamaError, get_model_name
from developer_agent import DeveloperValidationError, run_developer


BASE_DIR = Path(__file__).resolve().parent


def save_envelope(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent,
            prefix=".developer_", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_python_module(code: str, output_path: Path) -> None:
    """Write the developer's code to a .py file. Atomic replace."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent,
            prefix=".navigation_", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(code)
            if not code.endswith("\n"):
                handle.write("\n")
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_plan(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8-sig")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("plan.json must contain a JSON object.")
    required = {"strategy", "decisions", "stop_condition"}
    if set(data) != required:
        raise ValueError(
            f"plan.json keys must be exactly {required}, got {set(data)}."
        )
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=BASE_DIR / "artifacts" / "plan.json")
    parser.add_argument(
        "--module-output",
        type=Path,
        default=BASE_DIR / "navigation_logic.py",
        help="Where to save the developer's Python code.",
    )
    parser.add_argument(
        "--envelope-output",
        type=Path,
        default=BASE_DIR / "artifacts" / "developer_output.json",
        help="Where to save the full JSON envelope from the Developer Agent.",
    )
    args = parser.parse_args(argv)

    try:
        plan = load_plan(args.plan)
        print(f"Calling local Qwen: {get_model_name()}", flush=True)
        print(f"Developer receives only the validated plan (context isolation).", flush=True)
        envelope = run_developer(plan)
        write_python_module(envelope["code"], args.module_output)
        save_envelope(envelope, args.envelope_output)
    except (OllamaError, DeveloperValidationError, ValueError, OSError, UnicodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("本次未生成新结果；现有文件可能属于之前的运行。", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 -- runner is the last line of defense.
        print(f"UNEXPECTED ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("本次未生成新结果；现有文件可能属于之前的运行。", file=sys.stderr)
        return 1

    print(f"Validated developer envelope saved to: {args.envelope_output.resolve()}")
    print(f"Python module saved to: {args.module_output.resolve()}")
    print(f"Description: {envelope['description']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())