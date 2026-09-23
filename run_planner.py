"""Read requirements.json, call the Planner, and save only validated Qwen output.

Run after ``run_analyst.py`` produces ``artifacts/requirements.json``. The
Planner must receive ONLY the validated requirements -- never the original
brief or the Analyst conversation. This is the "context isolation" rule from
the SCSE '26 project instructions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

from analyst_agent import OllamaError, get_model_name
from planner_agent import PlanValidationError, run_planner


BASE_DIR = Path(__file__).resolve().parent


def save_plan(data: dict[str, Any], output_path: Path) -> None:
    """Write atomically: temp file + replace, never half-written JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent,
            prefix=".plan_", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_requirements(path: Path) -> dict[str, Any]:
    """Load requirements.json; reject anything that is not a 4-field dict."""
    raw = path.read_text(encoding="utf-8-sig")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("requirements.json must contain a JSON object.")
    required = {"goal", "allowed_actions", "safe_stop", "avoid_obstacles"}
    if set(data) != required:
        raise ValueError(
            f"requirements.json keys must be exactly {required}, got {set(data)}."
        )
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=BASE_DIR / "artifacts" / "requirements.json")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "artifacts" / "plan.json")
    args = parser.parse_args(argv)

    try:
        requirements = load_requirements(args.requirements)
        print(f"Calling local Qwen: {get_model_name()}", flush=True)
        print(f"Planner receives only the validated requirements (context isolation).", flush=True)
        plan = run_planner(requirements)
        save_plan(plan, args.output)
    except (OllamaError, PlanValidationError, ValueError, OSError, UnicodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("本次未生成新结果；现有文件可能属于之前的运行。", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 -- runner is the last line of defense.
        print(f"UNEXPECTED ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("本次未生成新结果；现有文件可能属于之前的运行。", file=sys.stderr)
        return 1

    print(f"Validated plan saved to: {args.output.resolve()}")
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    print("Review the plan against requirements.json; schema checks do not prove semantic correctness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())