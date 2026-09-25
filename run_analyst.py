"""Read brief.txt, call the analyst, and save only validated Qwen output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

from analyst_agent import OllamaError, get_model_name, run_analyst

BASE_DIR = Path(__file__).resolve().parent


def save_requirements(data: dict[str, Any], output_path: Path) -> None:
    """Write atomically so a failed write does not destroy an older result."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent,
            prefix=".requirements_", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brief", type=Path, default=BASE_DIR / "brief.txt")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "artifacts" / "requirements.json")
    args = parser.parse_args(argv)
    try:
        brief_text = args.brief.read_text(encoding="utf-8-sig")
        print(f"Calling local Qwen: {get_model_name()}", flush=True)
        requirements = run_analyst(brief_text)
        save_requirements(requirements, args.output)
    except (OllamaError, ValueError, OSError, UnicodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("No new result was generated this run; existing files may belong to a previous run.", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 -- runner is the last line of defense.
        print(f"UNEXPECTED ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("No new result was generated this run; existing files may belong to a previous run.", file=sys.stderr)
        return 1
    print(f"Validated requirements saved to: {args.output.resolve()}")
    print(json.dumps(requirements, ensure_ascii=False, indent=2))
    print("Review the goal against brief.txt; passing schema checks does not prove semantic correctness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
