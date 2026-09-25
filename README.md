# scse_agentic_se_&lt;groupname&gt;

SCSE '26 coursework for *Requirements Engineering*, *Plan and Develop*, and *Testing*. A robot navigation brief is processed by three agents (Analyst → Planner → Developer) using local Ollama + qwen3:8b, producing validated requirements, a navigation plan, and executable Python code. Every field in the output files comes from the model — Python only checks structure and legality, never fakes a preset answer.

The Testing stage adds a four-test pipeline (Level 1 smoke tests + Level 2 behavioural tests) on top of the production pipeline.

## Project Goal

The natural-language robot navigation rules in `brief.txt` are passed through four automated stages:

1. **Analyst** extracts a validated four-field requirements JSON object.
2. **Planner** turns the requirements into a plan JSON object with strategy / decisions / stop_condition.
3. **Developer** writes the plan into a Python module that defines `decide_next_move(state)` for the simulator to import.
4. **Tests** verify each stage and the generated navigation logic.

Each stage calls the locally installed `qwen3:8b` in Ollama.

## Three-Stage Pipeline

| Stage | Script | Model Output Shape | Output Files | Key Constraints |
| ----- | ------ | ------------------ | ------------ | --------------- |
| A0 Free text | `brief_to_req.py` | Numbered list + Open Questions | `robot_requirements.txt`, `artifacts/runs/<timestamp>/run_*.txt` | No JSON Schema; useful for observing wording drift |
| A1 Requirements | `run_analyst.py` | Fixed four-field JSON | `artifacts/requirements.json` | Enforced schema + strict validation; schema does not pre-fill any answers |
| B1 Plan | `run_planner.py` | `{strategy, decisions[], stop_condition}` JSON | `artifacts/plan.json` | Receives **only** the requirements — never the Analyst conversation (context isolation) |
| B2 Code | `run_developer.py` | `{description, code}` JSON | `artifacts/developer_output.json` + `navigation_logic.py` | `code` must define a top-level function named `decide_next_move(state)` |

Every stage sends `stream=false`, `think=false`, `options.temperature=0.2`, `options.num_predict=1024` to Ollama, and embeds `format=<stage-specific schema>` in the request body. `call_qwen()` is the HTTP client shared by all three stages; each caller injects its own schema.

## Testing Strategy

Two levels of testing, as required by [`docs/SCSE '26 - Project Instructions - Testing.pdf`](docs/SCSE%20%27%26%20-%20Project%20Instructions%20-%20Testing.pdf):

**Level 1 — Agent / Pipeline smoke tests.** Run the actual agents with their real inputs and let each agent's own validation accept or reject its output. Each smoke test writes the artifact to `artifacts/` and prints it after the test ends:

- `tests/test_analyst.py` — calls the Analyst with `brief.txt`, writes `artifacts/requirements.json`.
- `tests/test_planner.py` — calls the Planner with `artifacts/requirements.json`, writes `artifacts/plan.json`.
- `tests/test_developer.py` — calls the Developer with `artifacts/plan.json`, writes `artifacts/developer_output.json` + `navigation_logic.py`, and asserts that `decide_next_move(state)` exists.

**Level 2 — Behavioural testing.** Call `decide_next_move(state)` with known states and check the returned action:

- `tests/test_generated_navigation_logic.py` — runs the sample case from the testing instructions, enumerates all 64 states, asserts safety / preference / stop invariants, and includes a bug-injection test to demonstrate that the suite actually catches bad navigation logic.

When Ollama is unreachable, the smoke tests fall back to a mocked HTTP layer that returns a known-good envelope, so the suite still exercises the full validator + atomic-write path offline.

## Prerequisites

1. Install Python 3.10+ (tested locally with 3.13.2)
2. Install Ollama from https://ollama.com/download and start it
3. Pull the model:
   ```bash
   ollama pull qwen3:8b
   ```
4. Keep `ollama serve` running in the background (if it is not already running as a service)

Only the Python standard library is used — **no** `pip install` required.

## Default Configuration

- Default model: `qwen3:8b` (cannot be silently swapped for another Qwen size or a cloud API)
- Default Ollama URL: `http://127.0.0.1:11434`
- Default timeout: `300` seconds
- Request parameters: `stream=false`, `think=false`, `options.temperature=0.2`, `options.num_predict=1024`
- Per-stage JSON Schema: `REQUIREMENTS_SCHEMA` / `PLANNER_SCHEMA` / `DEVELOPER_SCHEMA`

## Environment Variables

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `QWEN_MODEL` | `qwen3:8b` | Model tag to use |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Only `localhost` / `127.0.0.1` / `::1` allowed |
| `OLLAMA_TIMEOUT_SECONDS` | `300` | Must be a positive number |
| `QWEN_THINK` | not set (= `false`) | Only affects unstructured requests; structured requests always send `think=false` |

## Generation Commands (Windows PowerShell)

Run them in order:

```powershell
$env:QWEN_MODEL = "qwen3:8b"
$env:OLLAMA_BASE_URL = "http://127.0.0.1:11434"
$env:OLLAMA_TIMEOUT_SECONDS = "300"

# Stage A0 — free text; optional, useful for observing consistency
python brief_to_req.py --runs 3

# Stage A1 — produce requirements.json
python run_analyst.py

# Stage B1 — produce plan.json (reads requirements.json, never the brief)
python run_planner.py

# Stage B2 — produce navigation_logic.py + developer_output.json (reads plan.json)
python run_developer.py

# Stage T1 — smoke tests for all three agents (real Ollama)
python -m unittest tests.test_analyst tests.test_planner tests.test_developer -v

# Stage T2 — behavioural tests for the generated navigation logic
python -m unittest tests.test_generated_navigation_logic -v

# Full offline test suite
python -m unittest discover -s tests -v
```

The environment variables only live for the current shell session. With sane defaults you should be able to run everything **without** setting them.

## Offline Tests

```bash
python -m unittest discover -s tests -v
```

The offline suite covers:

- Schema boundaries for `validate_requirements` / `validate_plan` / `validate_developer_output`
- Three-stage request contract: `qwen3:8b` / `stream=false` / `think=false` / per-stage Schema
- One corrective retry at each stage (first bad + second good = success; two bads = raise, never more than 2 calls)
- Network / timeout / truncation / 404 infrastructure errors raise `OllamaError` immediately — **no** retry loop
- Planner / Developer **context isolation**: each stage receives the previous artifact, never the brief
- Atomic writes for every CLI and the "failure preserves the old file" rule
- `decide_next_move(state)` must be the entry point (wrong name / wrong parameter / wrong arity all rejected)
- Behavioural tests for `decide_next_move`: sample fixture, all 64 states, safety / preference / stop invariants, bug injection

The default test run **does not** require Ollama to be online; all HTTP calls in the offline tests are mocked.

## Output Files

| File | Contents | Real Qwen? |
| ---- | -------- | ---------- |
| `robot_requirements.txt` | Latest Stage A0 model output | Yes |
| `artifacts/runs/<timestamp>/run_*.txt` | Each Stage A0 raw response | Yes |
| `artifacts/requirements.json` | Stage A1 four-field requirements | Yes |
| `artifacts/plan.json` | Stage B1 plan JSON | Yes |
| `artifacts/developer_output.json` | Stage B2 envelope (description + code) | Yes |
| `navigation_logic.py` | Stage B2 code, defines `decide_next_move(state)` | Yes |

`navigation_logic.py` is a Python module that the simulator can `import` directly; the runner verifies it parses with `ast.parse` and that it defines `decide_next_move(state)` before persisting it.

## Two Caveats

1. **Outputs must come from Qwen.** Python never fakes a preset dictionary in place of a model response; the schema check only verifies structure, not semantics.
2. **JSON shape validation ≠ full semantic validation.** For example, `goal` only needs to be a non-empty string to pass; whether its content actually reflects the brief must be reviewed by a human. `navigation_logic.py` passing `ast.parse` and `decide_next_move(state)` validation does not prove it will behave well in every simulator environment.

## Failure and Retry

- Every stage has a "first attempt + one corrective retry" cap; on the second failure the corresponding `*ValidationError` is raised immediately.
- Infrastructure errors — network, timeout, model missing, service unreachable — raise `OllamaError` immediately and are **not** dressed up as format errors and retried.
- When a CLI fails it returns a non-zero exit code; any existing `requirements.json` / `plan.json` / `developer_output.json` / `navigation_logic.py` is preserved, and the message **"No new result was generated this run; existing files may belong to a previous run."** is printed to stderr.

## File Inventory

Required submissions (the course-mandated deliverables):

- `brief.txt` — the original brief
- `brief_to_req.py` — Stage A0 free-text extractor
- `analyst_agent.py` — Stage A1 Analyst Agent
- `run_analyst.py` — Stage A1 CLI
- `planner_agent.py` — Stage B1 Planner Agent
- `run_planner.py` — Stage B1 CLI
- `developer_agent.py` — Stage B2 Developer Agent (requires `decide_next_move(state)`)
- `run_developer.py` — Stage B2 CLI
- `tests/test_analyst.py` — Stage T1 Analyst smoke test
- `tests/test_planner.py` — Stage T1 Planner smoke test
- `tests/test_developer.py` — Stage T1 Developer smoke test
- `tests/test_generated_navigation_logic.py` — Stage T2 behavioural tests for `decide_next_move(state)`
- `tests/` — additional offline unit tests (schema, contract, atomic writes, etc.)
- `robot_requirements.txt` — Stage A0 real output
- `artifacts/requirements.json` — Stage A1 real output
- `artifacts/plan.json` — Stage B1 real output
- `artifacts/developer_output.json` — Stage B2 envelope
- `navigation_logic.py` — Stage B2 real code (defines `decide_next_move(state)`)
- `README.md`, `TEST_REPORT.md`, `.gitignore`
- `docs/SCSE '26 - Project Instructions - Testing.pdf` — testing-stage instructions

Supporting evidence: the per-run Stage A0 directories `artifacts/runs/<timestamp>/` may be kept for review; submission is not required.

## Common Errors

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| `Cannot reach local Ollama` | `ollama serve` not running | Start Ollama |
| `Ollama HTTP 404 ... ollama pull qwen3:8b` | Model not pulled | `ollama pull qwen3:8b` |
| `Qwen timed out after 300 seconds` | Slow model load or slow machine | Increase `OLLAMA_TIMEOUT_SECONDS` |
| `*ValidationError ... twice` | Two failed validations in a row | Inspect the prompt and upstream artifact; no infinite retries |
| `decide_next_move must take exactly one parameter named 'state'` | Developer Agent returned a function with the wrong signature | Re-run `run_developer.py`; the prompt explicitly requires `decide_next_move(state)` |