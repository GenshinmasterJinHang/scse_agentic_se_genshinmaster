# TEST_REPORT

Faithfully records the execution environment, commands, results, and known limitations.

## 1. Execution Environment

- **Operating system:** Windows 11 Home China (10.0.26200)
- **Python:** 3.13.2
- **Ollama:** local service running; `/api/tags` returns `qwen3:8b` and `gemma4:26b`
- **Source documents read:** `docs/SCSE '26 - Project Instructions - Testing.pdf` (extracted from the course template repository `prabhatram/scse-26-project-testing`).
- **Not read:** the original *Requirements Engineering* and *Plan and Develop* PDFs from Moodle (only the testing-stage PDF was supplied through the template repo); requirements for those stages come from the brief and prior conversation context.
- No models were downloaded from the network. Firewall, proxy, and PATH were not modified.

## 2. Files Changed and Added

**Requirements Engineering stage** (carried over from the previous round, referenced here)
- `analyst_agent.py` — already implemented; this round **added** the `schema` parameter to `call_qwen` so all three stages can share one HTTP client
- `brief_to_req.py`, `run_analyst.py` — unchanged

**Plan and Develop stage** (this round)
- `planner_agent.py` — `run_planner(requirement)` + `validate_plan(data)` + `PLANNER_SCHEMA` + one corrective retry + context isolation (only sends requirements as user payload)
- `developer_agent.py` — `run_developer(plan)` + `validate_developer_output(data)` + `DEVELOPER_SCHEMA` + `ast.parse` validation that the code is parseable + the entry point must be `decide_next_move(state)`
- `run_planner.py` — reads `artifacts/requirements.json` → calls `run_planner` → atomically writes `artifacts/plan.json`; on failure prints `"No new result was generated this run; existing files may belong to a previous run."` to stderr
- `run_developer.py` — reads `artifacts/plan.json` → calls `run_developer` → atomically writes `navigation_logic.py` + `artifacts/developer_output.json`

**Testing stage** (this round)
- `tests/test_analyst.py` — smoke test for the Analyst pipeline; writes `artifacts/requirements.json` and prints it
- `tests/test_planner.py` — smoke test for the Planner pipeline; writes `artifacts/plan.json` and prints it
- `tests/test_developer.py` — smoke test for the Developer pipeline; writes `artifacts/developer_output.json` + `navigation_logic.py` and asserts `decide_next_move(state)` exists
- `tests/test_generated_navigation_logic.py` — behavioural tests for `decide_next_move`: sample fixture, all 64 states, safety / preference / stop invariants, bug injection

**Existing offline tests** (carried over, no source changes required)
- `tests/test_planner_agent.py` — 17 cases (schema validation, request contract, retry, duplicate keys, context isolation)
- `tests/test_developer_agent.py` — 18 cases (schema validation, code parse, entry-point name detection, retry, context isolation)
- `tests/test_runners_cli.py` — 9 cases (CLI subprocess, atomic writes, failure preserves old file, plan/req must be a dict with the right key set)

Total: **129** unittest cases.

**Documentation** (this round rebuilt)
- `README.md` — describes the three-stage pipeline plus the testing stage
- `TEST_REPORT.md` — this file
- `docs/SCSE '26 - Project Instructions - Testing.pdf` — testing-stage instructions

## 3. Commands Actually Run and Their Results

```bash
# Unit tests
python -m unittest discover -s tests -v
# Ran 129 tests in ~16s — OK

# Stage A0 real Qwen
python brief_to_req.py --runs 3
# 3/3 successful; Unique exact-text responses: 2/3
# 5 mandatory requirements identical 3/3; Open Questions wording differs slightly but meaning is consistent

# Stage A1 real Qwen
python run_analyst.py
# Calling local Qwen: qwen3:8b
# Validated requirements saved to: artifacts/requirements.json

# Stage B1 real Qwen
python run_planner.py
# Calling local Qwen: qwen3:8b
# Planner receives only the validated requirements (context isolation).
# Validated plan saved to: artifacts/plan.json

# Stage B2 real Qwen
python run_developer.py
# Calling local Qwen: qwen3:8b
# Developer receives only the validated plan (context isolation).
# Validated developer envelope saved to: artifacts/developer_output.json
# Python module saved to: navigation_logic.py
```

All exit codes were 0.

`navigation_logic.py` was verified syntactically with `ast.parse` and actually imported with `importlib`; two representative inputs were called and the results matched the function's documented behaviour.

## 4. Offline Test Results

`python -m unittest discover -s tests -v`: **129 / 129 pass**. All HTTP calls are intercepted by `mock_opener`; the default test run does not require Ollama to be online.

## 5. Real qwen3:8b Calls

- Stage A0: 3 independent requests, each with `message.content` saved independently
- Stages A1 / B1 / B2: 1 request each, each carrying the stage-specific JSON Schema
- All three stages send `stream=false`, `think=false`, `options.temperature=0.2`, `options.num_predict=1024` to Ollama; the request body contains `messages=[system, user(only the current stage's input)]`

## 6. Formal Output Files

| File | Path | Source |
| ---- | ---- | ------ |
| Stage A0 main output | `robot_requirements.txt` | Real qwen3:8b, third response |
| Stage A0 raw per-run records | `artifacts/runs/<timestamp>/run_*.txt` | Real qwen3:8b, multiple runs |
| Stage A1 output | `artifacts/requirements.json` | Real qwen3:8b + `validate_requirements` |
| Stage B1 output | `artifacts/plan.json` | Real qwen3:8b + `validate_plan` |
| Stage B2 envelope | `artifacts/developer_output.json` | Real qwen3:8b + `validate_developer_output` |
| Stage B2 code | `navigation_logic.py` | The `code` field of the real qwen3:8b envelope, written verbatim |

## 7. Multi-Run Consistency (3 Stage-A0 runs)

5 mandatory requirements identical 3/3; 3 Open Question wordings differ slightly (see the previous round's TEST_REPORT for details).

Conclusion: **wording differs but meaning is consistent; no requirement is missing; no requirement goes beyond the original brief; no conflict with the safety constraints.**

## 8. Manual Review of Stage B1 / B2 Output

**plan.json:**
- `strategy`: *"Navigate toward the goal direction (ahead, left, or right) whenever it can be done safely, prioritizing forward movement and avoiding obstacles."* The phrase "prioritizing forward" is a preference the model introduced on its own; the strict brief-to-strategy mapping is "prefer goal direction when safe". The model has slightly added to it.
- `decisions`: 4 entries FORWARD / LEFT / RIGHT / STOP, covering the brief's three points (safe-direction forward, goal-direction preference, stop when all blocked).
- `stop_condition`: *"... when it cannot move forward, left, or right without encountering an obstacle"* — matches the brief.

**navigation_logic.py:**
- Defines `decide_next_move(state)`; `state` is a dict; only the standard library is used.
- Logic: prefer goal direction when safe; otherwise any clear direction (forward preferred); otherwise STOP.
- **Minor note:** when `goal=right` and `front_blocked=False`, the function returns `FORWARD` (forward preferred). This matches `plan.json`'s "prioritizing forward" but does not strictly follow the brief's "prefer goal direction when safe" when forward is also clear. This is a semantic interpretation choice by Qwen; it does not affect schema validation.
- Verified by `importlib`-importing the module and calling it twice.

**Known limitation:** the simulator does not read the README; it can only `import navigation_logic` and call `decide_next_move(state)`. The simulator is responsible for agreeing on the keys inside `state` (e.g. `goal_ahead` / `goal_on_left` / `goal_on_right` / `front_blocked` / `left_blocked` / `right_blocked`).

## 9. Known Issues and Outstanding Items

- **Requirements Engineering PDF not read:** not present in the project directory; the Requirements portion follows the user prompt.
- **Plan and Develop PDF not read directly:** the testing-stage PDF was supplied via the template repo; the planning-stage instructions were inferred from the prior round.
- **Team-name placeholder:** the repository directory is still `scse_agentic_se_genshinmaster`; the README header uses `<groupname>` as a placeholder until the group name is confirmed.
- **Not yet pushed to GitHub:** `.git` exists with prior commits; `gh auth status` shows the user is logged in as `GenshinmasterJinHang` (with `repo` scope). Awaiting the user's explicit authorization before `git add && git push`.
- **Not yet submitted to Moodle:** by the user's instruction, GitHub submission is done by one group member; this session will not auto-login to Moodle.
- **Semantic QA is not exhaustive:** stage B1 / B2 semantic correctness depends on Qwen's interpretation. This test suite verifies schema and executability, not whether every simulator environment gets the expected reward.
- **Stage B2 entry-point signature:** the entry point must be `decide_next_move(state)`; the simulator must agree on the keys inside `state`.

## 10. Pre-Submission Checklist

1. Confirm the actual group name (default guess `genshinmaster`; please confirm). Replace the `<groupname>` placeholder in the README header and the repository directory name accordingly.
2. Decide where on GitHub the repository lives:
   - Create `scse_agentic_se_<groupname>` under `GenshinmasterJinHang` (recommended; `gh auth status` is already logged in to this account), or
   - Push to an existing same-named repository owned by another team member (requires that account to be logged in to `gh`).
3. Local `.git` already has prior commits; a clean `git add . && git commit -m "..." && git push` will publish the new content.
4. `brief.txt`, `requirements.json`, `plan.json`, `developer_output.json`, `navigation_logic.py` are all produced by real local runs — submit as-is, no replacement needed.
5. **No automatic push or Moodle submission will be performed without your explicit consent.**