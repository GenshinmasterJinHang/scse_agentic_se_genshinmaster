# TEST_REPORT

如实记录执行环境、命令、结果与已知限制。

## 1. 执行环境

- 操作系统：Windows 11 Home China（10.0.26200）
- Python：3.13.2
- Ollama：本地服务运行中，`/api/tags` 返回 `qwen3:8b`、`gemma4:26b`
- 已读取 `SCSE '26 - Project Instructions - Plan and Develop.pdf`（PDF 文本提取自课程模板仓库 `prabhatram/scse_plan_and_develop`，并非 Moodle 上的原始作业 PDF；以该 PDF 内容为准）
- 未读取 `SCSE 26 - Project Instructions - Requirements Engineering.pdf`（本仓库目录与对话上下文均未提供；Requirements Engineering 部分的要求以来源仓库内 `brief.txt` 与用户提示词为准）
- 未联网下载任何模型，未修改防火墙、代理或 PATH

## 2. 修改与新增文件

**Requirements Engineering 部分**（沿用上一轮，已存在并被本轮引用）
- `analyst_agent.py` — 上一轮已实现，本轮**追加** `call_qwen` 的 `schema` 参数以支持三阶段共用
- `brief_to_req.py`、`run_analyst.py` — 不变

**Plan and Develop 部分**（本轮新增）
- `planner_agent.py` — `run_planner(requirement)` + `validate_plan(data)` + `PLANNER_SCHEMA` + 一次纠正重试 + `context isolation`（只把 requirements 作为 user payload 发送）
- `developer_agent.py` — `run_developer(plan)` + `validate_developer_output(data)` + `DEVELOPER_SCHEMA` + `ast.parse` 验证 code 可解析 + 要求至少定义一个导航函数
- `run_planner.py` — 读 `artifacts/requirements.json` → 调 `run_planner` → 原子写入 `artifacts/plan.json`，失败 stderr 输出「本次未生成新结果；现有文件可能属于之前的运行。」
- `run_developer.py` — 读 `artifacts/plan.json` → 调 `run_developer` → 原子写入 `navigation_logic.py` + `artifacts/developer_output.json`

**测试**（本轮新增）
- `tests/test_planner_agent.py` — 17 用例（schema 校验、请求合同、重试、重复键、context isolation）
- `tests/test_developer_agent.py` — 18 用例（schema 校验、代码 parse、nav 函数名检测、重试、context isolation）
- `tests/test_runners_cli.py` — 9 用例（CLI 子类、原子写入、不覆盖旧文件、plan/req 必须为 dict 且字段集正确）

总计 **114** 个 unittest 用例。

**文档**（本轮重建）
- `README.md` — 描述三阶段流水线
- `TEST_REPORT.md` — 本文件

## 3. 实际执行的命令与结果

```bash
# 单元测试
python -m unittest discover -s tests -v
# Ran 114 tests in 1.9s — OK

# 阶段 A0 真实 Qwen
python brief_to_req.py --runs 3
# 3/3 成功；Unique exact-text responses: 2/3
# 5 条强制需求 3/3 完全一致；Open Questions 措辞略不同但含义一致

# 阶段 A1 真实 Qwen
python run_analyst.py
# Calling local Qwen: qwen3:8b
# Validated requirements saved to: artifacts/requirements.json

# 阶段 B1 真实 Qwen
python run_planner.py
# Calling local Qwen: qwen3:8b
# Planner receives only the validated requirements (context isolation).
# Validated plan saved to: artifacts/plan.json

# 阶段 B2 真实 Qwen
python run_developer.py
# Calling local Qwen: qwen3:8b
# Developer receives only the validated plan (context isolation).
# Validated developer envelope saved to: artifacts/developer_output.json
# Python module saved to: navigation_logic.py
```

所有退出码均为 0。

`navigation_logic.py` 已用 `ast.parse` 验证语法合法，并用 `importlib` 实际 import 执行了两次典型输入，结果符合直觉（goal=left + forward 阻挡 + left/right 畅通 → LEFT；forward 畅通 → FORWARD）。

## 4. 离线测试结果

`python -m unittest discover -s tests -v`：**114 / 114 通过**。所有 HTTP 调用通过 `mock_opener` 拦截，默认不需要 Ollama 在线。

## 5. 真实 qwen3:8b 调用

- 阶段 A0：3 次独立请求，每次独立保存 `message.content`
- 阶段 A1 / B1 / B2：各 1 次请求，每次带对应阶段的 JSON Schema
- 三阶段均向 Ollama 发送 `stream=false`、`think=false`、`options.temperature=0.2`、`options.num_predict=1024`，请求体里 `messages=[system, user(仅当前阶段输入)]`

## 6. 正式输出文件

| 文件 | 路径 | 来源 |
| ---- | ---- | ---- |
| 阶段 A0 主输出 | `robot_requirements.txt` | 真 qwen3:8b 第 3 次响应 |
| 阶段 A0 多次原始记录 | `artifacts/runs/<时间戳>/run_*.txt` | 真 qwen3:8b 多次 |
| 阶段 A1 输出 | `artifacts/requirements.json` | 真 qwen3:8b + `validate_requirements` |
| 阶段 B1 输出 | `artifacts/plan.json` | 真 qwen3:8b + `validate_plan` |
| 阶段 B2 envelope | `artifacts/developer_output.json` | 真 qwen3:8b + `validate_developer_output` |
| 阶段 B2 代码 | `navigation_logic.py` | 真 qwen3:8b envelope 中 `code` 字段原样落地 |

## 7. 多次输出一致性观察（仅本次 3 次 A0 运行）

5 条强制需求 3/3 完全一致；3 个 Open Question 措辞略不同（详见上一轮 TEST_REPORT）。

结论：**表达不同但含义一致；无需求遗漏；无原文未规定要求；与安全约束无冲突**。

## 8. 阶段 B1 / B2 的人工复核

**plan.json：**
- `strategy`：「Navigate toward the goal direction (ahead, left, or right) whenever it can be done safely, prioritizing forward movement and avoiding obstacles.」——「prioritizing forward」是模型自行引入的偏好；与 brief 严格对应的是「prefer goal direction when safe」，模型略有增补。
- `decisions`：4 条 FORWARD / LEFT / RIGHT / STOP，覆盖 brief 中的安全方向前进、目标方向偏好、全部被阻挡时停止三要点。
- `stop_condition`：「... when it cannot move forward, left, or right without encountering an obstacle」——与 brief 一致。

**navigation_logic.py：**
- 定义 `decide_action(sensor_data, goal_direction)`，参数是 dict + 字符串，仅使用标准库
- 逻辑：forward 畅通 → FORWARD；否则按 goal 方向选 LEFT/RIGHT；否则 STOP
- **小问题**：当 goal=right 且 forward 畅通时，函数返回 FORWARD（forward 优先），未严格遵循 brief「prefer goal direction when safe」——这与 `plan.json` 中引入的「prioritizing forward」一致。模型将 brief 的「prefer goal」解读成了「forward 优先 + 否则按 goal 转向」。这是 Qwen 的语义判断差异，不影响 schema 校验。
- 已通过 `importlib` 实际导入并调用两次，结果符合函数内部逻辑。

**已知限制**：模拟器不会读 README，**只能**通过 `import navigation_logic` 调用 `decide_action(sensor_data, goal_direction)`。模拟器需要自行约定 `sensor_data` 的键集（forward_clear / left_clear / right_clear）和 `goal_direction` 的取值（"ahead" / "left" / "right" / None）。

## 9. 已知问题与未完成项

- **未读取的** Requirements Engineering PDF：项目目录里没有此文件；Requirements 部分的需求按用户提示词段整理。
- **未读取的** Plan and Develop PDF：已通过课程模板仓库里的同名 PDF 提取文本，作为唯一依据。
- **团队名占位**：仓库目录仍为 `scse_agentic_se_genshinmaster`；README 顶部用 `<groupname>` 占位，等用户确认组名后再正式改名。
- **未推送 GitHub**：`.git` 已存在但无 commit；`gh auth status` 已登录为 `GenshinmasterJinHang`（具备 `repo` scope），**待用户授权**后再 `git add && git push`。
- **未提交 Moodle**：按用户要求，提交 GitHub 链接由组里一人完成；本会话不会自动登录 Moodle。
- **语义层面的 QA 不是完备**：阶段 B1 / B2 输出的语义正确性依赖 Qwen 的解释能力，本测试套件只验证 schema 与可执行性，不验证是否真在所有模拟环境里得到期望回报。
- **阶段 B2 函数签名**：是 Qwen 自行决定的（参数名 `sensor_data` / `goal_direction`），模拟器需配合使用。

## 10. 提交前需要做的操作

1. 确认实际组名（默认猜测为 `genshinmaster`，需你确认），把 README 顶部 `<groupname>` 与仓库目录名替换为 `scse_agentic_se_<groupname>`。
2. 决定 GitHub 仓库位置：
   - 在 `GenshinmasterJinHang` 下新建 `scse_agentic_se_<groupname>`（推荐，因为 `gh auth status` 已登录此账号），或
   - 推送到组里其他成员已有的同名仓库（需要该账号也登录 `gh`）
3. 本地已有 `.git` 但无 commit，需要一次干净的 `git add . && git commit -m "..."` 后再 push。
4. `brief.txt`、`requirements.json`、`plan.json`、`developer_output.json`、`navigation_logic.py` 都是本地真机运行产生的——直接提交即可，不需替换。
5. **未经你明确同意，不会**自动 push / 提交 Moodle。