# scse_agentic_se_&lt;groupname&gt;

SCSE 26《Requirements Engineering》与《Plan and Develop》课程作业：用本地 Ollama + qwen3:8b，从机器人导航 brief 出发，依次通过 Analyst → Planner → Developer 三个 Agent 生成可执行的需求、计划与 Python 导航代码。

## 项目目标

`brief.txt` 里自然语言描述的机器人导航规则，经三阶段自动传递：

1. **Analyst** 提取可校验的需求 JSON（四字段）。
2. **Planner** 把需求转化为带策略/决策/停止条件的计划 JSON。
3. **Developer** 把计划写成一个可被模拟器使用的 Python 模块。

每一步都直接调用本机 Ollama 中已下载的 qwen3:8b，输出文件中的字段**全部来自模型响应**，Python 代码只做格式与合法性校验，不会用预设答案冒充。

## 三阶段流水线

| 阶段 | 程序 | 模型输出形态 | 输出文件 | 关键约束 |
| ---- | ---- | ------------ | -------- | -------- |
| A0 自由文本 | `brief_to_req.py` | 编号列表 + Open Questions | `robot_requirements.txt`、`artifacts/runs/<时间戳>/run_*.txt` | 不带 JSON Schema，便于观察表达差异 |
| A1 需求 | `run_analyst.py` | 固定四字段 JSON | `artifacts/requirements.json` | 强制 schema + 严格校验，schema 不预设 goal 答案 |
| B1 计划 | `run_planner.py` | `{strategy, decisions[], stop_condition}` JSON | `artifacts/plan.json` | 接收的**只有**需求，不含 Analyst 对话（context isolation） |
| B2 代码 | `run_developer.py` | `{description, code}` JSON | `artifacts/developer_output.json` + `navigation_logic.py` | code 必须是可解析的 Python 且定义 nav 函数 |

每一步都向 Ollama 发送 `stream=false`、`think=false`、`options.temperature=0.2`、`options.num_predict=1024`，并在请求体里带 `format=<该阶段专属 Schema>`。`call_qwen()` 是三阶段共用的 HTTP 客户端，schema 由调用方注入。

## 准备环境

1. 安装 Python 3.10+（本机实测 3.13.2 可用）
2. 安装 Ollama：从 https://ollama.com/download 下载并启动
3. 拉取模型：
   ```bash
   ollama pull qwen3:8b
   ```
4. 保持 `ollama serve` 在后台运行（如果未以服务方式自动启动）

仅使用 Python 标准库，**无需** `pip install`。

## 默认配置

- 默认模型：`qwen3:8b`（不可静默替换为其他 Qwen 尺寸或云端 API）
- 默认 Ollama 地址：`http://127.0.0.1:11434`
- 默认超时：`300` 秒
- 请求参数：`stream=false`、`think=false`、`options.temperature=0.2`、`options.num_predict=1024`
- 每阶段带各自的 JSON Schema：`REQUIREMENTS_SCHEMA` / `PLANNER_SCHEMA` / `DEVELOPER_SCHEMA`

## 环境变量

| 变量 | 默认 | 说明 |
| ---- | ---- | ---- |
| `QWEN_MODEL` | `qwen3:8b` | 使用的模型 tag |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | 仅允许 localhost / 127.0.0.1 / ::1 |
| `OLLAMA_TIMEOUT_SECONDS` | `300` | 必须为正数 |
| `QWEN_THINK` | 未设置（等同于 `false`） | 仅对非结构化请求生效；结构化请求一律 `think=false` |

## 作业生成命令（Windows PowerShell）

按顺序执行：

```powershell
$env:QWEN_MODEL = "qwen3:8b"
$env:OLLAMA_BASE_URL = "http://127.0.0.1:11434"
$env:OLLAMA_TIMEOUT_SECONDS = "300"

# 阶段 A0：自由文本，可选，用于观察一致性
python brief_to_req.py --runs 3

# 阶段 A1：生成 requirements.json
python run_analyst.py

# 阶段 B1：生成 plan.json（接收 requirements.json，不再接收 brief）
python run_planner.py

# 阶段 B2：生成 navigation_logic.py + developer_output.json（接收 plan.json）
python run_developer.py

# 单元测试
python -m unittest discover -s tests -v
```

环境变量仅在当前终端会话中生效。默认配置已正确时，**不设置**这些变量也应能跑通。

## 离线测试

```bash
python -m unittest discover -s tests -v
```

- 114 个 unittest 用例覆盖：
  - `validate_requirements` / `validate_plan` / `validate_developer_output` 的所有 schema 边界
  - 三阶段请求合同：`qwen3:8b` / `stream=false` / `think=false` / 各自带 Schema
  - 阶段 A1 / A2 / A4 一致重试（首次坏 + 第二次好 = 成功；两次都坏 = 抛错且不超 2 次）
  - 网络/超时/截断/404 等基础设施错误立即抛 `OllamaError`，**不**反复重试
  - Planner / Developer 的 **context isolation**：只接收上一阶段产物，**不**接收 brief
  - 各 CLI 的原子写入与「失败不覆盖旧文件」
- 默认测试**不**要求 Ollama 在线，所有 HTTP 调用都已 mock
- 真机验证见 `TEST_REPORT.md`

## 输出文件清单

| 文件 | 内容 | 是否真实 Qwen |
| ---- | ---- | ------------ |
| `robot_requirements.txt` | 阶段 A0 最新模型输出 | 是 |
| `artifacts/runs/<时间戳>/run_*.txt` | 阶段 A0 各次原始响应 | 是 |
| `artifacts/requirements.json` | 阶段 A1 四字段需求 | 是 |
| `artifacts/plan.json` | 阶段 B1 计划 JSON | 是 |
| `artifacts/developer_output.json` | 阶段 B2 envelope（含 description 与 code） | 是 |
| `navigation_logic.py` | 阶段 B2 代码落地，供模拟器 `import` | 是 |

`navigation_logic.py` 是可被模拟器直接 `import` 的 Python 模块，已通过 `ast.parse` 解析验证。

## 必须知道的两条限制

1. **输出必须来自 Qwen**——Python 不会用预设字典冒充模型输出；schema 校验只检查结构，不证明语义正确。
2. **JSON 结构校验 ≠ 完整语义校验**——例如 `goal` 字段只要非空字符串就通过校验，但内容是否准确反映 brief 需要人工复核；`navigation_logic.py` 通过 `ast.parse` 不代表它真的能跑通所有模拟环境。

## 失败与重试

- 三阶段都有「首次 + 一次纠正」上限：第二次失败立刻抛 `*ValidationError`。
- 网络/超时/模型缺失/服务不可达等基础设施错误立即抛 `OllamaError`，**不会**伪装成格式错误反复重试。
- 命令行程序失败时返回非零退出码；已有的 `requirements.json` / `plan.json` / `developer_output.json` / `navigation_logic.py` 不会被覆盖，会打印「本次未生成新结果；现有文件可能属于之前的运行。」

## 文件清单

正式提交（课程要求的产出）：

- `brief.txt` — 作业原始 brief
- `brief_to_req.py` — 阶段 A0 自由文本提取
- `analyst_agent.py` — 阶段 A1 Analyst Agent
- `run_analyst.py` — 阶段 A1 CLI
- `planner_agent.py` — 阶段 B1 Planner Agent
- `run_planner.py` — 阶段 B1 CLI
- `developer_agent.py` — 阶段 B2 Developer Agent
- `run_developer.py` — 阶段 B2 CLI
- `tests/` — 离线单元测试（114 用例）
- `robot_requirements.txt` — 阶段 A0 真实输出
- `artifacts/requirements.json` — 阶段 A1 真实输出
- `artifacts/plan.json` — 阶段 B1 真实输出
- `artifacts/developer_output.json` — 阶段 B2 envelope
- `navigation_logic.py` — 阶段 B2 真实代码
- `README.md`、`TEST_REPORT.md`、`.gitignore`

辅助证据：阶段 A0 的 `artifacts/runs/<时间戳>/` 可保留以便复核，不强制提交。

## 常见错误

| 现象 | 原因 | 处理 |
| ---- | ---- | ---- |
| `Cannot reach local Ollama` | `ollama serve` 未启动 | 启动 Ollama |
| `Ollama HTTP 404 ... ollama pull qwen3:8b` | 模型未下载 | `ollama pull qwen3:8b` |
| `Qwen timed out after 300 seconds` | 模型加载或机器较慢 | 调大 `OLLAMA_TIMEOUT_SECONDS` |
| `*ValidationError ... twice` | 两次都未通过校验 | 检查提示词与上游产物；不会无限重试 |