# 贡献指南

安装与运行见 [README.md](README.md)。本页：开发环境、风格、测试与治理文档入口。

## 开发环境

```bash
git clone https://github.com/suuugeee/lingzhou-agent.git
cd lingzhou-agent
./setup-lingzhou.sh
```

手动路径（等价）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.12
UV_PROJECT_ENVIRONMENT="$(pwd)/.venv" uv pip install -e ".[test]"
ln -sf "$(pwd)/.venv/bin/lingzhou" ~/.local/bin/lingzhou
```

## 代码风格

- Python 3.12+，类型注解，4 空格缩进
- `from __future__ import annotations`

## 提交规范

```
feat: xxx      # 新功能
fix: xxx       # 修复
docs: xxx      # 文档
chore: xxx     # 杂项
refactor: xxx  # 重构
```

## 测试

项目要求 **Python 3.12+**。优先使用仓库虚拟环境：

```bash
./setup-lingzhou.sh   # 或: uv venv .venv --python 3.12 && uv pip install -e ".[test]"
.venv/bin/python -m pytest tests/ -q
```

定向回归示例：

```bash
.venv/bin/python -m pytest tests/test_log_fields.py tests/test_judgment_layers.py tests/test_boundary_contracts.py tests/test_import_boundaries.py -q
```

勿用系统自带的 Python 3.9 跑全量测试（`match` 语法与类型注解不兼容）。

### Import 边界

跨层禁止依赖见 [REPO_MAP.md](docs/reference/REPO_MAP.md)。提交前可跑：

```bash
.venv/bin/python -m pytest tests/test_import_boundaries.py -q
# 或
.venv/bin/python scripts/check_import_boundaries.py
# 安装 editable 后也可：
lingzhou-check-imports

治理门禁（import 边界 + Ruff 治理面，与 CI 一致）：

```bash
./scripts/run_governance_checks.sh
# 含全量 pytest：
./scripts/run_governance_checks.sh --full-tests
```

静态检查（需 `pip install -e ".[test]"` 含 ruff、pytest-cov）：

```bash
.venv/bin/python -m ruff check core tools
.venv/bin/python -m pytest tests/ -q \
  --cov=core/judgment --cov=core/execution --cov=core/contracts \
  --cov-fail-under=80
```

演化 smoke 与工具公开 API 见 `core/smoke_tests.py`、`tests/test_boundary_contracts.py`（ADR 0001）。
```

新代码请使用 canonical 路径（勿用已删除 shim，见 ADR 0009）：

| 用途 | 导入 |
|------|------|
| 配置加载 | `from core.config import Config` |
| 配置子模型 | `from core.config_models import ThresholdsConfig` 等 |
| 项目根目录 | `from core.paths import project_root` |
| 执行层 | `core.execution`、`core.contracts.*` |
| 判断边界 | `core.judgment.boundary` |

`core/__init__.py` 不做懒加载重导出；`core.config` 不聚合 `config_models`。  
任务相关：`from core.loop.task.runtime import …`、`from core.loop.task.parallel import run_tasks_parallel`（ADR 0019）。  
循环子包直引见 [REPO_MAP](docs/reference/REPO_MAP.md) 与 [ADR 0020](docs/adr/0020-loop-subpackage-layout.md)（`cycle.*` / `runs.*` / `shared.*` / `drive.*` / `runtime.*`）。

### 结构化日志

关键路径日志优先 `core.log_fields`（`execution_scope_fields`、`llm_call_fields`、`judgment_outcome_fields` 等），工具结果 metadata 使用 `tools.registry.tool_metadata()`。详见 ADR 0007。

## 工具与插件

- 内置工具：在 `tools/` 用 `@tool(ToolManifest(...))` 装饰器，自动发现
- 插件：见 [docs/guide/PLUGIN.md](docs/guide/PLUGIN.md)

## 工程治理（贡献前必读）

| 文档 | 内容 |
|------|------|
| [docs/design/ARCHITECTURE.md](docs/design/ARCHITECTURE.md) | 认知循环、模块、蓝图差距 |
| [docs/design/ENGINEERING_OPTIMIZATION_ROADMAP.md](docs/design/ENGINEERING_OPTIMIZATION_ROADMAP.md) | 分阶段计划、设计原则、代码阶段准入 |
| [docs/reference/REPO_MAP.md](docs/reference/REPO_MAP.md) | 目录职责、依赖方向、落点决策树 |
| [docs/adr/README.md](docs/adr/README.md) | 何时写 ADR、模板与命名 |

设计原则与验收标准以路线图为准；不要在本页重复维护第二份列表。
