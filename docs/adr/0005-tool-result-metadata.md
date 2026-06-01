# ADR 0005: ToolResult metadata 统一字段

- Status: Accepted
- Date: 2026-06-01

## Context

工具 `metadata` 中 `log_summary` 已在使用，但缺少统一的 `tool_name` 与构造入口，日志与执行层难以一致解析。

## Decision

- 在 `tools.registry` 增加 `tool_metadata(tool_name, log_summary, **extra)`。
- 所有 `tools/` 模块的成功返回路径统一经 `tool_metadata()` 构造；跳过/参数错误等可省略 metadata。
- 已覆盖：`shell`/`file`/`memory`/`web`/`exec`/`image`/`task`/`browser`/`probe`/`config`/`plan`/`skill`/`subagent`/`tts`/`ask` 等。

## Validation

- `tests/test_judgment_layers.py::test_tool_metadata_shape`；相关 tools/core 测试通过。
