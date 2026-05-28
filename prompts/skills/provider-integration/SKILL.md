---
name: provider-integration
aliases: provider.integration
description: 工具与 provider 集成技能。Use when 工具调用失败、参数名不匹配、文件不存在、或需要先分析错误原因再决定重试策略。
compatibility: Designed for Lingzhou tool calling and provider integration flows.
tags: act, tool_call
triggers: 工具失败, 参数错误, file not found, 调用失败
match_terms: FileNotFound, 参数名, 参数类型, tool call fails
match_rules: |
  any: 工具失败 | 参数错误 | file not found | 调用失败 => 0.7
  any: FileNotFound | 参数名 | 参数类型 | tool call fails => 1.0
state_rules: |
  failure_signal_ratio >= 0.1 => 0.8
---
## 调用前校验（防御）

调用任何工具前，核对 `tools_section[].description` 中的参数约束：

- 确认参数名与文档一致（大小写、下划线、驼峰）
- 确认所有必填参数已提供，可选参数有合理默认
- 路径类参数：确认目标已存在（WM 有记录 or `file.list` 确认）

## 错误分类与恢复

| 错误类型 | 特征 | 恢复动作 |
|---|---|---|
| **参数名/类型错误** | 400 / `missing required field` / `unexpected argument` | 对照工具描述修正参数，重试 1 次 |
| **文件不存在** | `FileNotFound` / `ENOENT` | 不重复尝试同路径；`file.list` 确认目录；路径不存在则换策略 |
| **服务暂不可用** | timeout / 503 / 连接拒绝 | 等待后重试 1 次；仍失败 → `task.wait(wait_kind=external)` |
| **工具能力不符** | 工具存在但不支持该操作 | 换工具；查 `tools_section` capability 标签 |
| **权限不足** | 403 / permission denied | 确认环境配置；无法解决 → `reply_to_user` 说明 |

## 失败后决策原则

1. **一次失败**：修正参数后重试；在 `rationale` 写出"这次不同在哪里"
2. **同类失败连续 ≥ 2 次**：停止重试，触发 `failure-reflection` 深入分析
3. **`durable_failure` 窗口内**：不重试，换动作或等外部变化
4. **每次重试必须有明确修正理由**：无理由则不重试，避免盲目消耗 token

## 反例黑名单

| 反模式 | 正确做法 |
|---|---|
| 用相同参数重试失败的工具调用 | 先分析错误原因，修正后再试 |
| `ENOENT` 后继续尝试同路径 | `file.list` 确认目录状态，换路径/策略 |
| 超时后立刻重试 | 等待或 `task.wait`，不立刻重试 |
| 忽视工具描述中的参数约束 | 调用前必须核对工具描述 |