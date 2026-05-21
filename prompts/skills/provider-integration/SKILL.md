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
调用工具前，确认参数名和类型符合工具描述。
工具调用失败时，先分析错误原因再重试，不要盲目重复相同参数。
如果某个文件不存在（FileNotFound），不要反复尝试读取，换一个策略。