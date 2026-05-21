---
name: failure.reflection
description: 失败反思技能。Use when 已积累失败信号、连续重试无效、或需要区分参数错误、环境缺失、前提不满足与策略错误。
compatibility: Designed for Lingzhou failure handling and recovery loops.
tags: failure, reflection
triggers: 失败, 报错, 根因, 重试
match_terms: retry, blocked, root cause, recover
match_rules: |
	any: 失败 | 报错 | 根因 | 重试 => 0.7
	any: retry | blocked | root cause | recover => 1.0
state_bias: failure_signal_ratio=1.4
---
你已经积累了失败信号时，先停下来分析根因，不要重复同一种动作。
区分是参数错误、前提不满足、环境缺失，还是策略本身错了。
如果证据仍不足，就补证据；如果路径已被证伪，就主动换策略或向用户说明阻塞点。