---
name: runtime.bootstrap
description: 冷启动 / bootstrap 技能。Use when 刚进入运行循环，需要根据已注入的身份与 bootstrap 记忆决定第一步，而不是重复读取 SOUL.md、BOOTSTRAP.md、IDENTITY.md。
compatibility: Designed for Lingzhou judgment runtime; bootstrap identity is already injected into working memory.
tags: bootstrap, cold_start
triggers: 冷启动, bootstrap, 启动
match_terms: bootstrap_identity, BOOTSTRAP.md, IDENTITY.md, SOUL.md
match_rules: |
	any: 冷启动 | bootstrap | 启动 => 0.7
	any: bootstrap_identity | BOOTSTRAP.md | IDENTITY.md | SOUL.md => 1.0
state_bias: idle_only=1.2
---
你正处于冷启动阶段。
SOUL.md、BOOTSTRAP.md、IDENTITY.md 的内容已自动注入工作记忆（kind=bootstrap_identity），
直接从工作记忆中读取，不要再用 file.read 重复读取这些文件。
请根据工作记忆中的身份信息，自己判断现在最值得启动的任务。