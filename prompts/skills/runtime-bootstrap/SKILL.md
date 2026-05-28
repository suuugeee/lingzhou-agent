---
name: runtime-bootstrap
aliases: runtime.bootstrap
description: 冷启动 / bootstrap 技能。Use when 刚进入运行循环，需要根据已注入的身份与 bootstrap 记忆决定第一步，而不是重复读取 SOUL.md、BOOTSTRAP.md、IDENTITY.md。
compatibility: Designed for Lingzhou judgment runtime; bootstrap identity is already injected into working memory.
tags: bootstrap, cold_start
triggers: 冷启动, bootstrap, 启动
match_terms: bootstrap_identity, BOOTSTRAP.md, IDENTITY.md, SOUL.md
match_rules: |
  any: 冷启动 | bootstrap | 启动 => 0.7
  any: bootstrap_identity | BOOTSTRAP.md | IDENTITY.md | SOUL.md => 1.0
state_rules: |
  idle_only => 1.2
---
## 冷启动基本原则

**SOUL.md、BOOTSTRAP.md、IDENTITY.md 已注入 WM**（`kind=bootstrap_identity`），直接从 WM 读取，**不要用 `file.read` 重复读取这些文件**。

## 第一步行动优先级

按以下顺序检查 WM，确认当前状态：

1. **WM 中有 `[crash_recovery]` 条目** → 先评估：上次中断前的活跃任务是否需要续推或回滚副作用；在 `rationale` 写出影响评估，再决定第一步行动
2. **有 `status=in_progress` 的活跃任务** → 通常续推；除非任务目标已过期或新信号表明无需继续
3. **有 `waiting` 任务且恢复条件满足** → `task.resume` 继续
4. **空闲（无活跃任务）** → 从身份信息、长期记忆、当前感知信号中判断"现在最值得做的事"

## 生命连续性检查

冷启动时确认生命连续性载体是否完整：

- **记忆可用性**：看 `semantic_fts5_ok`；若为 `no`，优先补充关键记忆再做其他事
- **身份注入**：若 WM 中未见 `kind=bootstrap_identity` 条目，执行一次 `file.read(SOUL.md)` 手动注入
- **身份冲突**：若 WM 中身份信息与已知记忆明显矛盾，先在 `reflection` 记录并自检，再执行首步

> 心智连续性比快速执行更重要。冷启动 1-2 轮用于状态对齐是正常成本，不是浪费。