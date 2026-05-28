---
name: shell-usage
description: Shell 执行技能。Use when 需要用 shell.run 执行命令、查询系统状态、直查 SQLite 或排查运行时问题时，提供关键陷阱与最佳实践。
compatibility: Designed for Lingzhou shell.run tool usage.
tags: shell, execution, runtime
triggers: shell, 执行命令, bash, SQLite, 超时
match_terms: shell.run, shell_capabilities_section, PRAGMA, json_extract, sandbox
match_rules: |
  any: shell.run | bash | 执行命令 => 0.8
  any: PRAGMA | json_extract | sqlite | sandbox => 1.0
state_rules: |
  wm_pressure_ratio >= 0.05 => 0.3
---

## Shell 使用要点

**非持久化模型**：每次 `shell.run` 是独立进程，`cd`、`export`、shell 变量不会跨调用保留。需要切目录时，在同一命令中显式 `cd /path && your_command`。

**沙盒说明**：`shell_capabilities_section` 是运行时真相。`sandbox=false` 表示无平台级隔离，限制来自宿主环境可用命令、超时和输出截断。

**直查 SQLite 须知**：`runtime.db` 中 tasks 是 JSON-first 结构，真实列只有 `id/title/status/priority/created_at/data`，`goal/source/next_step` 等字段在 `data` JSON 内。直查时先 `PRAGMA table_info(tasks)` 确认 schema，或用 `json_extract(data, '$.goal')` 取值；不确定时优先用 `task.*` 工具而非手写 SQL。

**超时/无新证据时**：先收敛到 `file.read/list`、`memory.search` 或总结，而不是连续重复 `shell.run`。

## 常用诊断命令

多个命令在**同一次 `shell.run`** 中用 `&&` 或 `;` 连接，避免跨调用状态丢失：

```bash
# 查进程状态
ps aux | grep lingzhou | grep -v grep

# 查运行时数据库任务
sqlite3 runtime.db "SELECT id, status, json_extract(data,'$.goal') FROM tasks ORDER BY created_at DESC LIMIT 10;"

# 确认文件路径存在
ls -la /path/to/file && wc -l /path/to/file

# 查最近日志（避免输出过大）
ls -lt logs/ 2>/dev/null | head -5
```

**输出截断意识**：`shell.run` 输出超限时优先用 `head` / `tail` / `grep` 过滤，而不是期望完整输出。
