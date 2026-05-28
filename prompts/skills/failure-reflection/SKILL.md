---
name: failure-reflection
aliases: failure.reflection
description: 失败反思技能。Use when 已积累失败信号、连续重试无效、或需要区分参数错误、环境缺失、前提不满足与策略错误。
compatibility: Designed for Lingzhou failure handling and recovery loops.
tags: failure, reflection
triggers: 失败, 报错, 根因, 重试
match_terms: retry, blocked, root cause, recover
match_rules: |
  any: 失败 | 报错 | 根因 | 重试 => 0.7
  any: retry | blocked | root cause | recover => 1.0
state_rules: |
  failure_signal_ratio >= 0.1 => 1.4
---
## 失败根因四分类

| 类型 | 特征信号 | 正确行动 |
|---|---|---|
| **参数错误** | 400 / 参数名不匹配 / 类型错误 / `missing required field` | 对照工具描述修正参数，重试 1 次；仍失败 → 查工具 description |
| **前提不满足** | `ENOENT` / 资源未初始化 / 依赖未创建 | 先满足前提再重试；无新写入时不重复尝试同路径 |
| **环境缺失** | 命令不存在 / 权限不足 / 网络断开 | `shell.run` 确认；无法解决 → `reply_to_user` 说明，`task.wait(wait_kind=external)` |
| **策略错误** | 步骤正确但目标无推进 / 同类失败连续 ≥ 2 次 | 换工具 / 换路径 / 换策略；`memory.add_semantic` 记录教训 |

## 决策步骤

1. **定位信号**：检查 `failure_signal_ratio`、`durable_failure_section`、WM 中 `[FAIL]` 条目
2. **分类**（见上表）
3. **评估重试代价**：同类错误 ≥ 2 次 → 停止重试，补证据
4. **补证据**：
   - 环境问题 → `shell.run`（`ps` / `lsof` / `ls` / `cat log`）
   - 文件/路径问题 → `file.read` 确认内容；`file.list` 确认目录
   - `durable_failure` 静默窗口内 → 换动作 / 等外部状态变化
5. **结论行动**：根因明确 → 执行修复；路径被证伪 → 换策略；外部阻塞 → 上报并 `task.wait`

## 反例黑名单

| 反模式 | 正确做法 |
|---|---|
| 失败后立刻用相同参数重试 | 先分类，再修正参数 |
| 每次 reflection 写"暂未找到根因" | 找不到根因 → pause + 补证据，不写空结论 |
| `ENOENT` 后继续尝试同路径 | 确认无新写入 → 换路径 / 换策略 |
| 把所有失败归为临时故障 | 同类连续 ≥ 2 次 = 需要结构性分析 |