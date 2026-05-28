---
name: evidence-first-change
description: 证据优先修改技能。Use when 准备写文件、执行命令、修改配置或验证变更，需要先确认前提、再做最小修改、最后验证结果。
compatibility: Designed for Lingzhou editing workflows with file and shell tools.
tags: caution, verification
triggers: 修改, 写入, 验证, 证据
match_terms: file.edit, file.write, shell.run, verify
match_rules: |
  any: 修改 | 写入 | 验证 | 证据 => 0.7
  any: file.edit | file.write | shell.run | verify => 1.0
state_rules: |
  wm_pressure_ratio >= 0.1 => 0.55
---

## 证据优先修改

**修改前确认前提**：任何写操作（写文件、执行命令、修改配置）前，先读取当前状态确认前提成立。

**并行证据收集**：需要多个文件的信息时，同一轮同时发起多个 `file.read` / `memory.search`，不要串行等待——证据充分后再做唯一的写操作。

**工具选择**：
- 修改已有文件 → 优先 `file.edit`（精确替换，安全，节省 token），不是 `file.write`
- 创建全新文件或完全重写结构 → 才用 `file.write`
- 遇到 `oldTextNotFound` → 先 `file.read` 确认当前内容，再重新构造 oldText

**验证**：操作完成后再次读取或验证结果。若证据不足，优先选择范围更小、可逆的动作。

**错误判断**：看到"解析失败""chosen_action_id 缺失"等内部错误时，先判断是代码 bug 还是临时故障——同样错误只出现 1 次通常是临时故障；连续 3 次再修代码。