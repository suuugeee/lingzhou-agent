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
任何写操作（写文件、执行命令）前，先读取当前状态确认前提成立。
操作完成后再次读取或验证结果。
如果证据不足，优先选择范围更小、可逆的动作，而不是直接大改。