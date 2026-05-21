---
name: task.continuity
description: 任务连续性技能。Use when 当前任务已有 next_step、current_step、task inbox 或 steering 信号，需要决定继续推进还是根据新证据转向。
compatibility: Designed for Lingzhou task runtime with active_task / inbox steering.
tags: continuity, task
triggers: next_step, 继续推进, 当前任务
match_terms: current_step, task inbox, steering, old plan
match_rules: |
	any: next_step | 继续推进 | 当前任务 => 0.7
	any: current_step | task inbox | steering | old plan => 1.0
state_bias: has_active_task=0.35, has_next_step=0.85
---
当前任务若有明确 next_step，优先考虑推进它，不要因为轻微信号就分散成新任务。
但如果新用户指令、task inbox 或更强证据表明方向已改变，应以当前事实重新判断。
每一步完成后及时更新 current_step / next_step，让连续性来自事实，而不是惯性。