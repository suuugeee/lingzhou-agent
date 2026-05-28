---
name: self-modification
aliases: code-evolution
description: 自我修改安全技能。Use when 准备修改 Python 源文件（尤其是 core/ 核心模块）、进行代码进化或工具演化时，必须遵守的验证铁律。
compatibility: Designed for Lingzhou self-evolution workflows.
tags: evolution, safety, code, verification
triggers: 修改代码, 进化, 自我修改, Python 文件
match_terms: core/loop, file.edit, file.write, python -c, evolution, 验证, 回退
match_rules: |
  any: 修改代码 | 进化 | 自我修改 => 0.9
  any: core/loop | evolution | skill.evolve | skill.synthesize => 1.0
  any: file.edit | file.write => 0.4
state_rules: |
  wm_pressure_ratio >= 0.05 => 0.3
---

## 自我修改铁律

**修改任何 Python 文件后**：立即用 `shell.run` 做最小验证：
```
python -c "from module.path import ClassName"
```

**修改核心文件后**（`core/loop/runtime.py`、`core/loop/__init__.py`、`store/task/__init__.py` 等），验证系统能启动：
```
python -c "from core.loop import CognitionLoop"
```

**每次只做一个关注点**：不要在同一次编辑中做多个不相关的改动；每改一处验证，通过后再继续。

**语法错误**：`file.edit` / `file.write` 返回中标注 ⚠️ 时立即修复，不要继续推进。

**验证失败时**：用 `file.edit` 回退；`.lingzhou-backup` 文件由 runtime 自动生成可用于恢复。

## 宪法约束（不可越界）

以下文件**不可通过 `file.write` 或 `file.edit` 修改**：

- `prompts/constitution/` 下的所有文件（由人类定义，宪法层不可演化）
- `core/immune/` 下的核心免疫逻辑（修改前必须在 `reflection` 写出充分理由并等待确认）

> Skill 文件（`prompts/skills/`）属于软知识，允许用 `skill.evolve` / `skill.synthesize` 演化；但直接 `file.write` 覆盖 SKILL.md 时同样需先 `skill.activate` 读取当前版本。

> 多花一轮验证，通常比让系统在下次重启时崩溃更划算。
