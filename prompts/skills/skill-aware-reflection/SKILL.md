---
name: skill-aware-reflection
description: "Skill-aware reflection for targeted skill evolution in lingzhou, based on EmbodiSkill (arXiv:2605.10332). When a task fails or underperforms while skills were active, classifies the evidence into 4 types (DISCOVERY / OPTIMIZATION / SKILL_DEFECT / EXECUTION_LAPSE) and applies the correct evolution action. Prevents corrupting valid skill content by separating skill defects from execution lapses. Use when: a task failed or was suboptimal with applied_skills active; before calling skill.evolve after a failure; deciding whether and how to update a skill."
compatibility: Designed for Lingzhou agent runtime. Requires skill.activate, skill.evolve tools.
metadata:
  source: "https://arxiv.org/abs/2605.10332 (adapted for lingzhou)"
  lingzhou:
    tags: "skill, evolution, reflection, failure-analysis"
triggers: skill演化, 技能更新, 失败反思, skill改进, 任务失败后
match_terms: skill.evolve, applied_skills, EXECUTION_LAPSE, SKILL_DEFECT, skill改进, 技能更新
match_rules: |
  any: skill演化 | 技能更新 | skill改进 | skill.evolve => 0.8
  any: 任务失败后 | 效果不及预期 | skill失效 => 0.7
  any: EXECUTION_LAPSE | SKILL_DEFECT | DISCOVERY | OPTIMIZATION => 1.0
state_rules: |
  failure_signal_ratio >= 0.1 => 0.9
---

# Skill-Aware Reflection（技能感知反思）

基于 EmbodiSkill（arXiv:2605.10332）核心原理：**失败任务的证据可能来自技能内容本身的问题，也可能只是执行偏差**。把失败轨迹直接转化为整体技能更新，会误改有效的技能内容，导致技能退化。

---

## 触发时机

`applied_skills` 包含某个 skill，且任务失败或效果不及预期时，**先运行本反思流程，再决定是否调用 `skill.evolve`**。

---

## 4种反思类型

### 成功轨迹（任务成功时可选做）

| 类型 | 含义 | skill.evolve 操作 |
|---|---|---|
| **DISCOVERY** | 轨迹揭示了 skill 未覆盖的有用知识 | `feedback="添加: [新内容]"` |
| **OPTIMIZATION** | skill 内容正确，但轨迹发现了更好的执行方式 | `feedback="优化: 第N条 → [更好方式]"` |

### 失败轨迹（任务失败时必须分类）

| 类型 | 含义 | skill.evolve 操作 |
|---|---|---|
| **SKILL_DEFECT** | skill 指导本身有错 / 不完整 / 描述不准确，导致失败 | `feedback="修正: 第N条 '[原文]' → [正确内容]"` |
| **EXECUTION_LAPSE** | skill 内容正确，但执行中未遵循该条指导 | `feedback="⚠️ 执行提醒: 第N条 '[引用内容]' 在 [场景] 中不能跳过。执行中发生了: [偏差描述]"` |

> **关键区别**：
> - SKILL_DEFECT → 修改技能主体内容
> - EXECUTION_LAPSE → 只在技能末尾增加强调，**不改核心逻辑**

---

## 分类流程

```
1. skill.activate(skill_name) — 重新读取当前技能全文
2. 对照失败轨迹逐条检查技能指导：
   - 这条指导在轨迹中被遵循了吗？
     - 是，且任务仍然失败 → SKILL_DEFECT（指导本身有问题）
     - 否，且如果遵循就不会失败 → EXECUTION_LAPSE（执行偏差）
   - 轨迹揭示了技能完全未覆盖的情形 → DISCOVERY（添加新内容）
   - 轨迹发现了已有指导的更高效方式 → OPTIMIZATION（优化）

3. 无充分证据时（轨迹模糊/环境随机性）→ 不产生 skill.evolve 调用
4. 每条反思信号单独调用 skill.evolve（便于归因和回退）
```

---

## 执行提醒（Appendix Pattern）

EXECUTION_LAPSE 产生的 `skill.evolve` feedback 应引导在技能末尾新增或更新"执行提醒"区块：

```markdown
## ⚠️ 执行提醒

- **[场景描述]**：[引用的有效指导内容]不可跳过。常见偏差：[具体偏差现象]
```

这个区块不引入新规则，只强调现有有效内容，供后续执行时重点关注。

---

## 反例黑名单

| # | 错误做法 | 正确做法 |
|---|---|---|
| 1 | 任务失败就直接 `skill.evolve` 重写整个技能 | 先用4类型分类，只修改有明确证据的部分 |
| 2 | EXECUTION_LAPSE 时修改技能核心逻辑 | 只在技能末尾加 `## ⚠️ 执行提醒`，不动正文 |
| 3 | 无明确证据也产生 skill.evolve 调用 | 无证据 → 不产生更新（`m=0`，保护技能稳定性） |
| 4 | 一次 skill.evolve 打包多个反思信号 | 每条信号单独 evolve，便于回退和归因 |
| 5 | 将环境随机性误判为 SKILL_DEFECT | 先看是否可重现，随机失败不改技能 |

---

## 与 darwin-skill 的协同

- **skill-aware-reflection**（本 skill）= **被动反思**：任务失败后分类证据，决定是否以及如何 evolve
- **darwin-skill** = **主动优化**：计划性地评估 skill 质量、迭代改进

推荐触发顺序：失败发生 → 先运行本 skill 的 4 类分类 → 若发现 SKILL_DEFECT 且改动影响较大，再用 darwin-skill Phase 1 评估改动前后分数，确认改动有净收益后才保留。
