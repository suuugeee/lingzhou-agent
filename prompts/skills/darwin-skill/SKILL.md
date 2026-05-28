---
name: darwin-skill
description: "Lingzhou skill quality evaluator and optimizer, inspired by Karpathy autoresearch and the darwin-skill project (github.com/alchaincyf/darwin-skill). Evaluates SKILL.md files against a 5-dimension rubric (description quality, workflow clarity, failure branches, actionable specificity, effectiveness), runs targeted improvement via skill.evolve, and only keeps changes that measurably improve quality (ratchet mechanism). Use when: user mentions '优化skill', 'skill质量', 'skill评分', 'evaluate skills', 'darwin', 'skill review'; or when a skill repeatedly fails to trigger or produces wrong outputs."
compatibility: Designed for Lingzhou agent runtime. Requires skill.list, skill.activate, skill.evolve tools.
metadata:
  source: "https://github.com/alchaincyf/darwin-skill (adapted for lingzhou)"
  license: MIT
  lingzhou:
    tags: "skill, evolution, quality, optimization, meta"
triggers: 优化skill, 改进技能, skill质量, skill评分, skill review, darwin
match_terms: skill.evolve, skill.list, skill质量, darwin, 优化skill, skill评分, skill review
match_rules: |
  any: 优化skill | 改进技能 | skill质量 | skill评分 | skill review | darwin => 0.9
  any: optimize skill | evaluate skill | skill打分 => 0.9
  any: skill.evolve | skill.synthesize => 0.4
state_rules: |
  wm_pressure_ratio >= 0.1 => 0.2
---

# Darwin Skill（达尔文技能优化器）

受 [Karpathy autoresearch](https://github.com/karpathy/autoresearch) 和 [darwin-skill](https://github.com/alchaincyf/darwin-skill) 启发，将持续优化循环搬到 lingzhou 的 skill 演化系统。

**核心理念**：评估 → 找最弱维度 → 改进 → 实测验证 → 只保留改进（棘轮）→ 人工确认

---

## 评估 Rubric（5维度，满分100）

| # | 维度 | 分值 | 评分标准 |
|---|---|---|---|
| 1 | **描述质量** | 12 | `description` 含"做什么+何时用+触发词"；无"灵活应用/根据情况"等空话尾巴 |
| 2 | **工作流清晰度** | 20 | 步骤有序号、每步有明确输入/输出；无"建议/可以考虑/视情况而定"等软化措辞 |
| 3 | **失败分支** | 20 | 显式写出"如果 X 失败 → Y"的分支；有 fallback 路径；只写正向流程扣 ≥3 分 |
| 4 | **具体可执行性** | 8 | 有具体工具名/参数/示例；无模糊指代 |
| 5 | **实测表现** | 40 | 带 skill 执行测试 prompt vs 不带 skill；输出质量是否明显提升 |

> **实测维度权重最高（40分）**。格式完美但跑出来效果差的 skill 分数低。
>
> 维度 2/3/4 是相关簇——修 dim3 时 dim2 常跟着涨；找最低维度时同时看相关簇再决定是否同步改。

---

## 优化流程

### Phase 0: 准备

1. `skill.list` 确认优化范围（全部 or 指定 skill）
2. 为每个 skill 设计 2-3 个测试 prompt（最典型使用场景，不是边缘 case）
3. 🔴 **CHECKPOINT · STOP**：展示测试 prompt，等用户确认后继续

### Phase 1: 基线评估

```
for each skill:
  1. skill.activate(name) 读取全文
  2. 按5维度逐项打分（1-10分 × 权重），附简短理由
  3. 模拟执行测试prompt（带skill vs 不带skill），打维度5分
  4. 计算总分，记录基线
```

展示评分卡（技能名 / 基线分 / 最弱维度）后 🔴 **CHECKPOINT**，等用户确认

### Phase 2: 优化循环（棘轮）

```
for each skill（按基线分升序，先优化最弱的）:
  round = 0
  while round < 3:
    round += 1
    1. 找得分最低的维度（注意相关簇 dim2/dim3/dim4）
    2. 生成1个具体改进方案（改哪段、为什么、预期+多少分）
    3. skill.evolve(name, feedback="[具体说明]")
    4. 重新评估（维度1-4静态分析，维度5重新模拟执行）
    5. 新分 > 旧分 → 保留；否则 → skill.evolve(name, feedback="回退：[原内容描述]")
    6. 连续2轮 Δ < 2分 → 触顶信号，break（不要硬凑轮次）
  🔴 CHECKPOINT：展示改动摘要+分数变化，等用户确认
```

### Phase 3: 汇总报告

展示所有 skill 的 before / after / Δ + 主要改进内容

---

## 使用方式

- **「优化所有 skills」** → Phase 0-3 完整流程
- **「优化 xxx skill」** → 只对指定 skill 执行 Phase 0-2
- **「评估 skill 质量」** → 只执行 Phase 0-1（不改动）

---

## 反例黑名单（每轮改动前对照一次）

| # | 反模式 | 为什么不做 | 替代做法 |
|---|---|---|---|
| 1 | 改完立刻自评 | LLM 乐观偏差，准确率接近随机（SkillLens 实证 46.4%） | 用独立子问题重新评估 |
| 2 | 一次改多个维度 | 分数升降无法归因 | 每轮只改 1 个维度 |
| 3 | 为凑分加冗余段落 | 体积膨胀质量不变 | 触顶（连续2轮 Δ < 2）就 break |
| 4 | 跳过测试 prompt 直接评分 | 维度5形同虚设 | Phase 0 强制设计 2-3 个 prompt |
| 5 | 静默跳过评分异常 | 破坏评估完整性 | 异常先告知用户再处理 |
| 6 | 不带 skill 基线就直接打分 | 无法判断 skill 是否真的有用 | 至少做一次"有/无 skill"对比思考 |

---

## 与 skill-aware-reflection 的协同

- **Darwin**（本 skill）= **主动评估与优化**：计划性地对 skill 质量打分、迭代改进
- **skill-aware-reflection** = **被动失败分类**：任务失败后先分类证据，再决定是否 evolve

推荐工作流：先用 darwin Phase 0-1 做基线评估；Phase 2 改进时，对每次 evolve 的结果都同步运行 skill-aware-reflection 的 4 类分类，确保改动来自真实 SKILL_DEFECT 而非 EXECUTION_LAPSE。
