# ADR（Architecture Decision Record）说明

本目录用于记录 Lingzhou 的关键架构决策，确保“为什么这么做”可追溯，而不只保留“最后代码长什么样”。

## 何时必须写 ADR

满足任一条件即需要新增 ADR：
- 主循环语义变化（并发、顺序、状态推进、热重载语义）
- provider 协议/认证链路变化
- memory/store 的 schema 或持久化语义变化
- 工具契约（`ToolResult` 核心字段）变化

## 命名规范

文件名格式：

`NNNN-<short-kebab-title>.md`

示例：
- `0001-run-reload-atomicity.md`
- `0002-provider-token-validation-boundary.md`

## 状态规范

`Status` 字段允许值：
- `Proposed`
- `Accepted`
- `Superseded`（需指向替代 ADR）
- `Deprecated`

## 编写流程

1. 复制模板：`docs/adr/0000-template.md`
2. 填写背景、决策、影响与验证方案
3. 在 PR 描述中引用该 ADR
4. 合并后将状态更新为 `Accepted`（若已达成）

## 与其他文档关系

| 文档 | 记录什么 |
|------|----------|
| 本目录（ADR） | 单次关键决策的背景、取舍、验证 |
| [ARCHITECTURE.md](../design/ARCHITECTURE.md) | 认知架构与蓝图差距 |
| [ENGINEERING_OPTIMIZATION_ROADMAP.md](../design/ENGINEERING_OPTIMIZATION_ROADMAP.md) | 分阶段工程计划与准入条件 |
| [REPO_MAP.md](../reference/REPO_MAP.md) | 目录职责与依赖边界 |
| `docs/reference/*.md` | 配置、工具等接口契约 |
