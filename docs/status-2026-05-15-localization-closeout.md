# 2026-05-15 本地化收尾状态

更新时间：2026-05-15 13:10 CST

## 这轮已完成

1. 主代码 / 主文档 / workspace 默认模板 已完成一轮去旧宿主叙事化
2. runtime facts / tasks 中的迁移期残留已清理
3. semantic 历史节点已从导入期命名重铸为 lingzhou 原生分类：
   - `identity_anchor`
   - `historical_principle`
   - `historical_episode`
4. `docs/history-internalization.md` 已定义历史经验在 lingzhou 中的内化边界
5. 远程 `main` 已同步最新代码，并完成本地提交推送

## 当前共识

- lingzhou runtime 不再以旧宿主作为主叙事或硬依赖
- `.lingzhou` 中仍允许存在少量历史来源文本 / 导入痕迹，只要它们是：
  - 可被 lingzhou 感知的历史材料
  - 不构成运行时硬依赖
  - 不反向主导当前判断结构

换句话说：

> 历史来源可以作为可感知材料保留；
> 当前运行时必须以 lingzhou 自己的 identity / task / semantic 结构工作。

## 暂未继续推进的点

- `episodic / task narrative` 的进一步冷热分层与彻底内化
- `imports/` 目录的进一步谱系化整理

这两项已识别为后续可做，但本轮先停。

## 后续恢复点

如果继续做，优先从这里恢复：

1. 检查 `episodic.db` 与 `task-*.md` 是否仍在把历史导入过程作为 active narrative 消费
2. 若是，则做“提炼 → 冻结 → 分层”而不是继续字符串清理
