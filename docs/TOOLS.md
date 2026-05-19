# 工具目录

[中文](TOOLS.md) | [English](TOOLS.en.md)

## 概览

当前仓库内置 56 个工具端点。所有工具通过 `@tool(ToolManifest(...))` 注册，由运行时自动发现并加载。

工具设计遵循两条规则：

1. 工具只负责执行，不负责替用户说话。
2. 工具能力通过 manifest 声明，而不是靠调用方硬编码猜测。

## 分类目录

### 1. 文件与配置

- `file.list`
- `file.read`
- `file.write`
- `file.edit`
- `file.delete`
- `config.get`
- `config.set`

### 2. Shell 与进程

- `shell.run`
- `shell.capabilities`
- `exec`
- `process.list`
- `process.poll`
- `process.log`
- `process.write`
- `process.kill`

### 3. 任务、计划与调度

- `task.add`
- `task.advance`
- `task.complete`
- `task.list`
- `task.update`
- `task.fail`
- `task.wait`
- `task.resume`
- `task.ask`
- `task.plan`
- `schedule.add`
- `schedule.list`
- `schedule.ack`
- `schedule.cancel`

### 4. 记忆与反思

- `memory.add_wm`
- `memory.add_semantic`
- `memory.set_fact`
- `memory.search`
- `memory.get_fact`
- `memory.snapshot`
- `failure.dismiss`
- `reflect.structural`

### 5. 网页、浏览器与媒体

- `web.fetch`
- `web.search`
- `browser.navigate`
- `browser.snapshot`
- `browser.click`
- `browser.type`
- `browser.scroll`
- `image.analyze`
- `image.generate`
- `tts.speak`

### 6. 技能、探针与通知

- `skill.list`
- `skill.search`
- `probe.install`
- `probe.remove`
- `probe.run`
- `probe.list`
- `probe.disable`
- `probe.enable`
- `wechat.send`

## Capability 标签

部分工具会声明 capability，用于判断层和执行层治理：

| capability | 含义 |
|------------|------|
| `ask_evidence` | 属于本地取证工具，适合在追问用户前先跑一轮 |
| `plan_bootstrap_exempt` | 允许在没有结构化 plan 的情况下先执行 |
| `plan_alignment_exempt` | 不受 plan 对齐门限制 |
| `completion_info_only` | 属于完成校验链路中的只读信息工具 |
| `completion_mutation` | 属于完成前的实际状态变更工具 |
| `completion_verify` | 属于完成前的验证工具 |

这些标签的作用是把“工具名硬编码”替换成“能力声明驱动”，让判断层根据能力而不是名字做决策。

## 工具返回约定

工具统一返回 `ToolResult`，常见字段包括：

- `summary`：给判断层和日志看的结构化摘要
- `evidence`：更详细的证据内容
- `error`：错误文本
- `skipped`：是否被跳过
- `state_delta`：本轮对任务或环境的状态增量
- `metadata`：调试或下游推理所需的附加数据

## 什么时候该扩展新工具

只有当以下条件成立时，才建议新增工具：

- 该动作具备稳定输入输出边界，适合被重复调用
- 该能力不应混进判断层 prompt 中伪装成“知识”
- 现有工具组合无法清晰表达该执行意图

如果只是提示词策略变化、路由规则变化或记忆利用变化，应优先改判断层、prompt 或 capability，而不是盲目新增工具。