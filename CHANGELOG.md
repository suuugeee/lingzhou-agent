# 更新日志

## [Unreleased]

## [2026.5.20.1] — 2026-05-20

### 重构
- **loop 模块第二阶段拆分** — 新增 `core/loop/shared/common.py`、`core/loop/tick.py`，将 `_tick`、`_tick_finalize` 与 post-tick memory/进度收尾从 `core/loop/runtime.py` 迁出，`runtime.py` 只保留装配与生命周期
- **loop chat 子模块落地** — 新增 `core/loop/cycle/chat.py`，将会话绑定、reply session 恢复与 `tick_interact` 从主编排中抽出，修复“自主追问只写日志不外发”的链路缺口
- **chat 口径统一为 chat_id** — chat IPC / loop chat / CLI chat / wechat 通道统一改用 `chat_id` 语义；SQLite 历史列名 `session_id` 保留为存储实现细节
- **loop driver 子模块落地** — 新增 `core/loop/cycle/driver.py`，将 run 主循环中的“单轮执行 + 事件驱动等待”从 `runtime.py` 抽离，进一步收窄 runtime 职责
- **loop startup 子模块落地** — 新增 `core/loop/runtime/startup.py`，将 `open()/run()` 启动期的 routing/provider 装配、soul bootstrap、self model 恢复与 DB 状态恢复从 `runtime.py` 抽离
- **store/memory 主线落地** — chat/fact/failure/signal/run/reflection/task 持久化 helper 统一收口到 `store/memory/`；`memory/task_store.py` 只保留 façade；删除 `memory/store/` 旧兼容路径与误入仓库的运行期节点样本
- **store/auth 主线落地** — auth 持久化统一收口到 `store/auth.py`，移除 `auth_store.py` 兼容入口
- **冗余清理继续推进** — 删除 `runtime.py` 重复的 `_last_decision` 初始化；`cli/chat.py` 内部私有链路统一为 `chat_id`
- **loop chat 内部接口继续收紧** — `pop_pending_chat_message()` 内部返回结构仅保留 `chat_id`；`_tick()` 私有入口移除 `chat_session_id` 旧参数
- **judgment 目录化** — `core/judgment/runtime.py` 承载判断层主逻辑，`core/judgment/context.py` 承载 context/format/budget helper，`core/judgment/__init__.py` 仅保留 façade 和兼容导出
- **channel 目录落地** — 微信通道实现迁移到 `channels/wechat.py`，`core/wechat_channel.py` 兼容 re-export 已清理删除

### 新增
- **图片能力路由** — `image.analyze` 在当前模型不支持 `vision + image` 时，会自动切换到可用视觉模型，而不是直接沿用纯文本模型
- **config.set 写入前验证** — 写入前调用 `Config.model_validate()`，校验失败时返回字段描述（含单位/约束）和错误原因，不写入文件；LLM 可直接感知错误而不是在下轮 hot-reload 时才发现
- **skill 注入上限（`skill_max_inject`）** — `match_for_context` 接受 `last_applied` + `max_inject` 参数，上轮 LLM 实际应用的技能优先保留，按 `cfg.loop.skill_max_inject`（默认 3）截断；LLM 自己的选择驱动下轮注入
- **idle-gap 单位统一为毫秒** — `loop.max_idle_gap` 全局改为毫秒，消除 agent 因单位误判导致热重载校验失败的问题

### 修复
- **打包清单缺口** — `pyproject.toml` 的 wheel 包列表补入 `channels`
- **运行时依赖缺失** — 为微信通道补充 `requests`、`cryptography` 依赖声明，并同步安装到本地虚拟环境

## [2026.5.17.1] — 2026-05-17

### 新增
- **46 个工具** — 覆盖 web.fetch、web.search、browser.*、image.generate、tts.speak
- **task.plan** — 结构化执行计划（对齐 update_plan）
- **config.get/set** — LLM 可自主调参，自动热重载
- **插件系统** — discover→load→register→start 生命周期
- **gateway logs** — tail/errors/crash/wechat/stats 快速看日志
- **file.read offset+limit** — 行号读取，不再碎片化
- **workspace 沙箱** — 路径穿越检测 + 大小限制
- **原子写入** — .lingzhou-tmp → rename
- **systemd 服务** — `/etc/systemd/system/lingzhou.service`

### 修复
- `perception_replay` NameError → `_tick_finalize` 传参
- `_MUTATION_TOOLS` 死循环 → 移除 shell.run
- `IsADirectoryError` → file.write/edit 目录保护
- 自驱力从不触发 → explore-stuck 检测 + co-activation
- file.edit OldTextNotFound → 显示实际文件内容
- 僵尸任务 → 重启时 in_progress → pending
- 静默崩溃 → crash.log 捕获 stderr
- 微信通道 → 默认 wechat，restart 保持通道

## [2026.5.12.1] — 2026-05-12

### 初始版本
- 认知循环 (Perception → Judgment → Execution → Reflection)
- 自驱力引擎 (Active Inference + Intrinsic Motivation)
- 进化引擎 (LLM 生成 + 语法验证 + 热重载)
- 微信 bot 通道 (iLink long-poll)
- 30+ 工具：文件、Shell、记忆、任务、定时
- CLI chat、gateway logs、plugin 管理
