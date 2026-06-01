# Run 驱动循环架构设计

> 状态：草案（Phase 3 规划）  
> 当前架构：tick 驱动，Run 仅做日志  
> 目标架构：Run 成为内循环一等公民

---

## 一、现状

### 当前 tick 驱动流程

```
dispatcher
  └─ _one_tick()
       ├─ 判断（judge）
       ├─ 执行（execution.py → add_run … update_run）
       └─ 结晶 / 代谢 / 情绪
```

Run 在 `core/execution.py` 作为执行的副产物被写入：
1. `add_run(status="running", run_type="tool_chain", model_tier=...)`
2. 工具链执行完毕后 `update_run(status="succeeded"/"failed", output_json=...)`

Run 目前的作用：日志、探针可查、子灵只读上下文注入。  
Run 目前**不**驱动循环——tick 生命周期与 run 生命周期松散耦合。

---

## 二、目标架构（Run-driven inner loop）

### 核心思想

> 内循环的最小调度单位从 "tick" 升级为 "Run"。  
> 每次决策产出一个 Run，Run 的完成（succeeded/failed）决定下一次调度时机。

```
dispatcher
  └─ 创建 Run（status="pending"，run_type 由决策策略选定）
       └─ RunDriver.execute(run)
            ├─ 按 run_type 路由到执行器
            ├─ 执行完毕 → update_run(status="succeeded/failed")
            └─ 触发下一轮 dispatcher 调度
```

### Run 类型表（run_type）

| run_type | 触发条件 | 执行路径 | 典型模型档位 |
|----------|----------|----------|-------------|
| `judge` | 无活跃任务，idle 判断 | 仅 LLM 判断，无工具 | `reader` |
| `tool_chain` | 有活跃任务，act=True | 当前 execution.py 路径 | `model_tier` from task |
| `chat_reply` | 用户消息进入 | chat 回复路径 | `reader` / `model` |
| `evolve` | 进化触发条件满足 | evolution.evolve 工具 | `reasoner` |
| `subagent` | 子灵任务 | subagent.py | 按任务策略 |
| `probe` | 探针定时 | probe/runner.py | `reader` |

### Run 状态机

```
pending → running → succeeded
                 └→ failed → (可重试 → pending)
```

- `pending`：已创建，等待调度器认领
- `running`：RunDriver 正在执行
- `succeeded`：执行正常结束
- `failed`：执行出错；是否重试由 `run_type` 的重试策略决定

---

## 三、关键设计决策

### 3.1 model_tier 随 run_type 静态路由

每个 `run_type` 在 `provider/models.json`（或内置路由表）中声明默认档位：

```json
{
  "run_type_routing": {
    "judge":      "reader",
    "tool_chain": "task_default",
    "chat_reply": "reader",
    "evolve":     "reasoner",
    "subagent":   "task_default"
  }
}
```

`task_default` 表示继承活跃任务的 `model_tier`。

### 3.2 并发安全

- 同一任务链（`task_id` 相同）上的 Run 仍严格串行。
- 不同任务链的 Run 可在 `max_concurrent_ticks` 范围内并发。
- 并发判断逻辑复用现有 `loop/task/parallel.py` 隔离机制。

### 3.3 生命史账本与 Run 对齐

代谢引擎写账本时携带 `run_id`（已在 `StateProposal.extras` 预留位），使每条状态变更可溯源到产生它的那次 Run。

### 3.4 免疫层不变

所有写入仍经过 `MetabolicEngine.submit()` → `check_tool_blocked()`，Run 驱动不绕过免疫层。

---

## 四、迁移路径（增量、可回滚）

### Phase 3a：Run 携带 run_type 语义（最小改动）

- 执行前按决策类型写 `run_type`（`judge`/`tool_chain`/`chat_reply`/`evolve`）
- 不改变调度逻辑，仅填充 `run_type` 字段
- 验证：`lingzhou logs runs --type judge` 可筛选纯判断 tick

### Phase 3b：RunDriver 路由层

- 新建 `core/loop/runs/driver.py`：接收 `Run`，按 `run_type` 分发到对应执行器
- tick.py 委托给 `RunDriver`，保持单入口
- 验证：现有测试 + `tests/test_run_driver.py`

### Phase 3c：model_tier 随 run_type 静态路由

- `provider/models.json` 增加 `run_type_routing` 段
- `execution.py` 在 `add_run` 前按 `run_type_routing` 解析 `model_tier`
- 去掉 tick.py 中硬编码的档位推断逻辑

### Phase 3d：Run-pending 驱动调度器

- dispatcher 直接 poll `runs WHERE status='pending'` 而非依赖 asyncio.Queue
- Run 的 `created_at` 作为调度优先级
- 现有 `loop/loop.py` 改为初始化时写入一个 bootstrap Run

---

## 五、对现有接口的影响

| 模块 | 影响 | 优先级 |
|------|------|--------|
| `core/execution.py` | `add_run` 新增 `run_type` 参数（已有字段，兼容） | Phase 3a |
| `store/task/run.py` | `add_run` 支持 `status="pending"` | Phase 3b |
| `core/loop/tick.py` | 委托给 RunDriver，tick 成为 RunDriver 的适配器 | Phase 3b |
| `provider/models.json` | 新增 `run_type_routing` | Phase 3c |
| `core/loop/loop.py` | bootstrap Run 初始化 | Phase 3d |
| `tests/` | 新增 `test_run_driver.py` | Phase 3b |

---

## 六、不在本次范围内

- 跨宿主分布式 Run 调度（单宿主约束，见宪法）
- Run 与外部队列（如 Redis）集成
- Run 级别的用量计费（可在 Phase 3c 后自然扩展）
