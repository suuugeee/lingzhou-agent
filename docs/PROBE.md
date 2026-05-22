# 探针系统（Probe System）

[中文](PROBE.md) | [English](PROBE.en.md)

灵舟的探针系统是 LLM 自主布放的感知网络。LLM 可以随时安装、拆除、触发探针，
将外部信息（进程状态、服务健康、文件变化、HTTP 接口等）拉入自身的感知上下文。

> **核心哲学**：探针不是固定的监控配置，而是 LLM 在认知过程中主动伸出的感知触手。
> 何时布放、监控什么、如何响应，全由 LLM 自主判断。

---

## 1. 探针类型（kind）

| kind     | spec 内容          | 说明                                |
|----------|--------------------|-------------------------------------|
| `shell`  | Shell 命令字符串   | 执行命令，stdout 作为结果            |
| `http`   | URL                | GET 请求，响应体作为结果             |
| `python` | Python 代码片段    | 执行代码，stdout 作为结果（`print`）  |

---

## 2. 触发方式（trigger）

| trigger              | 说明                                        |
|----------------------|---------------------------------------------|
| `interval:<秒>`      | 每隔 N 秒自动执行，如 `interval:30`          |
| `manual`             | 仅手动触发（通过 `probe.run` 工具）          |

---

## 3. 结果回传（data_back）

| data_back | 行为                                        |
|-----------|---------------------------------------------|
| `wm`      | 探针结果以 WM 条目写入工作记忆，下一轮可见   |
| `none`    | 仅记录日志，不主动回传                       |

> `manual` 探针的结果直接通过工具返回值返回给 LLM，不受 `data_back` 影响。

---

## 4. 告警机制（alert_expr）

可为探针设置 Python 布尔表达式作为告警条件，变量 `output` 为结果字符串。

```
alert_expr:    "float(output.strip()) > 85.0"
alert_message: "⚠ CPU 温度过高：{output}°C，需要检查进程"
```

告警触发时，`alert_message` 被写入 WM，LLM 在下一轮感知到告警。

---

## 5. 探针目的字段（purpose）

**`purpose` 是探针系统最重要的字段。**

安装探针时，LLM 必须填写 `purpose`，说明：
- 为什么要监控这个指标
- 看到读数后应该如何解读
- 读数异常时预期采取什么行动

这个字段会在 judgment context 的传感器面板中始终显示，让下一轮 LLM（即使是全新的上下文）
也能理解每个探针存在的意义，不会看到一堆读数却不知道该怎么反应。

**正确示例：**
```
purpose: "监控 Redis 内存使用率（MB），若超过 500MB 说明任务缓存堆积，需要重启任务调度器"
```

**错误示例（无信息量）：**
```
purpose: "监控 Redis"
```

此外，**盲点覆盖判断不再读取 `purpose/spec` 自由文本，而是只读取显式 `coverage_tags`。**

推荐的 coverage tags：
- `ops:channel_health`：关键外部通道/代理/API 网关健康
- `ops:api_quota`：API 配额、额度或速率限制
- `workspace:git_state`：git 变更与工作区状态

如果不声明 `coverage_tags`，探针仍可运行，但不会计入 blind spots 覆盖判断。

---

## 6. 可用工具

### `probe.install` — 安装探针

| 参数            | 必填 | 说明                                                    |
|-----------------|------|---------------------------------------------------------|
| `name`          | ✓    | 唯一名称（字母数字下划线横线）                           |
| `purpose`       | ✓    | 部署目的与结果处理预期（详见第 5 节）                    |
| `kind`          | ✓    | `shell` / `http` / `python`                            |
| `spec`          | ✓    | 执行内容（命令 / URL / Python 代码）                    |
| `trigger`       | ✓    | `interval:<秒>` 或 `manual`                            |
| `data_back`     | -    | `wm`（默认）或 `none`                                  |
| `coverage_tags` | -    | 显式覆盖标签列表；blind spots 只读取这里，不从 `purpose/spec` 猜测 |
| `alert_expr`    | -    | 告警条件 Python 表达式                                  |
| `alert_message` | -    | 告警提示文本，支持 `{output}` 占位符                    |

> **注意**：`interval` 探针安装后第一次执行会延迟一个完整间隔。
> 如需立即获取数据，安装后立即调用 `probe.run`。

### `probe.run` — 立即触发探针

立即执行指定探针并返回结果，无论探针是 `manual` 还是 `interval`。
结果直接通过工具返回值返回，同时更新 `last_result`。

```json
{ "name": "disk_usage" }
```

### `probe.list` — 列出所有探针

返回当前部署的所有探针状态，包括名称、目的、最近读数、是否启用等。

### `probe.disable` — 暂停探针

停止探针的定时调度，但保留配置（purpose / spec 等）。
可用 `probe.enable` 恢复，无需重新安装。

```json
{ "name": "disk_usage" }
```

### `probe.enable` — 恢复探针

重新启动一个已暂停的探针，立即恢复调度。

```json
{ "name": "disk_usage" }
```

### `probe.remove` — 拆除探针

停止并永久删除指定探针，释放感知资源。

```json
{ "name": "disk_usage" }
```

---

## 7. 感知面板（judgment context 中的显示）

每轮 judgment 的上下文中包含"传感器网络"面板，格式如下：

```
  ✓ [disk_usage] shell/interval:60 →wm
  └ 目的: 监控根分区磁盘使用率，若超过 85% 需要清理日志文件
  └ @14:32 → /dev/sda1 78% 120G/160G

  ✓ [http_health] http/interval:30 →none 🔔
  └ 目的: 检查 API 服务健康端点，告警时说明服务异常需要重启
  └ @14:33 → {"status":"ok","latency":12}
```

LLM 通过这个面板了解当前感知网络状态，结合 `purpose` 判断是否需要响应。

---

## 8. 存储机制

探针配置持久化在 `{workspace_dir}/probes.json`（JSON 文件），
与灵舟主 SQLite 数据库**完全解耦**。

优点：
- 探针配置可直接用文本编辑器查看和修改
- 主 DB schema 变更不影响探针
- 重启后探针状态（含最近读数）完整恢复

`probes.json` 结构示例：
```json
{
  "next_id": 3,
  "probes": {
    "disk_usage": {
      "id": 1,
      "name": "disk_usage",
      "kind": "shell",
      "spec": "df -h / | tail -1 | awk '{print $5}'",
      "trigger": "interval:60",
      "purpose": "监控根分区磁盘使用率，若超过 85% 需要清理日志",
      "data_back": "wm",
      "coverage_tags": ["workspace:git_state"],
      "alert_expr": "int(output.strip().rstrip('%')) > 85",
      "alert_message": "⚠ 磁盘使用率过高：{output}",
      "enabled": true,
      "created_at": "2025-01-01T12:00:00",
      "last_run_at": "2025-01-01T14:32:10",
      "last_result": "78%",
      "last_error": null
    }
  }
}
```

---

## 9. 典型使用场景

### 监控系统资源
```
kind:    shell
spec:    "sensors | grep 'Core 0' | awk '{print $3}'"
trigger: interval:30
purpose: 监控 CPU 温度，超过 85°C 说明有高负载进程，需要排查并终止
alert_expr: float(output.replace('+','').replace('°C','').strip()) > 85
```

### 检查 HTTP 服务健康
```
kind:    http
spec:    http://localhost:8080/health
trigger: interval:60
purpose: 检查本地 API 服务是否在线，503/超时说明需要重启服务
```

### 监控文件变化
```
kind:    shell
spec:    "wc -l /var/log/app/error.log"
trigger: interval:120
purpose: 监控错误日志增长速度，若行数快速增加说明有持续异常需要诊断
```

### 自定义 Python 计算
```
kind:    python
spec:    |
  import psutil
  print(f"{psutil.virtual_memory().percent:.1f}")
trigger: interval:30
purpose: 监控内存使用率，超过 90% 说明内存泄漏或任务积压，需要检查并重启
alert_expr: float(output.strip()) > 90
```
