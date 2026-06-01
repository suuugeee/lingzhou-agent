# ADR 0015: LLM 感知完整性（禁止机械截断改写）

- Status: Accepted
- Date: 2026-06-01

## Context

Lingzhou 依赖模型对 WM、工具结果、对话连续性与判断信号的**整体感知**来推理。若在模型可见路径上做头尾截断、中段省略号拼接或正则改写 `reply_to_user`，会破坏「能感知、能思考」的前提。

## Decision

### 允许（机制层，不篡改单条语义）

| 手段 | 说明 |
|------|------|
| **整段/整节省略** | `apply_context_budget` 按优先级**清空整个 context section**，不从中段切断句子 |
| **整轮对话省略** | `_fmt_chat_history` 仅**丢弃最旧完整轮次**（整行），不在单条消息内截断 |
| **整消息省略** | prompt 超窗时 `_trim_messages_for_prompt_limit` **用固定叙事 stub 替换整条 message**（含超大 system/最后 user），不切片正文 |
| **输出 overflow** | 不重试压缩 prompt；换模型/降 thinking/提高 max_tokens（已有行为） |
| **日志摘要** | `_clip_reply_for_log`、`log_summary` 仅用于日志/telemetry，不进模型 prompt |
| **边界归一化** | `normalize_action_shape` 等只修正 decision/工具 id 形态，**不改写** `reply_to_user` 措辞 |

### 禁止（默认不得进入模型可见内容）

- 对 assistant/user/tool/system **正文**做 head/tail 字符切片、`[...省略...]` / `[prompt 已压缩]` 拼接
- 用正则「润色」或替换模型已生成的 `reply_to_user`、记忆叙述
- 为省 token 在单条 WM/情节文本中段截断（应整节降级或整节不注入）

### 实现锚点

- `core/judgment/decision/helpers.py` — `_trim_messages_for_prompt_limit_impl`（整消息 stub）
- `core/judgment/context/budget.py` — section 级清空
- `core/judgment/context/chat_sections.py` — 按轮次丢弃旧历史
- `core/judgment/boundary/normalize.py` — 文档声明不改写回复正文
- `context/tokens._compress_*` — **不**用于 judgment 组装或 LLM messages；仅保留给单元测试/遗留工具

## Consequences

- 超窗时可能省略更多**整条**旧消息，模型通过 stub 文案知晓「证据在 WM/工具历史」
- 新增容量策略须先对照本 ADR；若必须压缩，应走**二次 LLM 摘要**（显式调用）而非静默切片

## Validation

- `tests/test_judgment_layers.py::test_trim_messages_omits_whole_messages_not_slices`
- `tests/test_judgment_ctx.py::test_chat_with_retry_output_overflow_skips_prompt_compression`
- 路线图「优化原则」与本 ADR 交叉引用
