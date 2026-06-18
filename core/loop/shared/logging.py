"""core/loop/shared/logging.py - loop 的日志与用户可见文本 helper。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core.contracts.execution import action_key_param

if TYPE_CHECKING:
    from core.judgment import JudgmentOutput
    from store.task import Task
    from tools.registry import ToolResult

DEFAULT_LOG_REPLY_CHARS = 240


def _strip_memory_context(text: str) -> str:
    """剥离 LLM 输出中意外泄露的 <memory-context>...</memory-context> 内容。"""
    import re as _re

    cleaned = _re.sub(r"<memory-context>.*?</memory-context>", "", text, flags=_re.DOTALL)
    return cleaned.strip() or text.strip()


class MemoryContextScrubber:
    """流式版 memory-context 剥离器，用于逐 chunk 消费 LLM 流式输出。

    用法::

        scrubber = MemoryContextScrubber()
        for chunk in stream:
            safe = scrubber.feed(chunk)
            if safe:
                print(safe, end="", flush=True)
        print(scrubber.flush(), end="", flush=True)

    设计要点：
    - 标签可能跨多个 chunk（边界缓冲 16 字节），不依赖一次性正则；
    - `flush()` 返回剩余缓冲中不含 memory-context 的内容并重置状态；
    - 线程不安全，每个流实例单独创建。
    """

    _OPEN_TAG = "<memory-context>"
    _CLOSE_TAG = "</memory-context>"
    # 开始标签最长为 17 字节，保留略大于此长度的边界缓冲
    _BOUNDARY = len(_OPEN_TAG) + 1

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False

    def feed(self, chunk: str) -> str:
        """消费一个 chunk，返回可立即输出的安全内容（可能为空字符串）。"""
        self._buf += chunk
        out_parts: list[str] = []
        while True:
            if self._inside:
                end = self._buf.find(self._CLOSE_TAG)
                if end == -1:
                    break  # 尚未遇到结束标签，继续缓冲
                self._buf = self._buf[end + len(self._CLOSE_TAG):]
                self._inside = False
            else:
                start = self._buf.find(self._OPEN_TAG)
                if start == -1:
                    # 保留边界缓冲，防止 '<memory-context>' 跨 chunk 被截断
                    safe_len = max(0, len(self._buf) - self._BOUNDARY)
                    out_parts.append(self._buf[:safe_len])
                    self._buf = self._buf[safe_len:]
                    break
                out_parts.append(self._buf[:start])
                self._buf = self._buf[start + len(self._OPEN_TAG):]
                self._inside = True
        return "".join(out_parts)

    def flush(self) -> str:
        """流结束时调用：输出剩余缓冲中不含 memory-context 的内容，并重置状态。"""
        if self._inside:
            # 整段 <memory-context> 未闭合，全部丢弃
            self._buf = ""
            self._inside = False
            return ""
        out = self._buf
        self._buf = ""
        return out


def _clip_reply_for_log(text: str, limit: int = DEFAULT_LOG_REPLY_CHARS) -> str:
    cleaned = _strip_memory_context(text).replace("\n", "\\n").strip()
    limit = max(1, int(limit or 1))
    if len(cleaned) <= limit:
        return cleaned
    if limit <= 3:
        return cleaned[:limit]
    return cleaned[: limit - 3].rstrip() + "..."


def _clip_signal_text(text: str, limit: int = 160) -> str:
    cleaned = " ".join((text or "").split())
    limit = max(1, int(limit or 1))
    if len(cleaned) <= limit:
        return cleaned
    if limit <= 3:
        return cleaned[:limit]
    return cleaned[: limit - 3].rstrip() + "..."


_INTERNAL_REPLY_BASIS_MARKERS = (
    "Action-first",
    "fallback",
    "行为门控",
    "行为门控制动",
    "通用问题解决守卫",
    "缺少 chosen_action_id",
    "LLM 输出解析失败",
    "无效 decision",
    "未知工具",
    "list index out of range",
    "not defined",
)


def _safe_reply_basis(text: str, default: str) -> str:
    basis = str(text or "").strip()
    if any(marker in basis for marker in _INTERNAL_REPLY_BASIS_MARKERS):
        basis = ""
    return _clip_signal_text(basis or default, 120)


def _summarize_state_delta(state_delta: dict[str, Any] | None, limit: int = 120) -> str:
    if not state_delta:
        return ""
    parts = [f"{key}={state_delta[key]}" for key in sorted(state_delta)]
    return _clip_signal_text("; ".join(parts), limit)


def _format_action_feedback_line(
    action: JudgmentOutput,
    result: ToolResult,
    *,
    progressful: bool,
) -> str:
    tool = action.action_label() or action.decision or "-"
    key = action_key_param(action.params) if action.decision == "act" else ""
    status = "error" if result.error else ("skipped" if result.skipped else ("ok" if action.decision == "act" else action.decision))
    parts = [f"tool={tool}"]
    if key:
        parts.append(f"key={key}")
    parts.append(f"status={status}")
    parts.append(f"progressful={progressful}")
    if result.error:
        parts.append(f"error={_clip_signal_text(result.error, 80)}")
    if result.state_delta:
        parts.append(f"state_delta={_summarize_state_delta(result.state_delta, 90)}")
    log_summary = ""
    if isinstance(result.metadata, dict):
        log_summary = str(result.metadata.get("log_summary") or "").strip()
    summary_text = log_summary or (result.summary or "")
    if summary_text:
        parts.append(f"summary={_clip_signal_text(summary_text, 100)}")
    return " | ".join(parts)


def _fallback_reply_for_user(action: JudgmentOutput, result: ToolResult, active_task: Task | None) -> str:
    state_delta = result.state_delta if isinstance(result.state_delta, dict) else {}
    recovery_next_step = str(
        state_delta.get("recovery_next_step")
        or state_delta.get("next_verification")
        or ""
    ).strip()
    next_step = str(action.next_step or recovery_next_step or (active_task.next_step if active_task else "") or "").strip()
    next_clause = f"下一步我会先处理：{_clip_signal_text(next_step, 80)}" if next_step else ""
    if result.error:
        detail = _clip_signal_text(result.summary or result.error, 120)
        return " ".join(part for part in [f"这轮工具执行失败（状态: error）：{detail}。", next_clause] if part).strip()

    if action.decision in {"wait", "pause"}:
        basis = _safe_reply_basis(action.rationale or "", "需要更多信息后再继续。")
        return " ".join(part for part in [f"我需要先停一下：{basis}", next_clause] if part).strip()

    task_status = str((result.state_delta or {}).get("task_status") or "").strip()
    if task_status == "waiting":
        wait_kind = str((result.state_delta or {}).get("wait_kind") or "external").strip()
        wait_key = str((result.state_delta or {}).get("wait_key") or "").strip()
        wait_desc = wait_kind + (f"/{wait_key}" if wait_key else "")
        return " ".join(part for part in [f"当前任务已转入等待：{wait_desc}。", next_clause] if part).strip()

    basis = _safe_reply_basis(action.rationale or "", "已完成本轮处理，正在整理基于证据的答复。")
    return " ".join(part for part in [f"我已完成本轮处理：{basis}", next_clause] if part).strip()
