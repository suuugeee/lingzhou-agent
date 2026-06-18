"""memory/working.py — 工作记忆（WorkingMemory）。

设计：内存优先队列，有容量上限，按 priority 降序排列。
     超过容量时自动驱逐最低优先级条目。
     不持久化——WM 本来就是瞬态的，重启后从情节/语义记忆重建。
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


TASK_SWITCH_PRESERVE_KINDS: frozenset[str] = frozenset({
    "bootstrap_identity",
    "self_awareness",
    "task_anchor",
    "task_reflection",
    "task_result",
    "task_replan",
    "routing_guard",
    "progress_crystal",
})


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    ascii_chars = sum(1 for char in text if ord(char) < 128 and not char.isspace())
    other = sum(1 for char in text if ord(char) >= 128 and not ("\u4e00" <= char <= "\u9fff"))
    return cjk + max(1, ascii_chars // 4) + max(1, other // 2)


def _wm_keywords(text: str) -> frozenset[str]:
    """从文本提取关键词集合（CJK 2-4字 n-gram + ASCII 词元）。

    不引入外部依赖，专为 salience_gate 设计。
    """
    result: set[str] = set()
    # CJK n-gram（2-4 字）
    cjk_chars = [c for c in text if "\u4e00" <= c <= "\u9fff"]
    for n in (2, 3, 4):
        for i in range(len(cjk_chars) - n + 1):
            result.add("".join(cjk_chars[i:i + n]))
    # ASCII 词元（长度 >= 3，过滤停用词）
    _ASCII_STOP = {"the", "and", "is", "in", "it", "of", "to", "a", "an", "are", "for",
                   "on", "at", "by", "or", "as", "be", "was", "has", "not", "with"}
    import re as _re
    for word in _re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text):
        w = word.lower()
        if len(w) >= 3 and w not in _ASCII_STOP:
            result.add(w)
    return frozenset(result)


def _has_wm_overlap(content: str, keywords: frozenset[str]) -> bool:
    """判断 WM 条目内容与关键词集合是否有实质重叠（至少 1 个关键词命中）。"""
    if not keywords or not content:
        return False
    # keywords 中的 ASCII 词已被 _wm_keywords 做过 lower()，
    # content 也 lower() 再比较，保证大小写不敏感（CJK 不受影响）。
    content_lower = content.lower()
    return any(kw in content_lower for kw in keywords)


@dataclass(order=True)
class WMItem:
    # heapq 是最小堆；_sort_key 存正优先级，heappop 弹出最小值 = 最低优先级 = 正确驱逐方向
    _sort_key: float = field(init=False, repr=False)
    kind: str = field(compare=False)
    content: str = field(compare=False)
    priority: float = field(compare=False, default=0.8)
    created_at: datetime = field(compare=False, default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        self._sort_key = self.priority

    @property
    def estimated_tokens(self) -> int:
        """粗估 token 数；口径与 judgment context 预算保持一致。"""
        return _estimate_tokens(self.content)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "content": self.content,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
        }


class WorkingMemory:
    """容量有界的工作记忆。线程/协程安全（asyncio 单线程模型下天然安全）。

    双重上限：
      - capacity：条目数上限（防止无限碎片化写入）
      - token_budget：token 估算上限（压力指标，超出时驱逐低优条目）
    pressure 属性基于 token 估算，比条目数更准确反映对 LLM 上下文的实际占用。
    """

    def __init__(self, capacity: int = 20, token_budget: int = 0, item_max_tokens: int = 0) -> None:
        self._capacity = capacity
        # token_budget=0 表示禁用 token 压力（回退到条目数压力）
        self._token_budget = token_budget
        # item_max_tokens=0 表示不限制单条 content 大小
        self._item_max_tokens = item_max_tokens
        self._items: list[WMItem] = []
        self._multi_item_kinds = {"meta_reflection"}

    def _replace_items(self, items: list[WMItem]) -> None:
        self._items = items
        heapq.heapify(self._items)

    @property
    def total_tokens(self) -> int:
        """当前 WM 所有条目的估算 token 总数。"""
        return sum(item.estimated_tokens for item in self._items)

    @property
    def pressure(self) -> float:
        """当前压力 [0, 1]。优先用 token 估算；未配置 token_budget 时回退到条目数比例。"""
        if self._token_budget > 0:
            return min(1.0, self.total_tokens / self._token_budget)
        return len(self._items) / self._capacity if self._capacity > 0 else 0.0

    def add(self, item: WMItem) -> None:
        """添加条目，若超条目上限或超 token 预算则驱逐优先级最低的。
        若 item.kind 非空，先移除同 kind 旧条目（防御性去重，避免同 kind 条目累积）。
        """
        if item.kind and item.kind not in self._multi_item_kinds:
            self._replace_items([i for i in self._items if i.kind != item.kind])
        heapq.heappush(self._items, item)
        # 先按条目数收敛
        while len(self._items) > self._capacity:
            heapq.heappop(self._items)
        # 再按 token 预算收敛（至少保留 1 条）
        if self._token_budget > 0:
            while len(self._items) > 1 and self.total_tokens > self._token_budget:
                heapq.heappop(self._items)

    def get_top(self, n: int | None = None) -> list[dict[str, Any]]:
        """按优先级降序返回前 n 条（不修改内部状态）。"""
        sorted_items = sorted(self._items, key=lambda x: x.priority, reverse=True)
        if n is not None:
            sorted_items = sorted_items[:n]
        return [item.to_dict() for item in sorted_items]

    def clear(self, preserve_kinds: set[str] | None = None, kinds: set[str] | None = None) -> None:
        """清空工作记忆。preserve_kinds 中列出的类型条目保留（如身份锚点 bootstrap_identity）。
        kinds 中列出的类型条目清除。若两者都指定，优先 preserve_kinds。"""
        if preserve_kinds:
            self._replace_items([item for item in self._items if item.kind in preserve_kinds])
        elif kinds:
            self._replace_items([item for item in self._items if item.kind not in kinds])
        else:
            self._items.clear()

    def salience_gate(
        self,
        user_message: str = "",
        *,
        preserve_kinds: set[str],
        priority_floor: float = 0.7,
        keyword_boost: float = 0.15,
    ) -> int:
        """显著性门控（全局工作空间理论 graded competition 轻量实现）。

        用户消息到达时调用：与消息内容有关键词重叠的条目被 boost，
        低优且无相关性的条目被丢弃，preserve_kinds 中的条目无条件保留。

        返回被丢弃的条目数（供日志使用）。
        """
        keywords = _wm_keywords(user_message) if user_message.strip() else frozenset()
        kept: list[WMItem] = []
        dropped = 0
        boosted_any = False
        for item in self._items:
            if item.kind in preserve_kinds:
                kept.append(item)
                continue
            if item.priority >= priority_floor:
                kept.append(item)
                continue
            if keywords and _has_wm_overlap(item.content, keywords):
                # 相关性 boost：priority + keyword_boost，上限 1.0
                boosted = WMItem(
                    kind=item.kind,
                    content=item.content,
                    priority=min(1.0, item.priority + keyword_boost),
                    created_at=item.created_at,
                )
                kept.append(boosted)
                boosted_any = True
                continue
            dropped += 1
        if dropped or boosted_any:
            self._replace_items(kept)
        return dropped

    def __len__(self) -> int:
        return len(self._items)
