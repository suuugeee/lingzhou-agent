"""core.config.budget - model-window based context budgeting helpers."""
from __future__ import annotations


def context_window_input_hard_budget(context_window: int) -> int:
    """模型窗口扣除输出预留后的输入硬上限。"""
    window = max(1, int(context_window))
    return max(1, window - max(1024, window // 4))


def adaptive_judgment_input_budget(context_window: int) -> int:
    """根据模型上下文窗口推导 judgment 工作集预算。

    context_window 是硬上限，不是每轮都应塞满的目标。小窗口模型尽量用满可用输入；
    大窗口模型保留为 32K/64K 级工作集，长证据通过存储、检索和摘要进入上下文。
    """
    window = max(1, int(context_window))
    hard_budget = context_window_input_hard_budget(window)
    if window <= 32_768:
        return hard_budget
    if window <= 262_144:
        return min(hard_budget, 32_768)
    return min(hard_budget, 65_536)
