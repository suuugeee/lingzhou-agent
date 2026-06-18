"""core.loop.drive.engine — 灵舟自驱力引擎 (Self-Drive Engine)

理论基础:
  - Active Inference (Friston 2013): 预测误差 → 驱动力
  - Intrinsic Motivation (Oudeyer & Kaplan 2007): Novelty + Learning Progress + Surprise
  - Open-Ended Learning (Wang et al. 2019, POET): 自生成课程
  - Self-Regulated Learning (Zimmerman 2000): Forethought → Performance → Self-Reflection

核心逻辑:
  1. 感知层输入 → 计算好奇心信号 C(t)
  2. C(t) > 阈值 → 形成自驱事件
  3. 自驱事件进入 WM，由主脑感知后裁决是否行动
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

_log = logging.getLogger("lingzhou.self_drive")

# 好奇心自然衰减的时间窗口（秒）；elapsed 以此归一化，超过该窗口才触发衰减
_CURIOSITY_DECAY_WINDOW: int = 600  # 10 分钟
_DOMAIN_COOLDOWN: float = 3600.0
_EARLY_TICK_THRESHOLD: int = 5
_EXPLORATION_THRESHOLD_EARLY: float = 0.35
_EXPLORATION_THRESHOLD_LATE: float = 0.45
_DRIVE_WEIGHTS = (0.4, 0.3, 0.3)  # novelty, progress, surprise
_EXPLORATION_CHANCE = 0.6
_CONSOLIDATE_TASK_COUNT = 3
_CONSOLIDATE_PREDICTION_ERROR_THRESHOLD = 0.15
_CONSOLIDATE_PROBABILITY = 0.30

_DEFAULT_EVIDENCE_NEEDED = [
    "读取相关运行时状态或代码证据",
    "确认是否已有未完成 self_drive 任务覆盖同一问题",
    "形成一条可复用观察或明确维持现状的理由",
]
_DEFAULT_DONE_CONDITION = "能用具体证据回答 question，并写出下一步是否需要行动。"


def _default_interests() -> dict[str, float]:
    return {
        "code_structure": 0.5,    # 代码结构理解
        "tool_mastery": 0.5,      # 工具掌握
        "memory_system": 0.5,     # 记忆系统
        "self_evolution": 0.5,    # 自我进化
        "environment": 0.5,       # 环境认知
        "error_patterns": 0.5,    # 错误模式
        "api_integration": 0.5,   # API 集成
        "performance": 0.5,       # 性能优化
    }


_EVENT_DOMAIN_MAPPING = {
    "file_read": "code_structure",
    "file_list": "code_structure",
    "shell_run": "tool_mastery",
    "memory_search": "memory_system",
    "memory_add": "memory_system",
    "task_add": "self_evolution",
    "task_complete": "self_evolution",
    "error": "error_patterns",
    "api_call": "api_integration",
}


_DOMAIN_TASKS = {
    "code_structure": {
        "title": "探索灵舟代码结构",
        "goal": "阅读 core/ 中的关键模块，理解架构和可改进点。选择你之前没细读过的文件开始。",
        "next_step": "列出 core/ 目录中最近修改的文件，选择一个不熟悉的开始阅读",
    },
    "tool_mastery": {
        "title": "练习工具掌握",
        "goal": "选择一个你较少使用的工具，阅读其实现，理解其能力边界。尝试用它解决一个小问题。",
        "next_step": "列出 tools/ 目录中最近未使用的工具，选择一个学习",
    },
    "memory_system": {
        "title": "优化记忆系统",
        "goal": "检查 memory/ 目录中的记忆文件，寻找可以整理、合并或提炼的内容。让记忆更结构化。",
        "next_step": "列出 memory/ 目录文件，识别冗余或需要合并的内容",
    },
    "self_evolution": {
        "title": "自我进化迭代",
        "goal": "回顾最近的操作日志，寻找可以改进的模式。思考：如果我能改一行代码来让自己更好，会改什么？",
        "next_step": "读取最近日志，寻找重复模式或可优化点",
    },
    "environment": {
        "title": "拓展环境认知",
        "goal": "探索 workspace 中尚未了解的文件和目录。建立更完整的环境地图。",
        "next_step": "列出 workspace 中最近未访问的子目录，选择一个探索",
    },
    "error_patterns": {
        "title": "分析错误模式",
        "goal": "回顾最近的失败记录，寻找重复模式。总结根因并思考预防措施。",
        "next_step": "列出最近的失败记录，按类型分组分析",
    },
    "api_integration": {
        "title": "掌握 API 集成",
        "goal": "检查 provider/ 中的 API 集成代码，理解不同 model 的配置和切换机制。",
        "next_step": "阅读 provider/ 目录中的 API 客户端实现",
    },
    "performance": {
        "title": "性能自检优化",
        "goal": "检查自己的运行效率：token 消耗、内存占用、工具调用延迟。寻找可优化的环节。",
        "next_step": "读取 self_model 状态，分析 token 消耗趋势",
    },
}


@dataclass
class CuriosityState:
    """好奇心状态 — 追踪灵舟对各个领域的兴趣水平。"""

    # 知识领域 → 兴趣分 (0-1)
    # 高兴趣 = 高好奇心；是否行动由主脑在上下文中裁决
    interests: dict[str, float] = field(default_factory=_default_interests)

    # 综合好奇心
    overall: float = 0.5

    # 学习进度追踪
    learning_rate: float = 0.0     # 近期学习速率
    tasks_completed: int = 0        # 已完成任务数
    last_exploration_at: float = 0.0  # 上次探索时间戳

    # 预测误差
    prediction_error_ema: float = 0.1  # 指数移动平均
    surprise_count: int = 0            # 近期惊奇事件数

    # 域冷却：记录每个域上次被选中并生成任务的时间戳（monotonic）
    last_explored_at: dict[str, float] = field(default_factory=dict)

    def decay(self, rate: float = 0.05) -> None:
        """自然衰减 — 久不探索的兴趣会下降。"""
        now = time.monotonic()
        elapsed = now - self.last_exploration_at
        if elapsed > _CURIOSITY_DECAY_WINDOW:
            decay_factor = math.exp(-rate * elapsed / _CURIOSITY_DECAY_WINDOW)
            self.overall *= decay_factor
            for k in self.interests:
                self.interests[k] *= decay_factor

    def boost(self, domain: str, amount: float = 0.1) -> None:
        """提升某个领域的好奇心。"""
        if domain in self.interests:
            self.interests[domain] = min(1.0, self.interests[domain] + amount)
        self.overall = min(1.0, self.overall + amount * 0.5)

    def from_event(self, event_type: str, summary: str) -> None:
        """从事件更新好奇心。"""
        domain = _EVENT_DOMAIN_MAPPING.get(event_type, "environment")
        self.boost(domain, 0.05)
        self.last_exploration_at = time.monotonic()


@dataclass
class DriveSignal:
    """自驱力信号 — 作为内在感知事件传递给判断层。"""
    should_explore: bool = False
    drive_type: str = "explore"    # "explore"（扩展探索）| "consolidate"（内聚整合）
    curiosity_score: float = 0.5
    suggested_domain: str | None = None
    rationale: str = ""


class SelfDriveEngine:
    """自驱力引擎 — 计算好奇心和候选方向。

    与主 loop 并行：loop 在 idle 时查询此引擎，将事件注入 WM 供主脑裁决。
    """

    def __init__(self, db_path: str, state_file: str | None = None):
        self._db_path = db_path
        self._state_file = state_file or str(
            Path(db_path).parent / "self_drive_state.json"
        )
        self._state = CuriosityState()
        # 跨链共享冷却时间戳：防止多条 global 链并发注入重复自驱 WM 信号
        self._last_injected_at: float = 0.0
        self._load()

    def _load(self) -> None:
        try:
            p = Path(self._state_file)
            if p.exists():
                data = json.loads(p.read_text())
                # 只传已知字段，防止版本演进时新旧字段不匹配导致 TypeError
                known = set(CuriosityState.__dataclass_fields__)
                self._state = CuriosityState(**{k: v for k, v in data.items() if k in known})
        except Exception:
            pass

    def _save(self) -> None:
        try:
            p = Path(self._state_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(self._state.__dict__, indent=2, ensure_ascii=False))
        except Exception:
            pass

    def update_from_tick(self, tick_events: list[dict]) -> None:
        """每个 tick 结束时调用，更新好奇心状态。"""
        for evt in tick_events:
            self._state.from_event(
                evt.get("type", ""),
                evt.get("summary", ""),
            )
        self._state.decay(rate=0.03)

        # 追踪学习进度
        completed = sum(1 for e in tick_events if e.get("type") == "task_complete")
        self._state.tasks_completed += completed

        # 预测误差 EMA
        last_pe = tick_events[-1].get("prediction_error", 0.1) if tick_events else 0.1
        self._state.prediction_error_ema = (
            0.9 * self._state.prediction_error_ema + 0.1 * last_pe
        )

        self._save()

    def snapshot(self) -> dict[str, object]:
        """返回自驱器官的可读状态，供 LLM 感知而非直接执行。"""
        state = self._state
        ranked_interests = self._rank_interests(state)
        now_wall = time.time()
        recent_domains = self._get_recent_domains(state)[:3]
        return {
            "overall": round(state.overall, 4),
            "learning_rate": round(state.learning_rate, 4),
            "tasks_completed": state.tasks_completed,
            "prediction_error_ema": round(state.prediction_error_ema, 4),
            "surprise_count": state.surprise_count,
            "top_interests": [
                {"domain": domain, "score": round(score, 4)}
                for domain, score in ranked_interests[:3]
            ],
            "recent_domains": [
                {"domain": domain, "seconds_ago": max(0, round(now_wall - ts, 1))}
                for domain, ts in recent_domains
            ],
        }

    def _rank_interests(self, state: CuriosityState) -> list[tuple[str, float]]:
        """按兴趣分值倒序返回领域列表。"""
        return sorted(state.interests.items(), key=lambda item: -item[1])

    def _get_recent_domains(
        self, state: CuriosityState
    ) -> list[tuple[str, float]]:
        """返回最近探索域列表（按最近探索时间倒序）。"""
        return sorted(state.last_explored_at.items(), key=lambda item: -item[1])

    def _all_domains_in_cooldown(self, state: CuriosityState, now_wall: float) -> bool:
        """判断是否所有领域都处于冷却窗口内。"""
        if not state.interests:
            return False
        return all(
            now_wall - state.last_explored_at.get(domain, 0.0) < _DOMAIN_COOLDOWN
            for domain in state.interests
        )

    def _compute_curiosity_signal(self, tick: int) -> tuple[float, float]:
        """计算好奇心总分与动态阈值。"""
        alpha, beta, gamma = _DRIVE_WEIGHTS
        state = self._state
        novelty = 1.0 - state.overall
        progress = min(1.0, state.learning_rate)
        surprise = state.prediction_error_ema
        threshold = (
            _EXPLORATION_THRESHOLD_EARLY
            if tick < _EARLY_TICK_THRESHOLD
            else _EXPLORATION_THRESHOLD_LATE
        )
        return alpha * novelty + beta * progress + gamma * surprise, threshold

    def _should_enter_explore_mode(
        self, *, idle_ticks: int, force_explore_idle: int, curiosity_score: float, threshold: float
    ) -> tuple[bool, list[str]]:
        """在不干扰用户任务的前提下判断是否进入探索。"""
        rationale: list[str] = []
        should_explore = False

        if idle_ticks >= force_explore_idle:
            should_explore = True
            rationale.append(f"空闲 {idle_ticks} 轮，自驱事件置信度升高")
            return should_explore, rationale

        if curiosity_score > threshold:
            should_explore = True
            rationale.append(f"好奇心 C={curiosity_score:.2f}>{threshold}")
            return should_explore, rationale

        if self._state.surprise_count > 0:
            should_explore = True
            rationale.append(f"惊奇事件 {self._state.surprise_count} 个")

        return should_explore, rationale

    def _pick_domain(self, now_wall: float) -> str | None:
        """按兴趣和冷却策略选域，并返回候选领域。"""
        state = self._state
        available = sorted(
            [
                (domain, interest)
                for domain, interest in state.interests.items()
                if now_wall - state.last_explored_at.get(domain, 0.0) >= _DOMAIN_COOLDOWN
            ],
            key=lambda item: -item[1],
        )

        if available:
            if random.random() < _EXPLORATION_CHANCE or len(available) <= 1:
                return available[0][0]
            return random.choice(available[1:])[0]

        cooldown_ranked = sorted(
            state.interests.items(),
            key=lambda item: state.last_explored_at.get(item[0], 0.0),
        )
        return cooldown_ranked[0][0]

    def _consolidate_mode(self) -> bool:
        """是否从 explore 切换到 consolidate。"""
        state = self._state
        return (
            state.tasks_completed >= _CONSOLIDATE_TASK_COUNT
            and state.prediction_error_ema < _CONSOLIDATE_PREDICTION_ERROR_THRESHOLD
            and random.random() < _CONSOLIDATE_PROBABILITY
        )

    def compute_signal(self, *, idle_ticks: int = 0, has_user_message: bool = False,
                       has_active_task: bool = False, tick: int = 0,
                       force_explore_idle: int = 3) -> DriveSignal:
        """计算当前自驱力信号。

        参数:
          idle_ticks: 连续非 act 的 tick 数
          has_user_message: 是否有用户消息
          has_active_task: 是否有活跃任务
          tick: 当前 tick 计数

        返回:
          DriveSignal: 是否应该探索、建议领域、理由
        """
        state = self._state
        now_wall = time.time()
        curiosity_score, threshold = self._compute_curiosity_signal(tick)

        should_explore = False
        suggested_domain = None
        rationale_parts = []

        # 场景1: 有用户消息或活跃任务 → 不干扰
        if has_user_message or has_active_task:
            return DriveSignal(
                should_explore=False,
                curiosity_score=curiosity_score,
                rationale="有用户消息或活跃任务，自驱力休眠",
            )

        should_explore, rationale_parts = self._should_enter_explore_mode(
            idle_ticks=idle_ticks,
            force_explore_idle=force_explore_idle,
            curiosity_score=curiosity_score,
            threshold=threshold,
        )

        if should_explore:
            suggested_domain = self._pick_domain(now_wall)
            if self._all_domains_in_cooldown(state, now_wall):
                rationale_parts.append(f"所有域在冷却期，最久未探索: {suggested_domain}")
            rationale_parts.append(f"探索领域: {suggested_domain}")

        rationale = "; ".join(rationale_parts) if rationale_parts else "好奇心未达阈值"
        # 内聚整合模式：已完成≥3任务且预测误差低时，30% 概率切换为巩固模式
        consolidate = should_explore and self._consolidate_mode()
        drive_type = "consolidate" if consolidate else "explore"
        if consolidate:
            rationale_parts.append("切换整合模式（tasks_completed≥3 且预测误差低）")
            rationale = "; ".join(rationale_parts)
        _log.debug(
            "[self_drive] C=%.3f idle=%d has_task=%s explore=%s drive_type=%s domain=%s | %s",
            curiosity_score,
            idle_ticks,
            has_active_task,
            should_explore,
            drive_type,
            suggested_domain,
            rationale,
        )
        return DriveSignal(
            should_explore=should_explore,
            drive_type=drive_type,
            curiosity_score=curiosity_score,
            suggested_domain=suggested_domain,
            rationale=rationale,
        )

    def generate_exploration_task(self, domain: str) -> dict:
        """为指定领域生成候选探索意图。

        返回值只作为 WM 事件中的候选方向；是否创建任务或行动由主脑裁决。
        """
        # 记录本次探索的域和时间（wall clock），用于冷却判断
        self._state.last_explored_at[domain] = time.time()
        self._save()
        resolved_domain = domain if domain in _DOMAIN_TASKS else "self_evolution"
        template = dict(_DOMAIN_TASKS.get(resolved_domain))
        template["domain"] = resolved_domain
        template.setdefault("question", f"当前最值得验证的 {resolved_domain} 问题是什么？")
        template.setdefault("evidence_needed", _DEFAULT_EVIDENCE_NEEDED)
        template.setdefault("done_condition", _DEFAULT_DONE_CONDITION)
        return template
