"""core.subagent — 子灵（完整实现）。

子灵是灵舟派生的有界任务执行体，提供四层能力：
  Tier-0（只读原型）: 共享父灵所有记忆，工具访问受限
  Tier-1（完整子灵）: 独立 memory namespace（EpisodicMemory + SemanticMemory）
  Tier-2（Ethos 继承）: 从父灵 soul:ethos_baseline 派生初始价值观
  Tier-3（结果合并）: SubagentResult 携带待吸收的语义记忆节点

设计原则：
  - 轻量：不运行完整 OCC 情绪 / Ethos 推导，只做 judgment + execution 循环
  - 只读默认：不写入父灵 TaskStore / SemanticMemory（isolated_memory=True 时独立写）
  - 可观测：每 tick 关键结果写入 observations，随 SubagentResult 返回
  - 可吸收：关键语义记忆可随结果返回，供父灵 tools/subagent.absorb 合并
"""
from __future__ import annotations

import json
import logging
import contextlib
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from typing import cast as _cast

from core.immune.policy import (
    _DEFAULT_BLOCKED_TOOLS,
)
from core.immune.policy import (
    is_readonly_blocked_tool as _is_readonly_blocked_tool,
)

from .task_store_view import _SubagentTaskStoreView

_log = logging.getLogger("lingzhou.subagent")

if TYPE_CHECKING:
    from core.execution import ExecutionLayer
    from core.judgment import JudgmentLayer
    from tools.registry import ToolContext, ToolRegistry
    from tools.view_protocols import (
        EpisodicViewProtocol,
        SemanticViewProtocol,
        TaskStoreViewProtocol,
    )

_NON_ABSORBABLE_MEMORY_KINDS: frozenset[str] = frozenset({
    "run_result",
    "meta_reflection",
    "rule_revision",
})


def _resolve_subagent_wm_limits(cfg: Any) -> tuple[int, int, int]:
    """子灵 WM 初始化参数回退层。

    优先走完整配置对象；在测试中的轻量 mock（SimpleNamespace）场景下，
    自动回退到与内置默认兼容的基线配置，避免初始化因缺字段失败。
    """
    memory_cfg = getattr(cfg, "memory", None)

    working_capacity = int(getattr(memory_cfg, "working_capacity", 40))
    token_budget = 0
    if hasattr(cfg, "effective_wm_token_budget"):
        with contextlib.suppress(Exception):
            token_budget = int(cfg.effective_wm_token_budget())
    if token_budget <= 0:
        ratio = float(getattr(memory_cfg, "wm_token_budget_ratio", 0.2))
        if hasattr(cfg, "judgment_input_token_budget"):
            with contextlib.suppress(Exception):
                token_budget = max(256, int(cfg.judgment_input_token_budget() * ratio))
        else:
            token_budget = 8000
    item_max_tokens = int(getattr(memory_cfg, "wm_item_max_tokens", 0))
    return working_capacity, token_budget, item_max_tokens


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _default_ethos_cfg() -> Any:
    """返回子灵继承逻辑使用的默认 EthosConfig。"""
    from core.config_models import EthosConfig

    return EthosConfig()


def _build_subagent_active_task(sub_id: str, goal: str) -> Any:
    from store.task import Task

    title = (goal or "").strip()
    title = f"子灵任务: {title}" if title else f"子灵任务: {sub_id}"
    return Task(
        id=-1,
        title=title,
        status="in_progress",
        priority="normal",
        created_at=_utc_now_iso(),
        goal=goal,
        source="subagent",
        chain_id=f"subagent:{sub_id}",
        extras={"subagent_id": sub_id, "virtual": True},
    )


class _SubagentEpisodicView:
    """父灵情节记忆的只读视图，子灵运行日志不回写父灵。"""

    def __init__(self, parent: Any) -> None:
        self._parent = parent

    def record(self, *args: Any, **kwargs: Any) -> None:
        return None

    def record_event(self, *args: Any, **kwargs: Any) -> None:
        return None

    def load_for_context(self, task_id: str | None, n_recent: int = 20) -> str:
        return self._parent.load_for_context(task_id, n_recent)

    def load_for_chat_context(
        self,
        chat_id: str | None,
        n_recent: int = 20,
        *,
        max_chars: int | None = None,
    ) -> str:
        return self._parent.load_for_chat_context(chat_id, n_recent, max_chars=max_chars)

    def load_for_interlocutor_context(
        self,
        interlocutor_id: str | None,
        n_recent: int = 20,
        *,
        max_chars: int | None = None,
    ) -> str:
        return self._parent.load_for_interlocutor_context(interlocutor_id, n_recent, max_chars=max_chars)

    def load_for_task_narrative(self, task_id: str | None, n_recent: int = 20) -> str:
        return self._parent.load_for_task_narrative(task_id, n_recent)

    def load_recent_daily_context(self, days: int = 2, max_chars: int = 1200) -> str:
        return self._parent.load_recent_daily_context(days, max_chars)

    def search(self, query: str, max_chars: int = 2000, exclude_task_id: str | None = None) -> str:
        return self._parent.search(query, max_chars=max_chars, exclude_task_id=exclude_task_id)

    def get_recent_turns(
        self,
        task_id: str | None = None,
        limit: int = 3,
        *,
        chat_id: str | None = None,
        interlocutor_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._parent.get_recent_turns(task_id=task_id, limit=limit, chat_id=chat_id, interlocutor_id=interlocutor_id)

    def list_recent_narrative(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._parent.list_recent_narrative(limit=limit)


class _SubagentSemanticView:
    """父灵语义记忆的只读视图，子灵反思与 run 结晶不回写父灵。"""

    def __init__(self, parent: Any) -> None:
        self._parent = parent

    def upsert(self, node: Any) -> None:
        return None

    @property
    def decay_lambda(self) -> float:
        return float(getattr(self._parent, "decay_lambda", 0.0) or 0.0)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        kind: str | None = None,
        tag: str | None = None,
        source: str | None = None,
        task_id: str | int | None = None,
        path_prefix: str | None = None,
        id_prefix: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._parent.retrieve(
            query, top_k=top_k, kind=kind, tag=tag, source=source,
            task_id=task_id, path_prefix=path_prefix, id_prefix=id_prefix,
        )

    def retrieve_multi_anchor(
        self,
        anchors: list[str],
        top_k: int = 5,
        convergence_bonus: float = 0.15,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._parent.retrieve_multi_anchor(
            anchors, top_k=top_k, convergence_bonus=convergence_bonus, source=source,
        )

    def stats(self) -> dict[str, Any]:
        return self._parent.stats()

    def get(self, node_id: str) -> Any:
        return self._parent.get(node_id)


# ── 数据模型 ────────────────────────────────────────────────────────────────────

@dataclass
class SubagentConfig:
    """子灵执行配置。"""
    goal: str
    max_ticks: int = 8
    allowed_tools: list[str] | None = None    # None = 继承所有（减去黑名单）
    blocked_tools: list[str] | None = None    # 额外黑名单（追加）
    subagent_id: str = field(default_factory=lambda: f"sub-{uuid.uuid4().hex[:8]}")
    label: str = ""                           # 可选标签（竞争进化时标识候选版本）
    # Tier-1: 独立 memory namespace
    isolated_memory: bool = False             # True = 使用独立 EpisodicMemory + SemanticMemory
    # Tier-2: Ethos 继承
    inherit_ethos: bool = True                # True = 从父灵 soul:ethos_baseline 派生价值观


@dataclass
class SubagentProposal:
    """子灵执行结束后的候选提交（公理 A7：子灵只能提交候选，父灵决定吸收/拒绝）。

    父灵可选择：吸收 / 拒绝 / 延后 / 重试 / 交由另一子灵复核。
    """
    observations: list[str] = field(default_factory=list)             # tick 级观察（做了什么）
    action_results: list[str] = field(default_factory=list)           # 工具执行摘要
    memory_candidates: list[dict[str, Any]] = field(default_factory=list)  # 建议吸收的记忆节点
    skill_candidates: list[dict[str, Any]] = field(default_factory=list)   # 建议长出的技能（保留）
    change_candidates: list[dict[str, Any]] = field(default_factory=list)  # 建议的变更（保留）
    self_assessment: str = ""                                          # 子灵自评摘要
    exceptions: list[str] = field(default_factory=list)               # 异常与风险


@dataclass
class SubagentResult:
    """子灵执行结果，注入到父灵 WM；可选携带待吸收的语义记忆节点。"""
    subagent_id: str
    goal: str
    ticks_run: int
    completed: bool          # LLM 判断 "wait"（无任务）= 认为完成
    error: str | None
    last_summary: str        # 最后一次工具执行摘要
    observations: list[str]  # 每 tick 的关键观察
    label: str = ""
    # Tier-3: 结果合并 — 子灵记录的语义记忆节点，供父灵选择性吸收
    absorbed_memories: list[dict[str, Any]] = field(default_factory=list)
    # 子灵独立记忆目录路径（供吸收时引用）
    memory_dir: str = ""
    # Phase 4: 正式候选提交（公理 A7）
    proposal: SubagentProposal = field(default_factory=SubagentProposal)

    def to_wm_content(self) -> str:
        status = "完成" if self.completed else ("错误" if self.error else "未完成")
        parts = [f"子灵[{self.subagent_id}] 目标={self.goal!r} 状态={status} ticks={self.ticks_run}"]
        if self.error:
            parts.append(f"错误: {self.error}")
        if self.observations:
            parts.append("关键观察:")
            parts.extend(f"  · {obs}" for obs in self.observations[-5:])  # 最近 5 条
        if self.last_summary:
            parts.append(f"最终结果: {self.last_summary}")
        if self.absorbed_memories:
            parts.append(f"可吸收记忆节点: {len(self.absorbed_memories)} 条（调用 subagent.absorb 合并）")
        return "\n".join(parts)


# ── 工具过滤代理 ────────────────────────────────────────────────────────────────

class _FilteredRegistry:
    """对 ToolRegistry 的只读代理，限制可调用的工具集合。"""

    def __init__(
        self,
        real: ToolRegistry,
        allowed: set[str] | None,
        blocked: set[str],
        local_mutation_allow: set[str] | None = None,
    ) -> None:
        self._real = real
        self._allowed = allowed   # None = 不限制 allowed（只走 blocked）
        self._blocked = blocked
        self._local_mutation_allow = set(local_mutation_allow or ())

    def _is_visible(self, name: str) -> bool:
        if name in self._blocked:
            return False
        if self._allowed is not None and name not in self._allowed:
            return False
        if name in self._local_mutation_allow:
            return True
        entry = self._real.get(name)
        manifest = entry.manifest if entry is not None else None
        return not _is_readonly_blocked_tool(name, manifest)

    def get(self, name: str):
        if not self._is_visible(name):
            return None
        return self._real.get(name)

    def list_manifests(self):
        return [m for m in self._real.list_manifests() if self._is_visible(m.name)]

    def list_manifests_as_dict(self):
        return [m for m in self._real.list_manifests_as_dict() if self._is_visible(m["name"])]

    # 透传 discover / reload_tool（子灵不调用这些，但防止 AttributeError）
    def discover(self, *args, **kwargs):
        return None

    def reload_tool(self, *args, **kwargs):
        return False


def _build_filtered_registry(
    registry: ToolRegistry,
    cfg: SubagentConfig,
) -> _FilteredRegistry:
    blocked = set(_DEFAULT_BLOCKED_TOOLS)
    if cfg.blocked_tools:
        blocked.update(cfg.blocked_tools)
    allowed: set[str] | None = set(cfg.allowed_tools) if cfg.allowed_tools else None
    local_mutation_allow = {"memory.add_semantic"} if cfg.isolated_memory else set()
    return _FilteredRegistry(
        registry,
        allowed,
        blocked,
        local_mutation_allow=local_mutation_allow,
    )


# ── 辅助：读取父灵 Ethos 基线 ────────────────────────────────────────────────────

async def _load_parent_ethos(task_store: TaskStoreViewProtocol | None) -> dict[str, float]:
    """从父灵 TaskStore 读取 soul:ethos_baseline，解析失败返回空 dict，由调用方决定 fallback。"""
    if task_store is None:
        return {}
    try:
        ethos_json, found = await task_store.get_fact("soul:ethos_baseline")
        if not found or not ethos_json:
            return {}
        return json.loads(ethos_json)
    except Exception:
        return {}


# ── 子灵运行器 ──────────────────────────────────────────────────────────────────

class SubagentRunner:
    """
    完整子灵运行器，支持 Tier-0 ~ Tier-3 全部能力。

    Tier-0: judgment + execution 循环，无完整情绪/Ethos 机制
    Tier-1: isolated_memory=True 时使用独立 EpisodicMemory + SemanticMemory
    Tier-2: inherit_ethos=True 时继承父灵价值观基线
    Tier-3: 关键语义记忆随 SubagentResult 返回
    """

    def __init__(
        self,
        sub_cfg: SubagentConfig,
        *,
        judgment: JudgmentLayer,
        execution: ExecutionLayer,
        parent_ctx: ToolContext,
        registry: ToolRegistry,
    ) -> None:
        self._sub_cfg = sub_cfg
        self._judgment = judgment
        self._execution = execution
        self._parent_ctx = parent_ctx
        self._registry = registry

    # ── 公开接口 ────────────────────────────────────────────────────────────────

    async def run(self) -> SubagentResult:
        """执行子灵 tick 循环，返回 SubagentResult。"""
        from core.perception import EmotionState, Percept, derive_ethos_state
        from memory.working import WMItem, WorkingMemory
        from tools.registry import ToolContext

        cfg = self._sub_cfg
        sub_id = cfg.subagent_id
        parent_cfg = self._parent_ctx.config

        _log.info("[subagent][%s] 启动 goal=%r max_ticks=%d isolated=%s inherit_ethos=%s",
                  sub_id, cfg.goal, cfg.max_ticks, cfg.isolated_memory, cfg.inherit_ethos)

        # ── Tier-1: 独立 memory namespace ──────────────────────────────────────
        sub_memory_dir = ""
        if cfg.isolated_memory:
            base_mem_dir = parent_cfg.memory_dir
            sub_mem_path = base_mem_dir / "subagents" / sub_id
            sub_mem_path.mkdir(parents=True, exist_ok=True)
            sub_memory_dir = str(sub_mem_path)

            from store.episodic import EpisodicMemory
            from store.semantic import SemanticMemory
            sub_episodic = EpisodicMemory(
                sub_mem_path / "episodic",
                max_events=getattr(parent_cfg.memory, "max_events", 0),
            )
            sub_semantic = SemanticMemory(sub_mem_path / "semantic")
            _log.debug("[subagent][%s] 独立记忆目录: %s", sub_id, sub_mem_path)
        else:
            # Tier-0: 共享父灵记忆（只读）
            sub_episodic = _SubagentEpisodicView(self._parent_ctx.episodic)
            sub_semantic = _SubagentSemanticView(self._parent_ctx.semantic)

        sub_active_task = _build_subagent_active_task(sub_id, cfg.goal)
        sub_task_store = _SubagentTaskStoreView(
            self._parent_ctx.task_store,
            active_task=sub_active_task,
        )

        # ── Tier-2: Ethos 继承 ─────────────────────────────────────────────────
        inherited_ethos_state = None
        if cfg.inherit_ethos:
            baseline_dict = await _load_parent_ethos(self._parent_ctx.task_store)
            # 用与父灵相同的 derive_ethos_state 逻辑，优先读取 DB，缺失时回退到 config soul baseline。
            from core.perception.ethos import EthosValues
            inherited_ethos_state = derive_ethos_state(
                failure_count=0,
                high_error_streak=0,
                has_active_task=True,
                has_next_step=True,
                perception_trend="neutral",
                emotion_down_regulate_streak=0,
                ethos_cfg=getattr(parent_cfg.soul, "ethos", None) or _default_ethos_cfg(),
                baseline=EthosValues.from_dict(baseline_dict) if baseline_dict else None,
            )
            if baseline_dict:
                _log.debug("[subagent][%s] 已继承父灵 Ethos 基线 keys=%s",
                           sub_id, list(baseline_dict.keys()))
            else:
                _log.debug("[subagent][%s] 父灵 Ethos 基线缺失，回退 cfg.soul.ethos.baseline", sub_id)

        # ── 独立 WM（不影响父灵）──────────────────────────────────────────────
        wm_capacity, wm_token_budget, wm_item_max_tokens = _resolve_subagent_wm_limits(parent_cfg)
        sub_wm = WorkingMemory(
            capacity=wm_capacity,
            token_budget=wm_token_budget,
            item_max_tokens=wm_item_max_tokens,
        )
        sub_wm.add(WMItem(kind="goal", content=f"子灵任务: {cfg.goal}", priority=1.0))

        filtered_reg = _build_filtered_registry(self._registry, cfg)
        task_store_view: TaskStoreViewProtocol = sub_task_store
        episodic_view: EpisodicViewProtocol = sub_episodic
        semantic_view: SemanticViewProtocol = sub_semantic

        # 中性情绪仍应尊重全局 emotion baseline，避免子灵隐式回退到 Python 默认值。
        neutral_emotion = EmotionState.from_config(parent_cfg)

        observations: list[str] = []
        last_summary = ""
        completed = False
        error_msg: str | None = None
        ticks_run = 0

        for tick in range(cfg.max_ticks):
            ticks_run = tick + 1

            # 最小感知快照
            percept = Percept(summary=cfg.goal if tick == 0 else "")

            try:
                from store.episodic import EpisodicMemory as _EpisodicMemory
                from store.semantic import SemanticMemory as _SemanticMemory
                from store.task import TaskStore as _TaskStore
                output = await self._judgment.decide(
                    percept,
                    sub_wm,
                    _cast("_TaskStore", task_store_view),
                    _cast("_EpisodicMemory", episodic_view),
                    _cast("_SemanticMemory", semantic_view),
                    neutral_emotion,
                    user_message=cfg.goal if tick == 0 else "",
                    ethos_state=inherited_ethos_state,  # Tier-2: 传入继承的 Ethos
                    registry_override=filtered_reg,
                )
            except Exception as exc:
                error_msg = f"judgment 异常: {exc}"
                _log.exception("[subagent][%s] tick=%d judgment 异常", sub_id, tick)
                break

            decision = output.decision

            if decision == "wait":
                completed = True
                _log.info("[subagent][%s] tick=%d 决定 wait，任务完成", sub_id, tick)
                break

            if decision != "act":
                completed = (decision == "pause")
                break

            # 构造子灵专用 ToolContext（使用独立记忆）
            sub_ctx = ToolContext(
                config=parent_cfg,
                wm=sub_wm,
                task_store=task_store_view,
                episodic=episodic_view,
                semantic=semantic_view,
                emotion=neutral_emotion,
                active_task=sub_active_task,
                judgment=self._judgment,
                execution=self._execution,
                registry=filtered_reg,
            )

            result = await self._execution.dispatch(output, sub_ctx)

            last_summary = result.summary
            observations.append(f"[tick={ticks_run}] {result.summary}")

            sub_wm.add(WMItem(
                kind=result.kind,
                content=result.summary,
                priority=result.priority,
            ))

            _log.debug("[subagent][%s] tick=%d act=%s summary=%s",
                       sub_id, tick, output.chosen_action_id, result.summary)

        # ── Tier-3: 收集待合并的语义记忆节点 ─────────────────────────────────
        absorbed_memories: list[dict[str, Any]] = []
        if cfg.isolated_memory:
            try:
                # 检索子灵语义记忆中评分最高的节点（最多 10 条）
                nodes = semantic_view.retrieve(cfg.goal, top_k=10)
                absorbed_memories = [
                    dict(item)
                    for item in nodes
                    if isinstance(item, dict)
                    and str(item.get("kind") or "") not in _NON_ABSORBABLE_MEMORY_KINDS
                    and str(item.get("title") or "").strip()
                    and str(item.get("body") or "").strip()
                ]
            except Exception:
                pass  # 内存检索失败不影响结果

        _log.info("[subagent][%s] 结束 ticks=%d completed=%s error=%s memories=%d",
                  sub_id, ticks_run, completed, error_msg, len(absorbed_memories))

        proposal = SubagentProposal(
            observations=observations,
            action_results=[last_summary] if last_summary else [],
            memory_candidates=absorbed_memories,
            self_assessment=last_summary if last_summary else "",
            exceptions=[error_msg] if error_msg else [],
        )

        return SubagentResult(
            subagent_id=sub_id,
            goal=cfg.goal,
            ticks_run=ticks_run,
            completed=completed,
            error=error_msg,
            last_summary=last_summary,
            observations=observations,
            label=cfg.label,
            absorbed_memories=absorbed_memories,
            memory_dir=sub_memory_dir,
            proposal=proposal,
        )


# ── 工厂函数 ────────────────────────────────────────────────────────────────────

def make_subagent_runner(
    sub_cfg: SubagentConfig,
    parent_ctx: ToolContext,
    judgment: JudgmentLayer,
    execution: ExecutionLayer,
    registry: ToolRegistry,
) -> SubagentRunner:
    """根据父灵上下文构造子灵 Runner，供 tools/subagent.py 调用。"""
    return SubagentRunner(
        sub_cfg,
        judgment=judgment,
        execution=execution,
        parent_ctx=parent_ctx,
        registry=registry,
    )
