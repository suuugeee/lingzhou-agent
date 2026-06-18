"""tools/view_protocols.py — 子灵隔离视图的结构化协议（Protocol）。

供 ToolContext / JudgmentLayer.decide() 使用，替换 cast(Any) 类型逃逸。
视图类（_SubagentTaskStoreView 等）和真实存储类（TaskStore 等）均结构上满足这些 Protocol。

公理 A7（子灵授权）：子灵通过隔离视图访问存储，不直接持有父灵 TaskStore 引用。

维护说明
--------
当 TaskStore / EpisodicMemory / SemanticMemory 增减公开方法后，
必须同步更新此文件的 Protocol 声明，否则 pyright 会在调用侧报错。
这是 Python 中等价于 TypeScript interface 的"合约检查"机制。
"""
from __future__ import annotations

from core.execution.run_profile import RUN_TYPE_TOOL_CHAIN, WORKER_TOOL_CHAIN
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TaskStoreViewProtocol(Protocol):
    """TaskStore 完整公开接口协议（供 ToolContext / decide() 参数类型约束）。

    方法签名须与 store/task/__init__.py:TaskStore 保持同步。
    """

    # ── facts ────────────────────────────────────────────────────────────
    async def get_fact(self, key: str) -> tuple[str, bool]: ...
    async def set_fact(self, key: str, value: str, scope: str = "general") -> None: ...
    async def delete_fact(self, key: str) -> None: ...
    async def list_facts(self, prefix: str = "", limit: int = 100) -> list[tuple[str, str]]: ...

    # ── tasks ────────────────────────────────────────────────────────────
    async def add_task(
        self,
        title: str,
        goal: str = "",
        priority: str = "normal",
        source: str = "external",
        *,
        status: str = "pending",
        next_step: str = "",
        chain_id: str = "",
        parent_task_id: str = "",
        current_step: str = "",
        wait_kind: str = "",
        wait_key: str = "",
        state_json: dict[str, Any] | None = None,
        wait_json: dict[str, Any] | None = None,
        result_json: dict[str, Any] | None = None,
        async_job_id: str = "",
        model_tier: str = "",
        extras: dict[str, Any] | None = None,
    ) -> int: ...
    async def get_task_by_id(self, task_id: int) -> Any: ...
    async def get_active(self) -> Any: ...
    async def list_tasks(self, status: str | None = None, limit: int = 50) -> list[Any]: ...
    async def list_runnable_tasks(self, limit: int = 20) -> list[Any]: ...
    async def update_status(
        self,
        task_id: int,
        status: str,
        next_step: str | None = None,
        *,
        current_step: str | None = None,
        model_tier: str | None = None,
        result_json: dict[str, Any] | None = None,
    ) -> None: ...
    async def mark_waiting(
        self,
        task_id: int,
        *,
        wait_kind: str,
        wait_key: str = "",
        wait_json: dict[str, Any] | None = None,
        current_step: str | None = None,
        next_step: str | None = None,
        result_json: dict[str, Any] | None = None,
    ) -> None: ...
    async def resume_task(
        self,
        task_id: int,
        *,
        status: str = "resumed",
        current_step: str | None = None,
        next_step: str | None = None,
        result_json: dict[str, Any] | None = None,
    ) -> None: ...
    async def update_task_data(self, task_id: int, extra_dict: dict[str, Any]) -> None: ...
    async def amend_task(
        self,
        task_id: int,
        *,
        title: str | None = None,
        goal: str | None = None,
        priority: str | None = None,
        amendment_reason: str = "",
    ) -> bool: ...
    async def update_task_result(self, task_id: int, result_json: dict[str, Any]) -> None: ...

    # ── runs ────────────────────────────────────────────────────────────
    async def add_run(
        self,
        *,
        task_id: int = 0,
        run_type: str = RUN_TYPE_TOOL_CHAIN,
        worker_type: str = WORKER_TOOL_CHAIN,
        status: str = "running",
        input_json: dict[str, Any] | None = None,
        output_json: dict[str, Any] | None = None,
        log_text: str = "",
        error_text: str = "",
        tool_name: str = "",
        session_id: str = "",
        model_tier: str = "",
        progress: str = "",
        extras: dict[str, Any] | None = None,
    ) -> int: ...
    async def update_run(
        self,
        run_id: int,
        *,
        task_id: int | None = None,
        status: str | None = None,
        output_json: dict[str, Any] | None = None,
        log_text: str | None = None,
        error_text: str | None = None,
        session_id: str | None = None,
        model_tier: str | None = None,
        progress: str | None = None,
        extras: dict[str, Any] | None = None,
    ) -> None: ...
    async def list_runs(
        self,
        *,
        task_id: int | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[Any]: ...
    async def add_meta_reflection(
        self,
        *,
        reflection_id: str,
        target_kind: str,
        trigger: str,
        loop_level: str,
        diagnosis: str,
        proposal: str,
        verification_plan: str = "",
        decision: str = "defer",
        task_id: int = 0,
        run_id: int = 0,
        tool_name: str = "",
        extras: dict[str, Any] | None = None,
    ) -> None: ...

    # ── failures ─────────────────────────────────────────────────────────
    async def record_failure(self, kind: str, summary: str, context: str = "", task_id: str = "") -> None: ...
    async def dismiss_failure(self, failure_id: int) -> None: ...
    async def list_failures(self, limit: int = 20) -> list[Any]: ...
    async def list_failures_for_task(self, task_id: str, limit: int = 20) -> list[Any]: ...

    # ── signals ──────────────────────────────────────────────────────────
    async def add_signal(self, title: str, run_at: str, repeat_secs: int = 0, payload: dict[str, Any] | None = None) -> int: ...
    async def ack_signal(self, signal_id: int) -> None: ...
    async def cancel_signal(self, signal_id: int) -> None: ...
    async def get_signal(self, signal_id: int) -> dict[str, Any] | None: ...
    async def list_signals(self, limit: int = 30, include_done: bool = False) -> list[dict[str, Any]]: ...

    # ── ledger ───────────────────────────────────────────────────────────
    async def ledger_append(
        self,
        op: str,
        key: str,
        value: str,
        *,
        scope: str = "task",
        source: str = "",
        accepted: bool = True,
        run_id: int = 0,
        reason: str = "",
        proposal_hash: str = "",
        decision_basis: str = "",
    ) -> None: ...
    async def ledger_recent(self, limit: int = 50) -> list[dict[str, Any]]: ...
    async def ledger_since(self, after_id: int, limit: int = 100) -> list[dict[str, Any]]: ...


@runtime_checkable
class EpisodicViewProtocol(Protocol):
    """EpisodicMemory 完整公开接口协议。

    方法签名须与 store/episodic/__init__.py:EpisodicMemory 保持同步。
    """

    def load_for_context(self, task_id: str | None, n_recent: int = 20) -> str: ...
    def load_for_chat_context(
        self,
        chat_id: str | None,
        n_recent: int = 20,
        *,
        max_chars: int | None = None,
    ) -> str: ...
    def load_for_interlocutor_context(
        self,
        interlocutor_id: str | None,
        n_recent: int = 20,
        *,
        max_chars: int | None = None,
    ) -> str: ...
    def load_for_task_narrative(self, task_id: str | None, n_recent: int = 20) -> str: ...
    def load_recent_daily_context(self, days: int = 2, max_chars: int = 1200) -> str: ...
    def search(self, query: str, max_chars: int = 2000, exclude_task_id: str | None = None) -> str: ...
    def get_recent_turns(
        self,
        task_id: str | None = None,
        limit: int = 3,
        *,
        chat_id: str | None = None,
        interlocutor_id: str | None = None,
    ) -> list[dict[str, Any]]: ...
    def list_recent_narrative(self, limit: int = 20) -> list[dict[str, Any]]: ...
    def record(
        self,
        role: str,
        content: str,
        task_id: str | None = None,
        source_type: str = "",
        affect: dict[str, Any] | None = None,
        *,
        chat_id: str | None = None,
        interlocutor_id: str | None = None,
    ) -> None: ...
    def record_event(self, event_type: str, data: dict[str, Any]) -> None: ...


@runtime_checkable
class SemanticViewProtocol(Protocol):
    """SemanticMemory 完整公开接口协议。

    方法签名须与 store/semantic/__init__.py:SemanticMemory 保持同步。
    """

    @property
    def decay_lambda(self) -> float: ...  # 被 memory_ops 读取

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
    ) -> list[dict[str, Any]]: ...
    def retrieve_multi_anchor(
        self,
        anchors: list[str],
        top_k: int = 5,
        convergence_bonus: float = 0.15,
        source: str | None = None,
    ) -> list[dict[str, Any]]: ...
    def upsert(self, node: Any) -> None: ...
    def stats(self) -> dict[str, Any]: ...
