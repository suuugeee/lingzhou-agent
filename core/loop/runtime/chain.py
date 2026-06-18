"""core.loop.runtime.chain — 并发链状态相关 helper。"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import logging
from collections import deque
from typing import Any

from core.judgment import JudgmentLayer
from core.loop.drive.behavior import BehaviorTracker
from core.perception import PerceptionLayer
from memory.working import WorkingMemory

_log = logging.getLogger("lingzhou.loop")


def _build_chain_behavior(loop: Any) -> BehaviorTracker:
    return BehaviorTracker(
        wait_streak_notify=list(loop._cfg.loop.wait_streak_notify),
        streak_threshold=loop._cfg.loop.behavior_streak_threshold,
        wm_priorities={
            "behavior_loop": loop._cfg.thresholds.wm_pri_user_msg,
            "edit_caution": loop._cfg.thresholds.wm_pri_self_aware,
            "belief_stale": loop._cfg.thresholds.wm_pri_critical,
        },
        registry=loop._registry,
        seq_window_warn_at=loop._cfg.thresholds.behavior_seq_window_warn_at,
        seq_window_gap_ratio=loop._cfg.thresholds.behavior_seq_window_gap_ratio,
        belief_stale_threshold=loop._cfg.thresholds.behavior_belief_stale_threshold,
        belief_window=loop._cfg.thresholds.behavior_belief_window,
    )


def new_chain_runtime_state(loop: Any, chain_state_cls: type[Any]) -> dict[str, Any]:
    chain_judgment = JudgmentLayer(loop._provider, loop._registry, loop._cfg)
    chain_judgment.set_identity_prefix(getattr(loop._judgment, "_identity_prefix", ""))
    # routing_providers 不在此处设置：dispatch loop 在每次 tick 前会无条件覆盖，在此调用会导致首 tick 打印重复日志
    chain_judgment._assembler._probe_manager = loop._probe_manager
    # 共享主 judgment 的模型健康表：402/429 cooldown 必须跨 chain 共享，
    # 否则每个新 chain 都会独立重试已达到 cooldown 的模型
    chain_judgment._executor._model_health = loop._judgment._executor._model_health
    chain_judgment._executor._provider_errors = loop._judgment._executor._provider_errors
    with contextlib.suppress(Exception):
        chain_judgment.self_model = copy.deepcopy(loop._judgment.self_model)

    state: dict[str, Any] = {
        "_wm": WorkingMemory(
            capacity=loop._cfg.memory.working_capacity,
            token_budget=loop._cfg.effective_wm_token_budget(),
            item_max_tokens=loop._cfg.memory.wm_item_max_tokens,
        ),
        "_emotion": copy.copy(loop._emotion),  # 继承当前全局情绪，而非重置为 baseline
        "_perception": PerceptionLayer(loop._cfg),
        "_behavior": _build_chain_behavior(loop),
        "_judgment": chain_judgment,
        "_conv_history": deque(maxlen=6),
    }
    for field in dataclasses.fields(chain_state_cls):
        if field.name == "_conv_history":
            continue
        state[field.name] = copy.deepcopy(getattr(loop, field.name))
    return state


def mount_chain_view(view: Any, state: dict[str, Any], chain_state_cls: type[Any]) -> None:
    view._wm = state["_wm"]
    view._emotion = state["_emotion"]
    view._perception = state["_perception"]
    view._behavior = state["_behavior"]
    view._judgment = state["_judgment"]
    for field in dataclasses.fields(chain_state_cls):
        setattr(view, field.name, state[field.name])


def sync_chain_state_from_view(loop: Any, state: dict[str, Any], view: Any, chain_state_cls: type[Any]) -> None:
    for field in dataclasses.fields(chain_state_cls):
        state[field.name] = getattr(view, field.name)
    # 运行镜像：供 wait_after_cycle/state_snapshot 读取最近完成 tick 的状态
    for field in dataclasses.fields(chain_state_cls):
        setattr(loop, field.name, state[field.name])
    # 情绪全局同步：将链的最新情绪回写全局，保证单心智情绪连续性
    # view._emotion 与 state["_emotion"] 是同一对象（mount_chain_view 直接赋引用），
    # 此处用 copy.copy 避免后续链写入影响全局快照。
    loop._emotion = copy.copy(view._emotion)


async def run_dispatched_tick(loop: Any, job: Any, chain_state_cls: type[Any]) -> None:
    try:
        async with loop._dispatch_state_lock:
            state = loop._chain_runtime_state.get(job.chain_key)
            if state is None:
                state = new_chain_runtime_state(loop, chain_state_cls)
                loop._chain_runtime_state[job.chain_key] = state

        view = copy.copy(loop)
        mount_chain_view(view, state, chain_state_cls)
        # 记录当前链标识，供 _maybe_inject_self_drive 判断链类型
        view._current_chain_key = job.chain_key
        # provider 热切换后，链内 judgment 始终跟随当前 provider
        view._judgment._executor._provider = loop._provider
        if loop._routing_providers:
            view._judgment.set_routing_providers(dict(loop._routing_providers))
        view._judgment._assembler._probe_manager = loop._probe_manager

        await view._tick(job.cycle, user_message=job.user_message, chat_id=job.chat_id)
    except Exception:
        if job.chat_message_ids:
            await loop._task_store.release_chat_messages(job.chat_message_ids)
        raise

    if job.chat_message_ids:
        await loop._task_store.mark_chat_messages_processed(job.chat_message_ids)

    async with loop._dispatch_state_lock:
        sync_chain_state_from_view(loop, state, view, chain_state_cls)
