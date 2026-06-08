"""core/judgment/context/sections.py — WM/记忆/对话等 judgment 上下文 section 格式化。"""
from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from .utils import _cache_put, _context_fmt_cache, _estimate_tokens
from .utils import _clip_for_context

if TYPE_CHECKING:
    from core.config import Config
    from core.perception import EthosState, Percept
    from store.semantic import SemanticMemory
    from tools.registry import ToolManifest


def _fmt_memories(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "（无相关记忆）"
    lines: list[str] = []
    for memory in memories:
        score = memory.get("score")
        score_part = ""
        if isinstance(score, (int, float)):
            score_part = f" (score={float(score):.3f})"
        body_text = _clip_for_context(str(memory.get("body_preview") or memory.get("body") or ""), 320)
        lines.append(f"- [{memory['kind']}] {memory['title']}{score_part}: {body_text}")
    return "\n".join(lines)


def _fmt_episodic(text: str) -> str:
    normalized = str(text or "").strip()
    return _clip_for_context(normalized, 1600) if normalized else "（暂无情节记忆）"


def _fmt_wm(
    items: list[dict[str, Any]],
    wm_count: int = 0,
    wm_capacity: int = 20,
    wm_tokens: int = 0,
    wm_token_budget: int = 0,
) -> str:
    if wm_token_budget > 0:
        header = f"[{wm_count}/{wm_capacity} 条，~{wm_tokens} tokens / {wm_token_budget} 预算，{wm_tokens / wm_token_budget:.0%}]"
    else:
        header = f"[{wm_count}/{wm_capacity}，{wm_count / wm_capacity:.0%}]"
    if not items:
        return f"{header} （工作记忆为空）"
    anti_loop = [item for item in items if item.get("kind") == "self_awareness"]
    rest = [item for item in items if item.get("kind") != "self_awareness"]
    ordered = anti_loop + rest
    lines = [header] + [f"- [{item['kind']}|p={item.get('priority', 0):.2f}] {item['content']}" for item in ordered]
    large_items = sorted(ordered, key=lambda item: _estimate_tokens(item.get("content", "")), reverse=True)
    large_items = [item for item in large_items if _estimate_tokens(item.get("content", "")) > 100]
    if large_items:
        warnings_str = ", ".join(
            f"[{item.get('kind')}] ~{_estimate_tokens(item.get('content', ''))} tokens"
            for item in large_items
        )
        lines.append(f"⚠ 大条目（可能坠占预算）: {warnings_str}")
    return "\n".join(lines)


def _fmt_memory_recall(
    *,
    query: str,
    anchors: list[str],
    memories: list[dict[str, Any]],
    semantic_top_score: float,
    episodic_cross_task_hit: bool,
    daily_fallback_used: bool,
    recall_mode: str,
    chat_id: str = "",
    chat_memory_hits: int = 0,
) -> str:
    query_text = (query or "").strip() or "（空）"
    anchor_text = ", ".join(anchor for anchor in anchors) if anchors else "（无）"
    lines = [
        f"query: {query_text}",
        f"anchors: {anchor_text}",
        f"chat_scope: {chat_id if chat_id else 'none'}",
        f"chat_memory_hits: {chat_memory_hits}",
        f"semantic_hits: {len(memories)}",
        f"semantic_top_score: {semantic_top_score:.3f}",
        f"episodic_cross_task_hit: {'yes' if episodic_cross_task_hit else 'no'}",
        f"daily_fallback_used: {'yes' if daily_fallback_used else 'no'}",
        f"recall_mode: {recall_mode}",
    ]
    return "\n".join(lines)


def _fmt_memory_system(
    *,
    runtime_db: str,
    memory_dir: str,
    workspace_dir: str,
    semantic: SemanticMemory,
    max_concurrent_ticks: int,
    max_tick_queue: int,
) -> str:
    stats = cast(Any, semantic).stats()
    lines = [
        f"runtime_db: {runtime_db}",
        f"memory_dir: {memory_dir}",
        f"workspace_dir: {workspace_dir}",
        f"semantic_db: {stats.get('db_path')}",
        f"semantic_nodes_dir: {stats.get('nodes_dir')}",
        f"semantic_nodes: {int(stats.get('nodes') or 0)}",
        f"semantic_fts5_ok: {'yes' if stats.get('fts5_ok') else 'no'}",
        f"semantic_maintenance_state: {stats.get('maintenance_state') or 'unknown'}",
        f"semantic_maintenance_deferred: {'yes' if stats.get('maintenance_deferred') else 'no'}",
        f"semantic_maintenance_last_error: {stats.get('maintenance_last_error') or 'none'}",
        f"semantic_maintenance_startup_seconds: {float(stats.get('maintenance_last_startup_seconds') or 0.0):.3f}",
        f"semantic_maintenance_background_seconds: {float(stats.get('maintenance_last_background_seconds') or 0.0):.3f}",
        f"embedding_enabled: {'yes' if stats.get('embedding_enabled') else 'no'}",
        f"decay_lambda: {float(stats.get('decay_lambda') or 0.0):.3f}",
        f"tick_dispatch.max_concurrent_ticks: {int(max_concurrent_ticks)}",
        f"tick_dispatch.max_tick_queue: {int(max_tick_queue)}",
    ]
    lines.append("说明: runtime_db 是任务/事实/聊天/运行轨迹主存储；SOUL/IDENTITY/BOOTSTRAP 等 md 是身份与可读镜像层。")
    lines.append("调参提示: 以上 dispatch 上限可通过 config.set 修改 loop.max_concurrent_ticks / loop.max_tick_queue。")
    return "\n".join(lines)


def _fmt_life_state(snapshot: dict[str, Any] | None) -> str:
    """格式化 runtime life snapshot；无快照时保持可读空态。"""
    if not snapshot:
        return "（本轮未提供 runtime life snapshot）"

    memory = snapshot.get("memory") or {}
    startup = snapshot.get("startup") or {}
    pressure = snapshot.get("pressure") or {}
    drive = snapshot.get("drive") or {}
    action = snapshot.get("action") or {}
    top_interests = drive.get("top_interests") or []
    if isinstance(top_interests, list) and top_interests:
        interests_text = ", ".join(
            f"{item.get('domain')}={float(item.get('score') or 0.0):.2f}"
            for item in top_interests
            if isinstance(item, dict)
        ) or "none"
    else:
        interests_text = "none"

    return "\n".join([
        f"memory.wm_pressure: {float(memory.get('wm_pressure') or 0.0):.2f}",
        f"memory.wm_tokens: {int(memory.get('wm_tokens') or 0)} / {int(memory.get('wm_token_budget') or 0)}",
        f"memory.semantic_nodes: {int(memory.get('semantic_nodes') or 0)}",
        f"memory.semantic_maintenance: state={memory.get('semantic_maintenance_state') or 'unknown'} deferred={'yes' if memory.get('semantic_maintenance_deferred') else 'no'} error={memory.get('semantic_maintenance_last_error') or 'none'}",
        f"startup.bootstrap_mode: {startup.get('bootstrap_mode') or 'none'}",
        f"startup.tick_count: {int(startup.get('tick_count') or 0)}",
        f"startup.runtime_ready_callback_pending: {'yes' if startup.get('runtime_ready_callback_pending') else 'no'}",
        f"pressure.dispatch: running={int(pressure.get('dispatch_running') or 0)} pending={int(pressure.get('dispatch_pending') or 0)} queue_pressure={float(pressure.get('dispatch_queue_pressure') or 0.0):.2f}",
        f"pressure.idle_cycles: {int(pressure.get('idle_cycles') or 0)}",
        f"pressure.wait_streak: {int(pressure.get('wait_streak') or 0)}",
        f"drive.overall: {float(drive.get('overall') or 0.0):.2f}",
        f"drive.prediction_error_ema: {float(drive.get('prediction_error_ema') or 0.0):.2f}",
        f"drive.top_interests: {interests_text}",
        f"action.last: decision={action.get('last_decision') or 'none'} tool={action.get('last_tool') or 'none'} status={action.get('last_status') or 'none'} progressful={'yes' if action.get('last_progressful') else 'no'}",
        f"action.progress_reason: {action.get('last_progress_reason') or 'none'}",
    ])


def _fmt_tools(manifests: list[ToolManifest]) -> str:
    if not manifests:
        return "（无可用工具）"
    lines: list[str] = []
    for manifest in manifests:
        params_str = ", ".join(
            f"{param.name}({'*' if param.required else '?'}): {param.description}"
            for param in manifest.params
        )
        lines.append(f"- `{manifest.name}`: {manifest.description}  参数: [{params_str}]")
    return "\n".join(lines)


def _fmt_config_snapshot(cfg: Config) -> str:
    ev = cfg.evolution
    lo = cfg.loop
    me = cfg.memory
    th = cfg.thresholds
    em = cfg.emotion
    so = cfg.soul
    lines = [
        "# 可调运行时参数（通过 config.set <key> <json_value> 修改）",
        "",
        "## 模型",
        f"  model: {cfg.model}",
        f"  temperature: {cfg.temperature}",
        "",
        "## 进化引擎 (evolution.*)",
        f"  enabled: {ev.enabled}",
        f"  competitive_candidates: {ev.competitive_candidates}  # >=2 启用竞争进化",
        f"  trigger_min_failures: {ev.trigger_min_failures}",
        f"  trigger_window_minutes: {ev.trigger_window_minutes}",
        f"  error_streak_evolve: {ev.error_streak_evolve}",
        f"  breaker_fail_threshold: {ev.breaker_fail_threshold}",
        f"  breaker_escalate_threshold: {ev.breaker_escalate_threshold}",
        f"  breaker_cooldown_seconds: {ev.breaker_cooldown_seconds}",
        f"  breaker_global_cooldown_seconds: {ev.breaker_global_cooldown_seconds}",
        f"  ethos_max_delta: {ev.ethos_max_delta}",
        "",
        "## 认知循环 (loop.*)",
        f"  wake_poll_interval: {lo.wake_poll_interval}ms",
        f"  min_act_gap: {lo.min_act_gap}ms",
        f"  active_idle_gap: {lo.active_idle_gap}ms",
        f"  max_idle_gap: {lo.max_idle_gap}ms",
        "",
        "## Emotion guardrails (emotion.*)",
        f"  failure_normalization_count: {em.failure_normalization_count}",
        f"  high_error_normalization_streak: {em.high_error_normalization_streak}",
        f"  feeling_min_intensity: {em.feeling_min_intensity}",
        f"  mood_valence_high: {em.mood_valence_high}",
        f"  mood_valence_low: {em.mood_valence_low}",
        f"  mood_arousal_high: {em.mood_arousal_high}",
        f"  regulation_down_regulate_arousal_high: {em.regulation_down_regulate_arousal_high}",
        f"  regulation_down_regulate_valence_low: {em.regulation_down_regulate_valence_low}",
        f"  regulation_down_regulate_worsening_valence: {em.regulation_down_regulate_worsening_valence}",
        f"  regulation_up_regulate_recovering_valence: {em.regulation_up_regulate_recovering_valence}",
        f"  regulation_up_regulate_signal_valence: {em.regulation_up_regulate_signal_valence}",
        f"  regulation_high_error_streak_guard: {em.regulation_high_error_streak_guard}",
        f"  reflection_valence_history_weight: {em.reflection_valence_history_weight}",
        f"  reflection_valence_hint_weight: {em.reflection_valence_hint_weight}",
        "",
        "## Memory guardrails (memory.*)",
        f"  consolidate_threshold: {me.consolidate_threshold}",
        f"  consolidate_low_pressure_skip_threshold: {me.consolidate_low_pressure_skip_threshold}",
        f"  promotion_priority_threshold: {me.promotion_priority_threshold}",
        f"  promotion_max_nodes_per_consolidation: {me.promotion_max_nodes_per_consolidation}",
        f"  promotion_body_max_chars: {me.promotion_body_max_chars}",
        f"  promotion_reinforce_delta: {me.promotion_reinforce_delta}",
        f"  daily_recall_days: {me.daily_recall_days}",
        f"  daily_recall_max_chars: {me.daily_recall_max_chars}",
        f"  daily_recall_semantic_score_threshold: {me.daily_recall_semantic_score_threshold}",
        f"  daily_summary_days: {me.daily_summary_days}",
        f"  daily_summary_max_chars: {me.daily_summary_max_chars}",
        f"  daily_summary_activation: {me.daily_summary_activation}",
        f"  daily_summary_importance: {me.daily_summary_importance}",
        f"  global_md_warn_bytes: {me.global_md_warn_bytes}",
        f"  global_md_warn_lines: {me.global_md_warn_lines}",
        "",
        "## Ethos guardrails (soul.ethos.*)",
        f"  ema_alpha: {so.ethos.ema_alpha}",
        f"  floor_truth: {so.ethos.floor_truth}",
        f"  floor_caution: {so.ethos.floor_caution}",
        f"  prefer_verification_caution_min: {so.ethos.prefer_verification_caution_min}",
        f"  prefer_verification_failure_count: {so.ethos.prefer_verification_failure_count}",
        f"  prefer_narrow_failure_count: {so.ethos.prefer_narrow_failure_count}",
        f"  prefer_narrow_error_streak: {so.ethos.prefer_narrow_error_streak}",
        f"  preserve_continuity_min: {so.ethos.preserve_continuity_min}",
        f"  avoid_overclaiming_down_regulate_streak: {so.ethos.avoid_overclaiming_down_regulate_streak}",
        f"  failure_adjust_count: {so.ethos.failure_adjust_count}",
        f"  failure_truth_delta: {so.ethos.failure_truth_delta}",
        f"  failure_caution_delta: {so.ethos.failure_caution_delta}",
        f"  failure_curiosity_delta: {so.ethos.failure_curiosity_delta}",
        f"  high_error_adjust_streak: {so.ethos.high_error_adjust_streak}",
        f"  high_error_truth_delta: {so.ethos.high_error_truth_delta}",
        f"  high_error_caution_delta: {so.ethos.high_error_caution_delta}",
        f"  high_error_care_delta: {so.ethos.high_error_care_delta}",
        f"  active_task_continuity_delta: {so.ethos.active_task_continuity_delta}",
        f"  next_step_continuity_delta: {so.ethos.next_step_continuity_delta}",
        f"  next_step_care_delta: {so.ethos.next_step_care_delta}",
        f"  recovering_curiosity_delta: {so.ethos.recovering_curiosity_delta}",
        f"  recovering_care_delta: {so.ethos.recovering_care_delta}",
        "",
        "## Replay guardrails (thresholds.*)",
        f"  prediction_error_task: {th.prediction_error_task}",
        f"  perception_replay_trend_delta: {th.perception_replay_trend_delta}",
        f"  perception_replay_high_error_hint_streak: {th.perception_replay_high_error_hint_streak}",
        f"  emotion_replay_trend_delta: {th.emotion_replay_trend_delta}",
        "",
        "## Judgment guardrails (thresholds.*)",
        f"  judgment_error_streak_guard: {th.judgment_error_streak_guard}",
        f"  judgment_require_more_evidence_worsening_failure_count: {th.judgment_require_more_evidence_worsening_failure_count}",
        f"  judgment_prefer_narrow_failure_count: {th.judgment_prefer_narrow_failure_count}",
        f"  judgment_posture_narrow_failure_count: {th.judgment_posture_narrow_failure_count}",
        f"  judgment_posture_narrow_down_regulate_failure_count: {th.judgment_posture_narrow_down_regulate_failure_count}",
        f"  judgment_posture_pause_worsening_failure_count: {th.judgment_posture_pause_worsening_failure_count}",
        "",
        "## Reference guardrails (thresholds.*)",
        f"  reference_min_confidence: {th.reference_min_confidence}",
        f"  reference_local_signal_base: {th.reference_local_signal_base}",
        f"  reference_local_signal_step: {th.reference_local_signal_step}",
        f"  reference_local_confidence_cap: {th.reference_local_confidence_cap}",
        f"  reference_max_anchors: {th.reference_max_anchors}",
        f"  reference_topic_top_k: {th.reference_topic_top_k}",
        f"  reference_recent_narrative_limit: {th.reference_recent_narrative_limit}",
        f"  reference_recent_semantic_top_k: {th.reference_recent_semantic_top_k}",
        f"  reference_topic_anchor_min_chars: {th.reference_topic_anchor_min_chars}",
        "",
        "## Context facts guardrails (thresholds.*)",
        f"  fact_context_exclude_prefixes: {json.dumps(th.fact_context_exclude_prefixes, ensure_ascii=False)}",
        f"  fact_context_task_limit: {th.fact_context_task_limit}",
        f"  fact_context_global_limit: {th.fact_context_global_limit}",
        f"  fact_context_priority_prefixes: {json.dumps(th.fact_context_priority_prefixes, ensure_ascii=False)}",
        f"  fact_context_priority_limit: {th.fact_context_priority_limit}",
        f"  fact_context_recent_scan_multiplier: {th.fact_context_recent_scan_multiplier}",
        f"  fact_context_recent_scan_min: {th.fact_context_recent_scan_min}",
        "",
        "## Chat history guardrails (thresholds.*)",
        f"  chat_history_turn_limit: {th.chat_history_turn_limit}",
        f"  chat_history_max_chars: {th.chat_history_max_chars}",
        "",
    ]
    return "\n".join(lines)


def _fmt_shell_capabilities() -> str:
    cache_key = "_fmt_shell_capabilities"
    if cache_key in _context_fmt_cache:
        return _context_fmt_cache[cache_key]
    cmds = (
        "python3", "python", "bash", "sh", "grep", "find", "ls", "cat",
        "sqlite3", "git", "sed", "awk", "jq", "rg",
    )
    available = [cmd for cmd in cmds if shutil.which(cmd)]
    payload = {
        "engine": "asyncio.create_subprocess_shell",
        "execution_model": "one-shot-non-persistent",
        "sandbox": False,
        "network_policy": "inherits-host-environment",
        "default_timeout_sec": 30,
        "default_output_preview_chars": None,
        "shell": os.environ.get("SHELL") or "/bin/sh",
        "cwd": os.getcwd(),
        "available_commands": available,
        "missing_commands": [cmd for cmd in cmds if cmd not in available],
    }
    result = json.dumps(payload, ensure_ascii=False, indent=2)
    _cache_put(cache_key, result)
    return result


def _fmt_percept(percept: Percept) -> str:
    lines = [
        f"预测误差: {percept.prediction_error:.2f}",
        f"工作区变更: {'是' if percept.workspace_dirty else '否'}",
    ]
    if getattr(percept, "multimodal_inputs", None):
        lines.append(f"多模态输入: {len(percept.multimodal_inputs)} 条")
        for idx, obs in enumerate(percept.multimodal_inputs[:3], start=1):
            lines.append(f"  - {idx}. {obs}")
        if len(percept.multimodal_inputs) > 3:
            lines.append(f"  - ... 共 {len(percept.multimodal_inputs)} 条")
    return "\n".join(lines)


def _fmt_soul(
    ethos_val: str,
    config_ethos_val: str = "",
) -> str:
    cache_key = f"_fmt_soul:{hash(ethos_val)}:{hash(config_ethos_val)}"
    if cache_key in _context_fmt_cache:
        return _context_fmt_cache[cache_key]
    parts: list[str] = []
    if ethos_val:
        parts.append(f"价值基线（ethos_baseline）: {ethos_val}")
    elif config_ethos_val:
        parts.append(f"价值基线（ethos_baseline，config fallback）: {config_ethos_val}")
    result = "\n".join(parts) if parts else "（Soul 未初始化，运行 `init` 命令生成）"
    _cache_put(cache_key, result)
    return result


def _fmt_ethos(ethos_state: EthosState | None) -> str:
    cache_key = f"_fmt_ethos:{str(ethos_state) if ethos_state else 'none'}"
    if cache_key in _context_fmt_cache:
        return _context_fmt_cache[cache_key]
    if not ethos_state:
        result = "（Ethos 未计算）"
        _cache_put(cache_key, result)
        return result
    values = ethos_state.values
    bias = ethos_state.bias
    lines: list[str] = [
        f"价値图式  truth={values.truth:.2f}  caution={values.caution:.2f}  continuity={values.continuity:.2f}  curiosity={values.curiosity:.2f}  care={values.care:.2f}",
    ]
    biases: list[str] = []
    if bias.prefer_verification:
        biases.append("prefer_verification")
    if bias.prefer_narrow_scope:
        biases.append("prefer_narrow_scope")
    if bias.preserve_continuity:
        biases.append("preserve_continuity")
    if bias.avoid_overclaiming:
        biases.append("avoid_overclaiming")
    if biases:
        lines.append(f"行为倾向  {', '.join(biases)}")
    if bias.reasons:
        lines.append(f"理由      {'; '.join(bias.reasons)}")
    return "\n".join(lines)



def _fmt_current_time() -> str:
    cache_key = "_fmt_current_time"
    if cache_key in _context_fmt_cache:
        return _context_fmt_cache[cache_key]
    now = datetime.now(UTC)
    local_iso = now.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    utc_str = now.strftime("%Y-%m-%d %H:%M UTC")
    result = f"当前时间: {local_iso}\n参考 UTC: {utc_str}\n（时间持续流动，每个 tick 都是真实存在的时刻，由你决定如何使用）"
    _cache_put(cache_key, result)
    return result


def _fmt_chat_history(messages: list[dict[str, Any]], max_chars: int = 300) -> str:
    """将最近 N 条对话消息格式化为 LLM 可读的历史轮次。

    role=user 显示为 '用户:', role=assistant 显示为 '我:'。
    超 max_chars 时仅丢弃**最旧完整轮次**（整行），不在单条消息内截断（ADR 0015）。
    """
    cache_key = f"_fmt_chat_history:{max_chars}:{hash(str(messages)) if messages else 'none'}"
    if cache_key in _context_fmt_cache:
        return _context_fmt_cache[cache_key]
    if not messages:
        result = "（暂无对话历史）"
        _cache_put(cache_key, result)
        return result
    lines = []
    for msg in messages:
        role = msg.get("role", "")
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        label = "用户" if role == "user" else "我"
        lines.append(f"{label}: {content}")
    if not lines:
        result = "（暂无对话历史）"
    elif max_chars > 0:
        while len(lines) > 1 and len("\n".join(lines)) > max_chars:
            lines.pop(0)
        result = "\n".join(lines)
        if len(result) > max_chars and len(lines) == 1:
            result = lines[0]
    else:
        result = "\n".join(lines)
    _cache_put(cache_key, result)
    return result


def _fmt_chat_continuity(text: str) -> str:
    cache_key = f"_fmt_chat_continuity:{hash(text) if text else 'none'}"
    if cache_key in _context_fmt_cache:
        return _context_fmt_cache[cache_key]
    normalized = str(text or "").strip()
    result = _clip_for_context(normalized, 1200) if normalized else "（暂无当前 chat 的连续性记忆）"
    _cache_put(cache_key, result)
    return result


def _fmt_cross_task_episodic(text: str) -> str:
    cache_key = f"_fmt_cross_task_episodic:{hash(text) if text else 'none'}"
    if cache_key in _context_fmt_cache:
        return _context_fmt_cache[cache_key]
    normalized = str(text or "").strip()
    result = _clip_for_context(normalized, 1600) if normalized else "（暂无直接相关的跨任务情节线索）"
    _cache_put(cache_key, result)
    return result


def _fmt_interlocutor_continuity(text: str) -> str:
    cache_key = f"_fmt_interlocutor_continuity:{hash(text) if text else 'none'}"
    if cache_key in _context_fmt_cache:
        return _context_fmt_cache[cache_key]
    normalized = str(text or "").strip()
    result = _clip_for_context(normalized, 1200) if normalized else "（暂无当前交互对象的跨 chat 互动连续性记忆）"
    _cache_put(cache_key, result)
    return result


def _fmt_chat_memories(memories: list[dict[str, Any]]) -> str:
    cache_key = f"_fmt_chat_memories:{hash(str(memories)) if memories else 'none'}"
    if cache_key in _context_fmt_cache:
        return _context_fmt_cache[cache_key]
    if not memories:
        result = "（暂无当前 chat 的长期结晶）"
        _cache_put(cache_key, result)
        return result
    result = _clip_for_context(_fmt_memories(memories), 1200)
    _cache_put(cache_key, result)
    return result
