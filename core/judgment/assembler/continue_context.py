from __future__ import annotations

from typing import Any

from core.judgment.context.budget import apply_context_budget, resolve_judgment_prompt_budget
from core.judgment.context.utils import _clip_for_context, _estimate_tokens, _fill_template
from core.judgment.tiers import JUDGMENT_TIERS

from ..output import _structured_tool_history_window


def _clip_continue_summary(text: str, limit: int = 2048) -> str:
    return _clip_for_context(text or "", limit)


def _format_continue_wm_delta(wm_delta: list[dict[str, Any]] | None) -> str:
    if not wm_delta:
        return ""
    keep_last = 8
    items = list(wm_delta)[-keep_last:]
    omitted = max(0, len(wm_delta) - len(items))
    delta_lines: list[str] = []
    if omitted:
        delta_lines.append(f"- （已压缩早期 {omitted} 条本轮 WM 更新；完整内容保留在 WM/run 中）")
    for item in items:
        kind = str(item.get("kind", "") or "")
        priority = item.get("priority", 0)
        try:
            priority_text = f"{float(priority):.2f}"
        except Exception:
            priority_text = "0.00"
        content = _clip_for_context(str(item.get("content", "") or ""), 360)
        delta_lines.append(f"- [{kind}|p={priority_text}] {content}")
    return "## 本轮新增工作记忆（WM 更新，初始上下文之后）\n" + "\n".join(delta_lines) + "\n\n"


def _continue_convergence_contract(tool_history: list[dict[str, Any]], *, user_message: str) -> str:
    if not tool_history:
        return ""
    last = tool_history[-1]
    tool = str(last.get("tool") or "").strip() or "unknown"
    params = last.get("params") if isinstance(last.get("params"), dict) else {}
    key = (
        params.get("path")
        or params.get("query")
        or params.get("command")
        or params.get("id")
        or params.get("name")
        or ""
    )
    status = str(last.get("status") or "").strip() or ("error" if last.get("error") else "ok")
    lines = [
        "## Continue 收敛契约",
        f"- 最近动作: tool={tool} status={status}{f' key={key}' if key else ''}",
        "- 先判断最近结果是否已经回答了当前 next_step / 用户问题；若已足够，应生成 reply_to_user 或执行完成/工作台沉淀，而不是继续取同类证据。",
        "- 若证据不足，下一动作必须换一个能降低不确定性的证据源；避免对同一路径、同一查询、同一命令做低增量重复。",
        "- 若发现重复、漂移、失败恢复或完成条件不清，优先 task.workbench 写清 evidence / hypothesis / next_verification / completion_checks。",
    ]
    if user_message:
        lines.append("- 用户正在等待：除非必须再取一个关键证据，否则本轮应尽快形成可验证答复。")
    return "\n".join(lines) + "\n\n"


def _continuation_prompt_budget(assembler: Any) -> int:
    cached = int(getattr(assembler, "_last_context_budget", 0) or 0)
    if cached > 0:
        return cached
    cfg = getattr(assembler, "_cfg", None)
    if cfg is None:
        return 16_000
    catalog_path = assembler._cfg.workspace_dir / "models.json"
    budgets: list[int] = []
    for tier in JUDGMENT_TIERS:
        _, model_ref = assembler._executor._resolve_tier_model(tier)
        budgets.append(resolve_judgment_prompt_budget(assembler._cfg, model_ref, catalog_path=catalog_path))
    return min(budgets) if budgets else assembler._cfg.judgment_input_token_budget()


def _build_continue_base_context(assembler: Any, *, reserve_text: str, reply_only: bool) -> str:
    capsule = str(getattr(assembler, "_last_context_compression_capsule", "") or "").strip()

    sections = dict(getattr(assembler, "_last_context_sections", {}) or {})

    if reply_only:
        # 最终口腔阶段禁止工具调用，工具/技能目录不再消耗上下文预算。
        for key in (
            "tools_section",
            "skills_catalog_section",
            "primary_skill_section",
            "skills_section",
            "shell_capabilities_section",
        ):
            if key in sections:
                sections[key] = ""

    base_budget = _continuation_prompt_budget(assembler)
    cfg = getattr(assembler, "_cfg", None)
    thresholds = getattr(cfg, "thresholds", None)
    reserve_cfg = int(getattr(thresholds, "continue_context_reserve_tokens", 4096) or 4096)
    reserve_tokens = max(reserve_cfg, _estimate_tokens(reserve_text))
    context_budget = max(1024, base_budget - reserve_tokens)
    if capsule:
        # 上一轮 prompt overflow 产生的胶囊是高价值材料，但续判还会追加本轮
        # 工具历史/回复指令；复用时也要纳入本轮 reserve，避免“胶囊+尾部”
        # 再次超预算而反复触发 provider 侧整消息裁剪。
        if _estimate_tokens(capsule) <= context_budget:
            return capsule
        return _clip_for_context(capsule, max(4096, context_budget * 4))

    if not sections:
        raw_context = str(getattr(assembler, "_last_context_text", "") or "")
        if _estimate_tokens(raw_context) <= context_budget:
            return raw_context
        return _clip_for_context(raw_context, max(4096, context_budget * 4))

    sections = apply_context_budget(
        sections,
        context_budget,
        skill_min_tokens=int(getattr(thresholds, "skill_min_budget_tokens", 0) or 0),
    )
    return _fill_template(assembler._judgment_template, sections)


def _format_continue_action_result(action_result: Any | None) -> str:
    if action_result is None:
        return ""
    action_ran = action_result.action_ran
    action_succeeded = action_result.action_succeeded
    if not action_ran:
        status = "未执行（本轮无工具调用）"
    elif action_succeeded is True:
        status = "成功"
    elif action_succeeded is False:
        status = f"失败（{action_result.error or '未知错误'}）"
    else:
        status = "已跳过/不确定"
    tool_line = f"\n- 工具: {action_result.tool_name}" if action_result.tool_name else ""
    summary_line = f"\n- 摘要: {_clip_continue_summary(action_result.summary)}" if action_result.summary else ""
    return (
        "## 本轮执行状态（请据此决定措辞，不要凭推测）\n"
        f"- 是否执行工具: {'是' if action_ran else '否'}\n"
        f"- 执行结果: {status}"
        f"{tool_line}"
        f"{summary_line}\n\n"
    )


def _format_continue_emotion_state(emotion_state: dict[str, Any] | None) -> str:
    if not emotion_state:
        return ""
    dominant = emotion_state.get("dominant", "")
    valence = emotion_state.get("valence", 0.0)
    arousal = emotion_state.get("arousal", 0.0)
    regulation = emotion_state.get("regulation_strategy", "")
    return (
        "## 当前情绪状态（请据此自然调整语气，无需显式说出情绪名）\n"
        f"- 主导情绪: {dominant}\n"
        f"- valence={valence:.2f}（正=积极 负=消极）  arousal={arousal:.2f}（高=紧张 低=平静）\n"
        f"- 调节策略: {regulation}\n\n"
    )


def _build_continue_context(
    assembler: Any,
    tool_history: list[dict[str, Any]],
    *,
    user_message: str,
    reply_only: bool,
    wm_delta: list[dict[str, Any]] | None,
    speech_intent: str = "",
    action_result: Any | None = None,
    emotion_state: dict[str, Any] | None = None,
) -> str:
    history_json_block, history_block = _structured_tool_history_window(tool_history)
    wm_delta_block = _format_continue_wm_delta(wm_delta)
    action_result_block = _format_continue_action_result(action_result) if reply_only else ""
    emotion_block = _format_continue_emotion_state(emotion_state) if reply_only else ""
    convergence_block = "" if reply_only else _continue_convergence_contract(tool_history, user_message=user_message)
    common_tail = (
        f"{wm_delta_block}"
        f"{action_result_block}"
        f"{emotion_block}"
        f"{convergence_block}"
        "## 结构化最近工具结果(JSON)\n"
        f"{history_json_block}\n\n"
        "## 本轮已执行工具历史\n"
        f"{history_block}\n\n"
    )

    if reply_only:
        intent_hint = f"\n执行前意图草稿「{speech_intent}」，请基于实际执行结果确认或修正。" if speech_intent else ""
        final_instruction = (
            f"你现在处于最终回复阶段。禁止再调用任何工具。{intent_hint}\n"
            "请只基于已有证据生成对用户的最终 reply_to_user。"
            "decision 只能是 pause 或 wait，chosen_action_id 必须留空。"
        )
    else:
        hint = "用户正在等待回复，尽快在本轮设置 reply_to_user 字段。" if user_message else ""
        final_instruction = (
            "优先依据结构化结果判断当前状态，不要只凭模糊回忆续写。\n\n"
            f"请根据以上结果继续执行下一个必要工具，或生成最终回复（reply_to_user 非空）。{hint}"
        )
    base_context = _build_continue_base_context(
        assembler,
        reserve_text=common_tail + final_instruction,
        reply_only=reply_only,
    )
    return (
        f"{base_context}\n\n"
        "---\n"
        f"{common_tail}"
        f"{final_instruction}"
    )
