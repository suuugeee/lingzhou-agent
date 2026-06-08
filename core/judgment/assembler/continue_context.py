from __future__ import annotations

from typing import Any

from core.judgment.context.budget import apply_context_budget, resolve_judgment_prompt_budget
from core.judgment.context.utils import _clip_for_context, _estimate_tokens, _fill_template

from ..output import _structured_tool_history_window


def _clip_continue_summary(text: str, limit: int = 2048) -> str:
    return _clip_for_context(text or "", limit)


def _continuation_prompt_budget(assembler: Any) -> int:
    cached = int(getattr(assembler, "_last_context_budget", 0) or 0)
    if cached > 0:
        return cached
    catalog_path = assembler._cfg.workspace_dir / "models.json"
    budgets: list[int] = []
    for tier in ("reader", "reasoner", "repair"):
        _, model_ref = assembler._executor._resolve_tier_model(tier)
        budgets.append(resolve_judgment_prompt_budget(assembler._cfg, model_ref, catalog_path=catalog_path))
    return min(budgets) if budgets else assembler._cfg.judgment_input_token_budget()


def _build_continue_base_context(assembler: Any, *, reserve_text: str, reply_only: bool) -> str:
    sections = dict(getattr(assembler, "_last_context_sections", {}) or {})
    if not sections:
        return str(getattr(assembler, "_last_context_text", "") or "")

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
    reserve_cfg = int(getattr(assembler._cfg.thresholds, "continue_context_reserve_tokens", 4096) or 4096)
    reserve_tokens = max(reserve_cfg, _estimate_tokens(reserve_text))
    context_budget = max(1024, base_budget - reserve_tokens)
    sections = apply_context_budget(
        sections,
        context_budget,
        skill_min_tokens=assembler._cfg.thresholds.skill_min_budget_tokens,
    )
    return _fill_template(assembler._judgment_template, sections)


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
    wm_delta_block = ""
    if wm_delta:
        delta_lines = [f"- [{item.get('kind', '')}|p={item.get('priority', 0):.2f}] {item.get('content', '')}" for item in wm_delta]
        wm_delta_block = "## 本轮新增工作记忆（WM 更新，初始上下文之后）\n" + "\n".join(delta_lines) + "\n\n"
    action_result_block = ""
    if action_result is not None:
        _ran = action_result.action_ran
        _succ = action_result.action_succeeded
        if not _ran:
            _status_str = "未执行（本轮无工具调用）"
        elif _succ is True:
            _status_str = "成功"
        elif _succ is False:
            _status_str = f"失败（{action_result.error or '未知错误'}）"
        else:
            _status_str = "已跳过/不确定"
        _tool_str = f"\n- 工具: {action_result.tool_name}" if action_result.tool_name else ""
        _summary_str = (
            f"\n- 摘要: {_clip_continue_summary(action_result.summary)}"
            if action_result.summary
            else ""
        )
        action_result_block = (
            "## 本轮执行状态（请据此决定措辞，不要凭推测）\n"
            f"- 是否执行工具: {'是' if _ran else '否'}\n"
            f"- 执行结果: {_status_str}"
            f"{_tool_str}"
            f"{_summary_str}\n\n"
        )

    emotion_block = ""
    if emotion_state:
        _dom = emotion_state.get("dominant", "")
        _val = emotion_state.get("valence", 0.0)
        _aro = emotion_state.get("arousal", 0.0)
        _reg = emotion_state.get("regulation_strategy", "")
        emotion_block = (
            "## 当前情绪状态（请据此自然调整语气，无需显式说出情绪名）\n"
            f"- 主导情绪: {_dom}\n"
            f"- valence={_val:.2f}（正=积极 负=消极）  arousal={_aro:.2f}（高=紧张 低=平静）\n"
            f"- 调节策略: {_reg}\n\n"
        )

    common_tail = (
        f"{wm_delta_block}"
        f"{action_result_block}"
        f"{emotion_block}"
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
        base_context = _build_continue_base_context(
            assembler,
            reserve_text=common_tail + final_instruction,
            reply_only=True,
        )
        return (
            f"{base_context}\n\n"
            "---\n"
            f"{common_tail}"
            f"{final_instruction}"
        )

    hint = "用户正在等待回复，尽快在本轮设置 reply_to_user 字段。" if user_message else ""
    final_instruction = (
        "优先依据结构化结果判断当前状态，不要只凭模糊回忆续写。\n\n"
        f"请根据以上结果继续执行下一个必要工具，或生成最终回复（reply_to_user 非空）。{hint}"
    )
    base_context = _build_continue_base_context(
        assembler,
        reserve_text=common_tail + final_instruction,
        reply_only=False,
    )
    return (
        f"{base_context}\n\n"
        "---\n"
        f"{common_tail}"
        f"{final_instruction}"
    )
