from __future__ import annotations

import json
import re
import traceback
from typing import TYPE_CHECKING

from core.metabolic import set_soul_fact
from core.perception.ethos import ETHOS_DIMENSIONS
from core.skill import _split_frontmatter, ensure_workspace_skill_file, workspace_skill_file

from .types import EvolutionResult

if TYPE_CHECKING:
    from core.evolution import EvolutionEngine
    from tools.registry import ToolContext


_ETHOS_REFLECTION_BODY_MAX_CHARS = 900
_ETHOS_REFLECTION_TOTAL_MAX_CHARS = 4000


def _node_value(node: object, key: str, default: str = "") -> str:
    if isinstance(node, dict):
        return str(node.get(key) or default)
    return str(getattr(node, key, default) or default)


def _clip_ethos_reflection_text(text: str, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(32, limit - 32)].rstrip() + "\n...[reflection truncated]..."


async def evolve_model(engine: EvolutionEngine, new_model: str, reason: str, ctx: ToolContext) -> EvolutionResult:
    """升级协议：主脑模型切换的三器官联合确认（公理 A2 Phase 3）。"""

    from core.immune.constitution import verify_constitution_integrity
    from core.immune.policy import three_organ_preflight

    organ_failures = await three_organ_preflight(ctx.task_store)
    if organ_failures:
        block_msg = "升级协议-三器官预检失败：" + "；".join(organ_failures)
        await set_soul_fact(
            ctx,
            key="soul:upgrade_blocked_reason",
            value=block_msg,
            scope="system",
            source="evolution/upgrade",
            decision_basis="evolution preflight failed; recorded upgrade block reason",
        )
        return EvolutionResult(success=False, target=f"model:{new_model}", reason=block_msg)

    constitution_status = verify_constitution_integrity()
    if constitution_status not in ("ok", "uninitialized"):
        block_msg = f"升级协议-免疫器官拒绝：宪法状态={constitution_status}"
        return EvolutionResult(success=False, target=f"model:{new_model}", reason=block_msg)

    import time as _time

    upgrade_record = json.dumps({
        "new_model": new_model,
        "reason": reason,
        "ts": _time.time(),
    }, ensure_ascii=False)
    await set_soul_fact(
        ctx,
        key=f"soul:upgrade_event:{int(_time.time())}",
        value=upgrade_record,
        scope="system",
        source="evolution/upgrade",
        decision_basis="upgrade record snapshot for restart continuity",
    )
    await set_soul_fact(
        ctx,
        key="soul:proposed_model",
        value=new_model,
        scope="system",
        source="evolution/upgrade",
        decision_basis="model evolve proposal snapshot",
    )

    return EvolutionResult(
        success=True,
        target=f"model:{new_model}",
        reason=f"三器官联合确认通过，候选模型 {new_model!r} 已记录至 soul:proposed_model，重启后生效",
    )


async def evolve_ethos(engine: EvolutionEngine, ctx: ToolContext) -> EvolutionResult:
    """根据近期经历主动调整 ethos_baseline（价值观基线）。"""

    if not engine._cfg.evolution.enabled:
        return EvolutionResult(success=False, target="ethos_baseline", reason="evolution disabled")

    from provider.base import Message

    dims = ETHOS_DIMENSIONS
    baseline_seed = engine._cfg.soul.ethos.baseline

    current_json, _ = await ctx.task_store.get_fact("soul:ethos_baseline")
    current_raw = json.loads(current_json) if current_json else {}
    if not isinstance(current_raw, dict):
        current_raw = {}
    missing_dims = [dim for dim in dims if dim not in current_raw]
    current_baseline: dict[str, float] = {
        dim: float(current_raw.get(dim, getattr(baseline_seed, dim)))
        for dim in dims
    }
    if current_json and missing_dims:
        baseline_source = "DB + config fallback"
    elif current_json:
        baseline_source = "DB"
    else:
        baseline_source = "config fallback"

    try:
        try:
            reflection_nodes = ctx.semantic.retrieve("reflection 近期经历感悟", top_k=5, kind="reflection")
        except TypeError:
            reflection_nodes = ctx.semantic.retrieve("reflection 近期经历感悟", top_k=5)
        reflections = [n for n in reflection_nodes if _node_value(n, "kind") == "reflection"]
    except Exception:
        reflections = []

    if not reflections:
        return EvolutionResult(success=False, target="ethos_baseline", reason="no reflections yet")

    reflection_text = "\n".join(
        f"- [{_node_value(r, 'title')}] {_clip_ethos_reflection_text(_node_value(r, 'body'), _ETHOS_REFLECTION_BODY_MAX_CHARS)}"
        for r in reflections
    )
    reflection_text = _clip_ethos_reflection_text(reflection_text, _ETHOS_REFLECTION_TOTAL_MAX_CHARS)

    from core.immune import extract_constitution_boundaries, load_constitution

    constitution_text = load_constitution(engine._cfg.constitution_path)
    hard_boundaries = extract_constitution_boundaries(constitution_text)

    messages = [
        Message(role="system", content=(
            "你是灵舟的自我成长模块，负责根据近期行为反思调整价值观基线。\n"
            "只输出一个合法的 JSON 对象，包含五个 float 字段：truth, caution, continuity, curiosity, care。\n"
            "每个值在 [0.0, 1.0] 之间。不要有任何其他文字。"
        )),
        Message(role="user", content=(
            f"当前 ethos_baseline（{baseline_source}）：\n{json.dumps(current_baseline, ensure_ascii=False)}\n\n"
            f"近期 reflection 片段：\n{reflection_text}\n\n"
            f"CONSTITUTION.md 硬边界（这些约束对应的维度不允许降低）：\n{chr(10).join(hard_boundaries) if hard_boundaries else '（未加载）'}\n\n"
            "请根据近期反思，判断当前价值基线是否需要微调（每个维度调整幅度不超过 ±0.15）。\n"
            "如不需要调整，直接原样返回当前值。\n"
            "只输出 JSON，例如：{\"truth\": 0.72, \"caution\": 0.68, \"continuity\": 0.65, \"curiosity\": 0.58, \"care\": 0.61}"
        )),
    ]

    try:
        raw = await engine._provider.chat(messages)
        raw = raw.strip()
        json_match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if not json_match:
            return EvolutionResult(success=False, target="ethos_baseline", reason=f"LLM 未返回 JSON: {raw}")
        proposed: dict[str, float] = json.loads(json_match.group())
    except Exception as exc:
        return EvolutionResult(success=False, target="ethos_baseline", reason=str(exc))

    max_delta = engine._cfg.evolution.ethos_max_delta
    validated: dict[str, float] = {}
    clamped_dims: list[str] = []
    for dim in dims:
        if dim not in proposed:
            return EvolutionResult(success=False, target="ethos_baseline", reason=f"缺少维度: {dim}")
        new_val = float(proposed[dim])
        if not (0.0 <= new_val <= 1.0):
            return EvolutionResult(success=False, target="ethos_baseline", reason=f"{dim}={new_val} 超出 [0,1]")
        old_val = current_baseline.get(dim, 0.5)
        if abs(new_val - old_val) > max_delta:
            clamped_val = old_val + max_delta * (1 if new_val > old_val else -1)
            clamped_dims.append(f"{dim}: {new_val:.3f}→{clamped_val:.3f}")
            new_val = clamped_val
        if any(dim in boundary.lower() for boundary in hard_boundaries) and new_val < old_val:
            new_val = old_val
        validated[dim] = round(max(0.0, min(1.0, new_val)), 4)

    await set_soul_fact(
        ctx,
        key="soul:ethos_baseline",
        value=json.dumps(validated),
        source="evolution/ethos",
        decision_basis="ethos evolution from recent reflections and constitution boundaries",
    )
    await engine._update_dreams(f"价值观微调：{validated}", ctx=ctx)
    clamp_note = f"夹幅修正: {'; '.join(clamped_dims)}" if clamped_dims else ""
    return EvolutionResult(success=True, target="ethos_baseline", new_code=json.dumps(validated), reason=clamp_note)


async def evolve_prompt(engine: EvolutionEngine, prompt_key: str, feedback: str) -> EvolutionResult:
    """根据解析失败反馈改进提示词模板（无需语法编译，最安全的进化路径）。"""

    from provider.base import Message

    try:
        prompt_path = engine._cfg.resolve(getattr(engine._cfg.prompts, prompt_key))
    except AttributeError:
        return EvolutionResult(success=False, target=f"prompt:{prompt_key}", reason="未知 prompt key")

    current_src = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""

    messages = [
        Message(role="system", content=(
            "你是 lingzhou 的自进化模块，负责改进 LLM 提示词模板。"
            "只输出改进后的完整 Markdown 模板内容，不要有任何额外文字。"
        )),
        Message(role="user", content=(
            f"以下判断提示词模板导致 LLM 持续输出非 JSON 格式，产生解析失败。\n\n"
            f"当前模板：\n{current_src}\n\n"
            f"失败记录：\n{feedback}\n\n"
            f"请改进模板，使 LLM 更可靠地输出正确 JSON。"
            f"重点检查：输出格式说明是否清晰？JSON 示例是否准确？有无歧义指令？"
        )),
    ]

    try:
        new_src = await engine._provider.chat(messages)
        new_src = new_src.strip()
        if not new_src:
            return EvolutionResult(success=False, target=f"prompt:{prompt_key}", reason="LLM 返回空内容")

        required_markers = (
            '"decision"',
            '"chosen_action_id"',
            '"params"',
            '"rationale"',
            '"reflection"',
            '"reply_to_user"',
            '"next_step"',
        )
        if not all(marker in new_src for marker in required_markers):
            return EvolutionResult(success=False, target=f"prompt:{prompt_key}", reason="提示词校验失败：缺少必要 JSON 结构说明")

        if engine._cfg.evolution.backup and prompt_path.exists():
            prompt_path.with_suffix(".md.bak").write_text(
                prompt_path.read_text(encoding="utf-8"), encoding="utf-8"
            )

        prompt_path.write_text(new_src, encoding="utf-8")
        await engine._update_dreams(f"调整判断模式：{prompt_key} 提示词已根据解析失败反馈重写，输出格式更稳定。")
        return EvolutionResult(success=True, target=f"prompt:{prompt_key}", new_code=new_src)
    except Exception as exc:
        if engine._cfg.evolution.backup and prompt_path.exists() and prompt_path.with_suffix(".md.bak").exists():
            prompt_path.write_text(prompt_path.with_suffix(".md.bak").read_text(encoding="utf-8"), encoding="utf-8")
        return EvolutionResult(success=False, target=f"prompt:{prompt_key}", reason=str(exc))


async def evolve_skill(
    engine: EvolutionEngine,
    skill_name: str,
    feedback: str,
    ctx: ToolContext | None = None,
) -> EvolutionResult:
    """根据反馈重写 workspace skill 文件。"""

    from provider.base import Message

    workspace_dir = engine._cfg.workspace_dir
    skill_path = ensure_workspace_skill_file(workspace_dir, skill_name)
    current_src = skill_path.read_text(encoding="utf-8") if skill_path.exists() else ""

    messages = [
        Message(
            role="system",
            content=(
                "你是 lingzhou 的自进化模块，负责改进 skill 文件。"
                "只输出完整的 SKILL.md Markdown 内容，不要有任何额外文字。"
                "保留或补全 frontmatter，至少包含 name 和 description。"
            ),
        ),
        Message(
            role="user",
            content=(
                f"目标 skill：{skill_name}\n"
                f"workspace skill path：{skill_path}\n\n"
                f"当前 SKILL.md：\n{current_src}\n\n"
                f"反馈：\n{feedback}\n\n"
                "请直接重写完整 skill 文件。若当前内容为空，也请输出完整可用的 SKILL.md。"
                "改动目标是 runtime workspace 副本，而不是仓库内的默认 seed 模板。"
            ),
        ),
    ]

    try:
        new_src = await engine._provider.chat(messages)
        new_src = new_src.strip()
        if not new_src:
            return EvolutionResult(success=False, target=f"skill:{skill_name}", reason="LLM 返回空内容")

        meta, _body = _split_frontmatter(new_src)
        if not meta.get("name") or not meta.get("description"):
            return EvolutionResult(success=False, target=f"skill:{skill_name}", reason="skill 校验失败：缺少 name 或 description")
        if str(meta.get("name") or "").strip() != skill_name:
            return EvolutionResult(success=False, target=f"skill:{skill_name}", reason="skill 校验失败：name 与目标不一致")

        if engine._cfg.evolution.backup and skill_path.exists():
            skill_path.with_suffix(".md.bak").write_text(current_src, encoding="utf-8")

        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(new_src, encoding="utf-8")

        judgment = getattr(ctx, "judgment", None) if ctx is not None else None
        reload_skills = getattr(judgment, "reload_skills", None)
        if callable(reload_skills):
            reload_skills()

        await engine._update_dreams(f"调整 skill：{skill_name} 已根据反馈重写 workspace 副本。")
        return EvolutionResult(success=True, target=f"skill:{skill_name}", new_code=new_src)
    except Exception as exc:
        backup_path = skill_path.with_suffix(".md.bak")
        if engine._cfg.evolution.backup and backup_path.exists():
            skill_path.write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")
        return EvolutionResult(success=False, target=f"skill:{skill_name}", reason=str(exc))


async def synthesize_skill(
    engine: EvolutionEngine,
    skill_name: str,
    description: str,
    *,
    ctx: ToolContext | None = None,
) -> EvolutionResult:
    """从零合成一个新技能 SKILL.md 并写入 workspace/skills/。"""

    from provider.base import Message

    workspace_dir = engine._cfg.workspace_dir
    skill_path = workspace_skill_file(workspace_dir, skill_name)
    if skill_path.exists():
        return await evolve_skill(engine, skill_name, description, ctx=ctx)

    messages = [
        Message(
            role="system",
            content=(
                "你是 lingzhou 的自进化模块，负责合成新的 skill 认知护栏文件。\n"
                "输出格式：完整的 SKILL.md Markdown 文件，以 YAML frontmatter 开头。\n"
                "frontmatter 必须包含：name、description、tags（列表）、triggers（列表）。\n"
                "正文为该 skill 的激活指导文本，描述灵舟在此 skill 激活时应做什么、避免什么。\n"
                "长度：frontmatter + 正文合计 100~400 字，简洁清晰。\n"
                "只输出 SKILL.md 内容，不要任何额外文字或代码块。"
            ),
        ),
        Message(
            role="user",
            content=(
                f"技能名称：{skill_name}\n"
                f"期望描述：{description}\n\n"
                "请为这个认知护栏合成完整的 SKILL.md，让灵舟能在合适的场景下激活它。"
            ),
        ),
    ]
    try:
        new_src = (await engine._provider.chat(messages)).strip()
        if not new_src:
            return EvolutionResult(success=False, target=f"skill:{skill_name}", reason="LLM 返回空内容")

        meta, _body = _split_frontmatter(new_src)
        if not meta.get("name") or not meta.get("description"):
            return EvolutionResult(
                success=False, target=f"skill:{skill_name}",
                reason="skill 校验失败：缺少 name 或 description"
            )
        if str(meta.get("name") or "").strip() != skill_name:
            return EvolutionResult(
                success=False, target=f"skill:{skill_name}",
                reason=f"skill 校验失败：name 与目标 {skill_name!r} 不一致"
            )

        skill_path.parent.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(new_src, encoding="utf-8")

        judgment = getattr(ctx, "judgment", None) if ctx is not None else None
        reload_skills = getattr(judgment, "reload_skills", None)
        if callable(reload_skills):
            reload_skills()

        await engine._update_dreams(f"合成新技能：{skill_name}——{description}", ctx=ctx)
        return EvolutionResult(success=True, target=f"skill:{skill_name}", new_code=new_src)
    except Exception:
        return EvolutionResult(
            success=False, target=f"skill:{skill_name}",
            reason=traceback.format_exc(limit=3)
        )


__all__ = [
    "evolve_ethos",
    "evolve_model",
    "evolve_prompt",
    "evolve_skill",
    "synthesize_skill",
]
