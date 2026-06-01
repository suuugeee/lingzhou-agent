"""tools/skill.py — skills catalog / activation / evolution 工具。

给 LLM 提供自我感知与进化能力：
- skill.list     列出当前可发现的 skills catalog
- skill.search   按关键词搜索 skills
- skill.activate 激活并读取完整 SKILL.md
- skill.evolve   根据反馈重写指定 skill 的 workspace 副本

目的：补足"只知道有 skill 存在，却不知道 catalog / location / activation 入口"的缺口，
同时让 skill 演化有明确的工具触发路径，而不只是隐式的内部方法调用。
"""
from __future__ import annotations

from typing import Any

from tools.registry import ToolContext, ToolManifest, ToolParam, ToolResult, tool, tool_metadata


def _skill_origin(skill) -> str:
    return str(getattr(skill, "origin", "dynamic") or "dynamic")


def _load_registry(ctx: ToolContext):
    from core.skill import SkillRegistry
    from tools.paths import skills_dir_from_ctx

    return SkillRegistry(skills_dir=skills_dir_from_ctx(ctx))


def _format_skill_line(skill) -> str:
    origin = _skill_origin(skill)
    triggers = getattr(skill, "triggers", []) or []
    trig = f" | triggers: {', '.join(triggers[:5])}" if triggers else ""
    source = getattr(skill, "source_path", "") or ""
    return f"- {skill.name} [{origin}] — {skill.description}{trig} | source: {source}"


def _skill_metadata_lines(skill) -> list[str]:
    lines = [_format_skill_line(skill)]
    compatibility = str(getattr(skill, "compatibility", "") or "").strip()
    if compatibility:
        lines.append(f"  compatibility: {compatibility}")
    allowed_tools = list(getattr(skill, "allowed_tools", []) or [])
    if allowed_tools:
        lines.append(f"  allowed-tools: {' '.join(allowed_tools)}")
    return lines


@tool(ToolManifest(
    name="skill.list",
    description="列出当前可发现的 skills catalog（seed + workspace）。当你不确定有哪些 skill 可供激活时调用。",
    prefer_tier="reader",
    capabilities=("completion_info_only",),
    params=[
        ToolParam("scope", "string", "all|workspace(custom)|seed(builtin)，默认 all", required=False),
        ToolParam("limit", "number", "最多返回多少条，默认 50", required=False),
    ],
))
async def skill_list(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    reg = _load_registry(ctx)
    skills = reg.all_skills()
    scope = str(params.get("scope") or "all").lower()
    limit = int(params.get("limit") or 50)

    if scope == "custom":
        skills = [s for s in skills if _skill_origin(s) == "workspace"]
    elif scope in {"builtin", "seed"}:
        skills = [s for s in skills if _skill_origin(s) == "seed"]

    skills = sorted(skills, key=lambda s: s.name)[:limit]
    if not skills:
        return ToolResult(summary="（没有匹配的 skills）")
    lines: list[str] = []
    for skill in skills:
        lines.extend(_skill_metadata_lines(skill))
    return ToolResult(summary=f"当前可发现 skills ({len(skills)} 个):\n" + "\n".join(lines))


@tool(ToolManifest(
    name="skill.search",
    description="按关键词搜索当前可发现 skills。当你怀疑有某类 skill 但当前未激活时调用。",
    prefer_tier="reader",
    capabilities=("completion_info_only",),
    params=[
        ToolParam("query", "string", "搜索关键词，如 bug/refactor/提醒/交互/学习", required=True),
        ToolParam("limit", "number", "最多返回多少条，默认 20", required=False),
    ],
))
async def skill_search(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    query = str(params.get("query") or "").strip().lower()
    if not query:
        return ToolResult(summary="query 不能为空", error="EmptyQuery")

    reg = _load_registry(ctx)
    limit = int(params.get("limit") or 20)
    hits = []
    for skill in reg.all_skills():
        hay = " ".join([
            skill.name,
            skill.description or "",
            " ".join(getattr(skill, "tags", []) or []),
            " ".join(getattr(skill, "triggers", []) or []),
        ]).lower()
        if query in hay:
            hits.append(skill)

    hits = sorted(hits, key=lambda s: s.name)[:limit]
    if not hits:
        return ToolResult(summary=f"没有找到与 {query!r} 匹配的 skills")
    lines: list[str] = []
    for skill in hits:
        lines.extend(_skill_metadata_lines(skill))
    return ToolResult(summary=f"skill.search 命中 {len(hits)} 个:\n" + "\n".join(lines))


@tool(ToolManifest(
    name="skill.activate",
    description=(
        "按 skill 名称激活并读取完整 SKILL.md。"
        "用于 Agent Skills 风格的 progressive disclosure：catalog 只给出 name/description，"
        "真正需要该 skill 时再加载完整 instructions 与资源目录。"
    ),
    prefer_tier="reader",
    capabilities=("completion_info_only",),
    params=[
        ToolParam("name", "string", "要激活的 skill 名称", required=True),
        ToolParam("include_frontmatter", "boolean", "是否返回完整 SKILL.md（含 frontmatter），默认 false", required=False),
        ToolParam("guidance_limit", "number", "可选：截断返回内容长度，默认不截断", required=False),
    ],
))
async def skill_activate(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    name = str(params.get("name") or "").strip()
    if not name:
        return ToolResult(summary="name 不能为空", error="EmptySkillName")

    include_frontmatter = bool(params.get("include_frontmatter"))
    guidance_limit_raw = params.get("guidance_limit")
    guidance_limit = int(guidance_limit_raw) if guidance_limit_raw else None

    reg = _load_registry(ctx)
    skill, activation_text = reg.activate(
        name,
        include_frontmatter=include_frontmatter,
        guidance_limit=guidance_limit,
    )
    if skill is None:
        return ToolResult(summary=f"未找到名为 {name!r} 的 skill", error="SkillNotFound")

    resources = skill.list_resources()
    return ToolResult(
        summary=activation_text,
        evidence=skill.description,
        kind="skill_activation",
        priority=0.95,
        resource_key=skill.name,
        metadata=tool_metadata(
            "skill.activate",
            f"skill.activate name={skill.name}",
            skill=skill.name,
            source_path=skill.source_path,
            skill_dir=str(skill.skill_dir),
            resources=resources,
            include_frontmatter=include_frontmatter,
        ),
    )


@tool(ToolManifest(
    name="skill.evolve",
    description=(
        "根据反馈重写指定 skill 的 workspace 副本（不影响仓库内默认 seed）。"
        "用于主动改进某个认知护栏，或在某个 skill 反复触发但方向错误时触发演化。"
        "写入后下一轮 tick 自动热重载。"
    ),
    prefer_tier="reasoner",
    params=[
        ToolParam("name", "string", "要进化的 skill 名称（支持 hyphen 规范名与历史 dotted 名）", required=True),
        ToolParam("feedback", "string", "进化反馈：描述该 skill 哪里需要调整，或期望的新行为方向", required=True),
    ],
))
async def skill_evolve(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    name = str(params.get("name") or "").strip()
    feedback = str(params.get("feedback") or "").strip()
    if not name:
        return ToolResult(summary="name 不能为空", error="EmptySkillName")
    if not feedback:
        return ToolResult(summary="feedback 不能为空", error="EmptyFeedback")

    judgment = getattr(ctx, "judgment", None)
    provider = getattr(judgment, "_provider", None)
    registry = getattr(judgment, "_registry", None)
    if provider is None or registry is None:
        return ToolResult(
            summary="skill.evolve 缺少 judgment provider/registry 上下文",
            error="MissingEvolutionContext",
        )

    try:
        from core.evolution import EvolutionEngine

        engine = EvolutionEngine(ctx.config, provider, registry)
        result = await engine.evolve_skill(name, feedback, ctx=ctx)
    except Exception as exc:
        return ToolResult(summary=f"skill.evolve 内部错误: {exc}", error="EvolutionError")

    if not result.success:
        return ToolResult(
            summary=f"skill {name!r} 进化失败: {result.reason}",
            error="EvolutionFailed",
        )

    return ToolResult(
        summary=f"skill {name!r} 已重写，新 SKILL.md 已写入 workspace。",
        evidence=(result.new_code or ""),
        kind="skill_evolution",
        priority=0.9,
        resource_key=name,
        metadata=tool_metadata(
            "skill.evolve",
            f"skill.evolve name={name}",
            skill=name,
            target=result.target or "",
        ),
    )


@tool(ToolManifest(
    name="skill.synthesize",
    description=(
        "从零合成一个全新的 skill 认知护栏，写入 workspace/skills/ 并立即热重载。\n"
        "用于：没有现成 skill 但你发现某类场景反复出现，需要一个新的行为准则时。\n"
        "若同名 skill 已存在，自动退化为 skill.evolve 逻辑。"
    ),
    prefer_tier="reasoner",
    params=[
        ToolParam("name", "string", "新 skill 的 hyphen 规范名，如 careful-deletion、slow-think", required=True),
        ToolParam("description", "string", "这个 skill 的用途与激活场景描述（100 字以内）", required=True),
    ],
))
async def skill_synthesize(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    name = str(params.get("name") or "").strip()
    description = str(params.get("description") or "").strip()
    # 规范化：小写 + 空格/下划线 → hyphen
    import re as _re
    name = _re.sub(r"[\s_]+", "-", name.lower())
    name = _re.sub(r"[^a-z0-9\-]", "", name).strip("-")
    if not name:
        return ToolResult(summary="name 不能为空", error="EmptySkillName")
    if not description:
        return ToolResult(summary="description 不能为空", error="EmptyDescription")

    judgment = getattr(ctx, "judgment", None)
    provider = getattr(judgment, "_provider", None)
    registry = getattr(judgment, "_registry", None)
    if provider is None or registry is None:
        return ToolResult(
            summary="skill.synthesize 缺少 judgment provider/registry 上下文",
            error="MissingEvolutionContext",
        )

    try:
        from core.evolution import EvolutionEngine

        engine = EvolutionEngine(ctx.config, provider, registry)
        result = await engine.synthesize_skill(name, description, ctx=ctx)
    except Exception as exc:
        return ToolResult(summary=f"skill.synthesize 内部错误: {exc}", error="EvolutionError")

    if not result.success:
        return ToolResult(
            summary=f"skill {name!r} 合成失败: {result.reason}",
            error="SynthesisFailed",
        )

    return ToolResult(
        summary=f"skill {name!r} 已合成写入 workspace，下一轮 tick 自动生效。",
        evidence=(result.new_code or ""),
        kind="skill_synthesis",
        priority=0.9,
        resource_key=name,
        metadata=tool_metadata(
            "skill.synthesize",
            f"skill.synthesize name={name}",
            skill=name,
            target=result.target or "",
        ),
    )


@tool(ToolManifest(
    name="model.upgrade",
    description=(
        "主脑升级协议：切换主脑模型前执行三器官联合确认（公理 A2 Phase 3）。\n"
        "验证记忆 / 人格 / 灵魂三器官均能延续后，将候选模型写入 soul:proposed_model。\n"
        "实际生效需重启系统并在 lingzhou.json 中确认 providers.*.model 字段。\n"
        "拒绝条件：任一器官连续性无法保证，或宪法被篡改。"
    ),
    prefer_tier="reasoner",
    progress_category="mutation",
    params=[
        ToolParam("new_model", "string", "候选主脑模型，格式 'provider/model-id'，如 'bailian/qwen3-plus'", required=True),
        ToolParam("reason", "string", "升级原因（100 字以内）", required=True),
    ],
))
async def model_upgrade(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    new_model = str(params.get("new_model") or "").strip()
    reason = str(params.get("reason") or "").strip()
    if not new_model:
        return ToolResult(summary="new_model 不能为空", skipped=True)
    if not reason:
        return ToolResult(summary="reason 不能为空", skipped=True)

    judgment = getattr(ctx, "judgment", None)
    provider = getattr(judgment, "_provider", None)
    registry = getattr(judgment, "_registry", None)
    if provider is None or registry is None:
        return ToolResult(
            summary="model.upgrade 缺少 judgment provider/registry 上下文",
            error="MissingEvolutionContext",
        )

    try:
        from core.evolution import EvolutionEngine

        engine = EvolutionEngine(ctx.config, provider, registry)
        result = await engine.evolve_model(new_model, reason, ctx=ctx)
    except Exception as exc:
        return ToolResult(summary=f"model.upgrade 内部错误: {exc}", error="EvolutionError")

    if not result.success:
        return ToolResult(
            summary=f"主脑升级协议被拒绝: {result.reason}",
            error="UpgradeRejected",
        )

    return ToolResult(
        summary=result.reason,
        evidence=f"proposed_model={new_model}",
        kind="model_upgrade",
        priority=1.0,
        metadata=tool_metadata(
            "model.upgrade",
            f"model.upgrade proposed={new_model}",
            new_model=new_model,
            reason=reason,
        ),
    )
