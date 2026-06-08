"""core/judgment/output.py — JudgmentOutput 数据模型及工具分层常量。

从 runtime.py 分离，避免 runtime.py 超长。
依赖：无 core.* 循环（仅用 tools.registry.tool_has_capability）。
"""
from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tools.registry import tool_has_capability
from .context.utils import _clip_for_context

if TYPE_CHECKING:
    from tools.registry import ToolRegistry

def _tool_manifest(tool_id: str, registry: ToolRegistry | None = None) -> Any | None:
    if registry is None or not tool_id:
        return None
    entry = registry.get(tool_id)
    return entry.manifest if entry else None


def is_reader_tool(tool_id: str, registry: ToolRegistry | None = None) -> bool:
    """判断工具是否属于 reader tier，完全依赖 ToolManifest（蓝图模式 2）。"""
    manifest = _tool_manifest(tool_id, registry)
    if manifest is None:
        return False
    if manifest.prefer_tier == "reader":
        return True
    if manifest.prefer_tier in {"reasoner", "repair"}:
        return False
    caps = set(manifest.capabilities or ())
    if "completion_mutation" in caps or "completion_verify" in caps or "multimodal" in caps:
        return False
    if "completion_info_only" in caps and "completion_mutation" not in caps:
        return True
    if manifest.progress_category == "info":
        return True
    if manifest.progress_category in {"mutation", "io"}:
        return False
    return False


def is_plan_alignment_exempt(tool_id: str, registry: ToolRegistry | None = None) -> bool:
    return tool_has_capability(registry, tool_id, "plan_alignment_exempt")


def registry_manifest_signature(registry: ToolRegistry) -> tuple[
    tuple[str, str | None, str, tuple[str, ...]],
    ...,
]:
    """工具清单签名（名称 + 分层相关 manifest 字段），供路由上下文缓存复用。"""
    return tuple(
        (
            manifest.name,
            manifest.prefer_tier,
            manifest.progress_category or "",
            tuple(sorted(manifest.capabilities or ())),
        )
        for manifest in sorted(registry.list_manifests(), key=lambda item: item.name)
    )


def _tier_for_manifest_fields(
    prefer_tier: str | None,
    progress_category: str,
    caps: tuple[str, ...],
) -> str:
    if prefer_tier in {"reader", "reasoner", "repair"}:
        return prefer_tier
    cap_set = set(caps)
    if "completion_mutation" in cap_set or "completion_verify" in cap_set or "multimodal" in cap_set:
        return "reasoner"
    if "completion_info_only" in cap_set and "completion_mutation" not in cap_set:
        return "reader"
    if progress_category in {"mutation", "io"}:
        return "reasoner"
    if progress_category == "info":
        return "reader"
    return "reasoner"


@functools.lru_cache(maxsize=16)
def _tool_tier_mapping_cached(
    signature: tuple[tuple[str, str | None, str, tuple[str, ...]], ...],
) -> dict[str, tuple[str, ...]]:
    mapping: dict[str, list[str]] = {"reader": [], "reasoner": [], "repair": []}
    for name, prefer_tier, progress_category, caps in signature:
        tier = _tier_for_manifest_fields(prefer_tier, progress_category, caps)
        mapping.setdefault(tier, []).append(name)
    return {tier: tuple(sorted(dict.fromkeys(names))) for tier, names in mapping.items()}


def tool_tier_mapping(registry: ToolRegistry | None = None) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {"reader": [], "reasoner": [], "repair": []}
    if registry is None:
        return mapping
    cached = _tool_tier_mapping_cached(registry_manifest_signature(registry))
    return {tier: list(names) for tier, names in cached.items()}


def _structured_tool_history_window(tool_history: list[dict[str, Any]]) -> tuple[str, str]:
    def _trim_value(value: Any, limit: int) -> str:
        return _clip_for_context(str(value or ""), limit)

    def _trim_state_delta(state_delta: dict[str, Any], *, key_limit: int = 16) -> dict[str, Any]:
        if not state_delta:
            return {}
        compacted: dict[str, Any] = {}
        for key in sorted(state_delta):
            if len(compacted) >= key_limit:
                compacted["..."] = f"({len(state_delta) - key_limit} keys omitted)"
                break
            compact_key = _trim_value(key, 64)
            value = state_delta[key]
            if isinstance(value, dict):
                compacted[compact_key] = {
                    str(inner_key): _trim_value(inner_value, 120) for inner_key, inner_value in list(value.items())[:8]
                }
            elif isinstance(value, list):
                compacted[compact_key] = [
                    _trim_value(item, 120) for item in list(value)[:8]
                ]
            else:
                compacted[compact_key] = _trim_value(value, 200)
        return compacted

    history_parts: list[str] = []
    structured_window: list[dict[str, Any]] = []
    start_index = max(0, len(tool_history) - 6)
    for index, item in enumerate(tool_history[start_index:], start=start_index + 1):
        raw_params = item.get("params")
        params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}
        status = str(item.get("status") or "").strip() or (
            "error" if str(item.get("error") or "").strip() else ("skipped" if item.get("skipped") else "ok")
        )
        summary = _trim_value((item.get("summary") or item.get("result") or ""), 400)
        error = str(item.get("error") or "").strip()
        state_delta = item.get("state_delta") if isinstance(item.get("state_delta"), dict) else {}
        compact_state_delta = _trim_state_delta(state_delta, key_limit=8)
        key = (
            params.get("path")
            or params.get("name")
            or params.get("title")
            or params.get("key")
            or str(params.get("id") or "")
            or params.get("command")
            or params.get("query")
            or ""
        )
        structured_window.append({
            "index": index,
            "tool": str(item.get("tool") or ""),
            "status": status,
            "key": str(_trim_value(key, 180)),
            "summary": summary,
            "error": error,
            "error_category": str(item.get("error_category") or ""),
            "state_delta": compact_state_delta,
        })
        parts = [f"[{index}] tool={item.get('tool', '')} status={status}"]
        if key:
            parts.append(f"key={key}")
        if summary:
            parts.append(f"summary={summary}")
        if error:
            parts.append(f"error={error}")
        if state_delta:
            try:
                state_text = json.dumps(compact_state_delta, ensure_ascii=False, sort_keys=True)
            except Exception:
                state_text = str(compact_state_delta)
            parts.append(f"state_delta={state_text}")
        history_parts.append(" | ".join(parts))
    return (
        json.dumps(structured_window, ensure_ascii=False, indent=2),
        "\n".join(history_parts),
    )


def _build_team_view_from_cfg(cfg: Any) -> str:
    """从配置构建思考模型的团队视图（不依赖运行时 model list）。"""
    routing = getattr(cfg, "routing", {}) or {}
    lines = ["## 团队架构（你作为思考模型，统筹以下资源）"]
    for tier in ("reader", "reasoner", "repair"):
        model = routing.get(tier, cfg.model)
        role = {
            "reader": "工具执行层 — 快速/低成本，由系统自动调度执行轻量工具",
            "reasoner": "思考层 — 你本人，负责所有决策、规划、推理与用户交互",
            "repair": "修复层 — 格式修复/小修小补",
        }.get(tier, tier)
        lines.append(f"- {tier}: {model}")
        lines.append(f"  {role}")
    lines.append("")
    lines.append("调度规则:")
    lines.append("- 你是 reasoner（思考层），负责所有判断与决策")
    lines.append("- reader 由系统自动调度，无需也不可通过 next_phase_tier 手动指定")
    lines.append("- 关键决策、代码修改、用户交互必须由你亲自处理")
    return "\n".join(lines)


def tool_tier(tool_id: str, registry: ToolRegistry | None = None) -> str:
    """判断工具应该用哪个 tier。

    优先级：manifest.prefer_tier / capabilities → 硬编码 fallback → 默认 reasoner。
    数据驱动的工具可以声明 prefer_tier，无需改此处。
    """
    manifest = _tool_manifest(tool_id, registry)
    if manifest is not None and manifest.prefer_tier:
        return manifest.prefer_tier
    if manifest is not None:
        caps = set(manifest.capabilities or ())
        if "completion_mutation" in caps or "completion_verify" in caps or "multimodal" in caps:
            return "reasoner"
        if "completion_info_only" in caps and "completion_mutation" not in caps:
            return "reader"
        if manifest.progress_category in {"mutation", "io"}:
            return "reasoner"
        if manifest.progress_category == "info":
            return "reader"
    if is_reader_tool(tool_id, registry):
        return "reader"
    return "reasoner"


# ── 路由/健康数据类 ──────────────────────────────────────────────────────────

@dataclass
class ModelSelection:
    phase: str
    tier: str
    model_ref: str
    thinking: str


@dataclass
class ModelHealth:
    cooldown_until: float = 0.0
    failure_streak: int = 0
    last_error: str = ""
    last_code: str = ""


# ── 判断输出 ───────────────────────────────────────────────────────────────────

@dataclass
class JudgmentOutput:
    decision: str = "wait"              # act | pause | wait
    chosen_action_id: str = ""          # 工具名称
    params: dict[str, Any] = field(default_factory=dict)  # type: ignore[assignment]
    rationale: str = ""                 # 内部推理过程（内部独白）
    reflection: str = ""                # 对最近经历的后验反思（写入语义记忆）
    speech_intent: str = ""             # 大脑发言意图草稿（执行前写入；由口腔器官在执行后确认/修正）
    reply_to_user: str = ""             # 口腔器官生成的最终对外回复（执行后写入，勿在判断阶段直接设置）
    next_step: str = ""
    model_strategy: dict[str, Any] = field(default_factory=dict)
    applied_skills: list[str] = field(default_factory=list)  # LLM 实际应用的技能名单
    parallel_actions: list[dict[str, Any]] = field(default_factory=list)
    delegate_tasks: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def wait(cls, reason: str = "") -> JudgmentOutput:
        return cls(decision="wait", rationale=reason, reply_to_user="")

    @staticmethod
    def _compact_items(prefix: str, items: list[str], *, max_items: int) -> str:
        if not items:
            return ""
        shown = items[:max_items]
        suffix = ""
        if len(items) > max_items:
            suffix = f", +{len(items) - max_items}"
        return f"{prefix}({len(items)})[{', '.join(shown)}{suffix}]"

    def action_label(self, *, max_items: int = 3) -> str:
        if self.chosen_action_id:
            return self.chosen_action_id
        if self.parallel_actions:
            action_ids = [
                str(item.get("action_id") or "").strip()
                for item in self.parallel_actions
                if str(item.get("action_id") or "").strip()
            ]
            return self._compact_items("parallel", action_ids, max_items=max_items)
        if self.delegate_tasks:
            task_ids = [
                str(item.get("id") or "").strip()
                for item in self.delegate_tasks
                if str(item.get("id") or "").strip()
            ]
            return self._compact_items("delegate", task_ids, max_items=max_items)
        return ""

    @staticmethod
    def _coerce_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value)

    @classmethod
    def from_llm(cls, text: str) -> JudgmentOutput:
        """从 LLM 输出文本解析 JudgmentOutput，容错处理。"""
        original = text.strip()
        # 保留 original 不做机械删改；仅用于解析时做局部清洗，避免 think block 破坏 JSON 提取。
        parse_text = original
        parse_text = re.sub(r"<think>[\s\S]*?</think>", "", parse_text).strip()
        if not parse_text.startswith("{") and not parse_text.startswith("```"):
            stripped = re.sub(
                r"(?:^|\n)(?:[\w./_-]+/\s*\n)?(?:[│├└──+|\\]+[ \t]+[\w.+/_-]+[ \t]*(?:#.*)?\n)+",
                "\n",
                parse_text,
            )
            if stripped != parse_text:
                parse_text = stripped.strip()
        if not parse_text or ("{" not in parse_text and "decision" not in parse_text):
            return cls(decision="pause", rationale=f"LLM 输出解析失败（非JSON）: {original}")
        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", parse_text)
        if match:
            parse_text = match.group(1).strip()
        else:
            start = parse_text.find("{")
            end = parse_text.rfind("}")
            if start != -1 and end != -1:
                parse_text = parse_text[start:end + 1]
        try:
            data = json.loads(parse_text)
        except json.JSONDecodeError:
            fixed = parse_text
            fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
            try:
                data = json.loads(fixed)
            except (json.JSONDecodeError, ValueError):
                return cls(decision="pause", rationale=f"LLM 输出解析失败: {parse_text}")

        return cls(
            decision=cls._coerce_text(data.get("decision", "wait")).lower(),
            chosen_action_id=cls._coerce_text(data.get("chosen_action_id", "")),
            params=dict(data.get("params") or {}),
            rationale=cls._coerce_text(data.get("rationale", "")),
            reflection=cls._coerce_text(data.get("reflection", "")),
            reply_to_user=cls._coerce_text(data.get("reply_to_user", "")),
            next_step=cls._coerce_text(data.get("next_step", "")),
            model_strategy=data.get("model_strategy") if isinstance(data.get("model_strategy"), dict) else {},
            applied_skills=[str(s) for s in (data.get("applied_skills") or []) if s],
            parallel_actions=[
                item for item in (data.get("parallel_actions") or [])
                if isinstance(item, dict) and isinstance(item.get("action_id"), str) and item["action_id"]
            ],
            delegate_tasks=[
                item for item in (data.get("delegate_tasks") or [])
                if isinstance(item, dict) and item.get("id") and item.get("goal")
            ],
        )
