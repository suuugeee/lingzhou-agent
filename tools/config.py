"""tools/config.py — config.get / config.set 工具（LLM 可自主调参）。

LLM 通过这些工具读取和修改自己的配置，无需人工编辑文件。
修改后自动触发热重载（loop 检测 mtime 变化）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from tools.registry import ToolContext, ToolManifest, ToolParam, ToolResult, tool, tool_metadata

CONFIG_PATH = Path("~/.lingzhou/lingzhou.json").expanduser()

_DURATION_VALUE_RE = re.compile(
    r"^\s*(?P<number>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?P<unit>ms|msec|milliseconds?|s|sec|secs|seconds?|m|min|mins|minutes?|h|hr|hrs|hours?)\s*$",
    re.IGNORECASE,
)

_DURATION_UNIT_TO_MS = {
    "ms": 1.0,
    "msec": 1.0,
    "millisecond": 1.0,
    "milliseconds": 1.0,
    "s": 1000.0,
    "sec": 1000.0,
    "secs": 1000.0,
    "second": 1000.0,
    "seconds": 1000.0,
    "m": 60_000.0,
    "min": 60_000.0,
    "mins": 60_000.0,
    "minute": 60_000.0,
    "minutes": 60_000.0,
    "h": 3_600_000.0,
    "hr": 3_600_000.0,
    "hrs": 3_600_000.0,
    "hour": 3_600_000.0,
    "hours": 3_600_000.0,
}


def _resolve_config_path(ctx: ToolContext | None = None) -> Path:
    cfg_obj = getattr(ctx, "config", None) if ctx is not None else None
    base_dir = getattr(cfg_obj, "_base_dir", None)
    if isinstance(base_dir, Path):
        candidate = base_dir / "lingzhou.json"
        if candidate.exists():
            return candidate
    return CONFIG_PATH


def _read_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


def _write_config(config_path: Path, cfg: dict) -> None:
    config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def _nested_get(d: dict, path: str) -> Any:
    """点号路径读取，如 'loop.max_idle_gap' → cfg['loop']['max_idle_gap']。"""
    keys = path.split(".")
    current = d
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k)
        else:
            return None
    return current


def _nested_set(d: dict, path: str, value: Any) -> None:
    """点号路径写入，自动创建中间字典。"""
    keys = path.split(".")
    current = d
    for k in keys[:-1]:
        if k not in current or not isinstance(current[k], dict):
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value


def _nested_has(d: dict[str, Any], path: str) -> bool:
    keys = path.split(".")
    current: Any = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


def _deprecated_key_hint(key: str) -> str:
    if key == "loop.interval":
        return (
            "固定 tick interval 已废弃；当前 runtime 是事件驱动。"
            "若想把响应粒度调到约 100ms，优先考虑 loop.wake_poll_interval、"
            "loop.min_act_gap 或 loop.active_idle_gap。"
        )
    return ""


def _normalize_numeric_value(value: float) -> int | float:
    return int(value) if float(value).is_integer() else value


def _duration_unit_family(key: str, field_desc: str) -> str | None:
    desc = field_desc or ""
    lowered = f"{key} {desc}".lower()
    if "毫秒" in desc or key.endswith("_ms"):
        return "milliseconds"
    if "分钟" in desc or "minutes" in lowered:
        return "minutes"
    if "秒" in desc or key.endswith(("_secs", "_seconds")):
        return "seconds"
    return None


def _coerce_duration_string(value_raw: Any, *, key: str, field_desc: str) -> Any:
    if not isinstance(value_raw, str):
        return value_raw
    match = _DURATION_VALUE_RE.match(value_raw)
    if not match:
        return value_raw
    family = _duration_unit_family(key, field_desc)
    if family is None:
        return value_raw
    number = float(match.group("number"))
    unit = match.group("unit").lower()
    duration_ms = number * _DURATION_UNIT_TO_MS[unit]
    if family == "milliseconds":
        return _normalize_numeric_value(duration_ms)
    if family == "minutes":
        return _normalize_numeric_value(duration_ms / 60_000.0)
    return _normalize_numeric_value(duration_ms / 1000.0)


def _field_description(key: str) -> str:
    """从 Pydantic schema 提取字段说明（含单位/约束信息），帮助 LLM 感知字段语义。"""
    try:
        from core.config import Config as _Config
        schema = _Config.model_json_schema()
        defs = schema.get("$defs", {})
        parts = key.split(".")
        current_props = schema.get("properties", {})
        for i, part in enumerate(parts):
            field_schema = current_props.get(part)
            if field_schema is None:
                return ""
            if i == len(parts) - 1:
                desc = field_schema.get("description", "")
                extras: list[str] = []
                minimum = field_schema.get("minimum")
                maximum = field_schema.get("maximum")
                if minimum is not None:
                    extras.append(f"ge={minimum}")
                if maximum is not None:
                    extras.append(f"le={maximum}")
                if extras:
                    desc += f"  [{', '.join(extras)}]"
                return desc
            ref = field_schema.get("$ref", "")
            if ref:
                model_name = ref.rsplit("/", 1)[-1]
                current_props = defs.get(model_name, {}).get("properties", {})
            else:
                current_props = field_schema.get("properties", {})
    except Exception:
        pass
    return ""


@tool(ToolManifest(
    name="config.get",
    description=(
        "读取配置文件中某个键的值。支持点号嵌套路径。\n"
        "示例: loop.max_idle_gap → 返回 45\n"
        "      evolution.trigger_min_failures → 返回 3"
    ),
    progress_category="info",
    params=[ToolParam("key", "string", "配置键（支持点号路径）", required=True)],
))
async def config_get(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    key = (params.get("key") or "").strip()
    if not key:
        return ToolResult(summary="key 不能为空", error="EmptyKey", skipped=True)

    try:
        config_path = _resolve_config_path(ctx)
        cfg = _read_config(config_path)
        value = _nested_get(cfg, key)
        if value is None and "." not in key:
            return ToolResult(
                summary=f"键 '{key}' 不存在或为 null。可用: {', '.join(k for k in cfg if not k.startswith('_'))}",
                skipped=True,
            )
        return ToolResult(
            summary=f"{key} = {json.dumps(value, ensure_ascii=False)}\n(source={config_path})",
            evidence=str(value),
            metadata=tool_metadata(
                "config.get",
                f"config.get key={key}",
                key=key,
                value=value,
            ),
        )
    except Exception as e:
        return ToolResult(summary=f"读取失败: {e}", error="ConfigError")


@tool(ToolManifest(
    name="config.set",
    description=(
        "修改配置文件中的某个值。支持点号嵌套路径。修改后 loop 自动热重载。\n"
        "时间类字段除 JSON 数字外，也接受带单位的字符串，如 100ms / 0.5s / 2m。\n"
        "可调的常见参数:\n"
        "  loop.max_concurrent_ticks — tick 并发上限（同 chain 仍串行）\n"
        "  loop.max_tick_queue — tick 等待队列上限\n"
        "  loop.wake_poll_interval — 事件轮询粒度(毫秒)\n"
        "  loop.max_idle_gap — 空闲等待上限(毫秒)\n"
        "  loop.min_act_gap — 动作间隔(毫秒)\n"
        "  loop.active_idle_gap — 有活跃任务但 wait/pause 时的等待上限(毫秒)\n"
        "  evolution.enabled — 是否启用自进化\n"
        "  evolution.competitive_candidates — 竞争进化候选数(1=关闭, >=2=启用)\n"
        "  evolution.trigger_min_failures — 触发进化所需失败数\n"
        "  evolution.trigger_window_minutes — 触发时间窗(分钟)\n"
        "  evolution.error_streak_evolve — 错误连击立即触发\n"
        "  memory.working_capacity — 工作记忆容量\n"
    ),
    progress_category="mutation",
    params=[
        ToolParam("key", "string", "配置键（支持点号路径）", required=True),
        ToolParam("value", "string", "新值（JSON 格式）", required=True),
    ],
))
async def config_set(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    key = (params.get("key") or "").strip()
    value_raw = params.get("value")
    if not key:
        return ToolResult(summary="key 不能为空", error="EmptyKey", skipped=True)

    field_desc = _field_description(key)
    try:
        value = json.loads(str(value_raw)) if isinstance(value_raw, str) else value_raw
    except json.JSONDecodeError:
        value = _coerce_duration_string(value_raw, key=key, field_desc=field_desc)

    try:
        config_path = _resolve_config_path(ctx)
        cfg = _read_config(config_path)
        old = _nested_get(cfg, key)
        _nested_set(cfg, key, value)
        try:
            from core.config import Config as _Config
            validated = _Config.model_validate(cfg)
        except Exception as ve:
            hint = f"\n  字段说明: {field_desc}" if field_desc else ""
            return ToolResult(
                summary=(
                    f"❌ {key}: 值 {json.dumps(value, ensure_ascii=False)} 验证失败，未写入\n"
                    f"  原因: {ve}{hint}"
                ),
                error="ValidationError",
            )
        if not _nested_has(validated.model_dump(mode="json"), key):
            hint_parts: list[str] = []
            if field_desc:
                hint_parts.append(f"字段说明: {field_desc}")
            deprecated_hint = _deprecated_key_hint(key)
            if deprecated_hint:
                hint_parts.append(deprecated_hint)
            hint = f"\n  {' '.join(hint_parts)}" if hint_parts else ""
            return ToolResult(
                summary=(
                    f"❌ {key}: 不是运行时可识别的配置键，未写入"
                    f"\n  原因: Config 会忽略该字段，修改后不会生效{hint}"
                ),
                error="UnknownConfigKey",
            )
        _write_config(config_path, cfg)
        return ToolResult(
            summary=(
                f"✅ {key}: {json.dumps(old, ensure_ascii=False)} → {json.dumps(value, ensure_ascii=False)}"
                f"\n(source={config_path})"
            ),
            evidence=f"{key}={value}",
            metadata=tool_metadata(
                "config.set",
                f"config.set key={key}",
                key=key,
                old=old,
                new=value,
            ),
            state_delta={"config_changed": key},
        )
    except Exception as e:
        return ToolResult(summary=f"写入失败: {e}", error="ConfigError")


# ── 分组元数据（与 cli/config.py 保持一致）──────────────────────────────────
_CONFIG_GROUPS: dict[str, list[str]] = {
    "model":      ["model", "routing", "model_fallbacks", "temperature", "timeout", "thinking"],
    "providers":  ["providers"],
    "loop":       ["loop"],
    "memory":     ["memory"],
    "emotion":    ["emotion"],
    "evolution":  ["evolution"],
    "soul":       ["soul"],
    "thresholds": ["thresholds"],
    "prompts":    ["prompts"],
}


@tool(ToolManifest(
    name="config.list_keys",
    description=(
        "列出可调配置键（按分组），附带字段说明和当前默认值。\n"
        "用于在调用 config.set 前发现可写入哪些字段及其语义。\n"
        "可选 group 过滤：loop / memory / emotion / evolution / soul / model / thresholds / prompts / providers。\n"
        "不传 group 则返回所有分组。\n"
        "示例: group=loop → 返回所有 loop.* 调节键及其含义"
    ),
    progress_category="info",
    params=[
        ToolParam("group", "string", "可选 group 名（loop/memory/emotion/evolution/soul/model/thresholds/prompts/providers）", required=False),
    ],
))
async def config_list_keys(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    group = (params.get("group") or "").strip().lower() or None

    if group and group not in _CONFIG_GROUPS:
        return ToolResult(
            summary=f"未知 group '{group}'，可选：{', '.join(_CONFIG_GROUPS)}",
            error="UnknownGroup",
            skipped=True,
        )

    try:
        from core.config import Config as _Config
        schema = _Config.model_json_schema()
        defs = schema.get("$defs", {})
        top_props = schema.get("properties", {})

        def _expand_section(top_key: str) -> list[tuple[str, str, Any]]:
            """返回 [(path, description, default), ...]。"""
            field_schema = top_props.get(top_key, {})
            ref = field_schema.get("$ref", "") or (
                (field_schema.get("anyOf") or [{}])[0].get("$ref", "")
            )
            if ref:
                model_name = ref.rsplit("/", 1)[-1]
                section_props = defs.get(model_name, {}).get("properties", {})
                results = []
                for fname, fschema in section_props.items():
                    if fname.startswith("_"):
                        continue
                    path = f"{top_key}.{fname}"
                    desc = fschema.get("description", "")
                    default = fschema.get("default", None)
                    results.append((path, desc, default))
                return results
            # 简单顶层字段
            return [(top_key, field_schema.get("description", ""), field_schema.get("default", None))]

        groups_to_show = (
            {group: _CONFIG_GROUPS[group]}
            if group
            else _CONFIG_GROUPS
        )

        lines: list[str] = []
        total = 0
        for gname, top_keys in groups_to_show.items():
            entries = []
            for top_key in top_keys:
                entries.extend(_expand_section(top_key))
            total += len(entries)
            lines.append(f"\n── {gname} ({'、'.join(top_keys)}) ──")
            for path, desc, default in entries:
                default_str = f"  [default={json.dumps(default, ensure_ascii=False)}]" if default is not None else ""
                lines.append(f"  {path}{default_str}")
                if desc:
                    lines.append(f"    {desc}")

        summary = "\n".join(lines)
        return ToolResult(
            summary=f"共 {total} 个可调键（group={group or 'all'}）：\n{summary}",
            evidence=summary,
            metadata=tool_metadata(
                "config.keys",
                f"config.keys total={total} group={group or 'all'}",
                total_keys=total,
                group=group or "all",
            ),
        )
    except Exception as e:
        return ToolResult(summary=f"列举失败: {e}", error="ConfigError")
