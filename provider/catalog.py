"""provider/catalog.py — provider 模型与 run_type 映射查询."""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

# 内置模板路径（随包发布，只读种子）

BUILTIN_CATALOG_PATH: Path = Path(__file__).parent / "models.json"

# 运行时路径（兼容 fallback；主运行时应优先显式传 catalog_path）
_runtime_path: Path = BUILTIN_CATALOG_PATH
_context_window_hints: dict[str, int] = {}


def _catalog_key(catalog_path: Path | None) -> str | None:
    """将可选 catalog_path 转成缓存 key。"""
    return str(catalog_path.expanduser()) if catalog_path is not None else None


def _coerce_string_mapping(
    raw: dict[str, Any] | None,
    *,
    keep_empty: bool = False,
    normalize_key_case: bool = False,
    normalize_value_case: bool = False,
) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        normalized_key = key.strip()
        normalized_value = value.strip()
        if normalize_key_case:
            normalized_key = normalized_key.lower()
        if normalize_value_case:
            normalized_value = normalized_value.lower()
        if not keep_empty and (not normalized_key or not normalized_value):
            continue
        out[normalized_key] = normalized_value
    return out


def _merged_model(defaults: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    if not defaults:
        return model
    merged = {**defaults, **model}
    for key, value in defaults.items():
        if isinstance(value, dict) and isinstance(model.get(key), dict):
            merged[key] = {**value, **model[key]}
    return merged


@functools.lru_cache(maxsize=8)
def _load(catalog_path: str | None = None) -> dict[str, dict[str, Any]]:
    """读取 catalog 文件（显式路径优先，回退兼容运行时路径/内置模板）。"""
    requested = Path(catalog_path).expanduser() if catalog_path else _runtime_path
    src = requested if requested.exists() else BUILTIN_CATALOG_PATH
    raw: dict[str, Any] = json.loads(src.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if isinstance(v, dict)}


def get_run_type_routing(*, catalog_path: Path | None = None) -> dict[str, str]:
    """返回 run_type → 模型档位 映射（来自 models.json 顶层 run_type_routing 段）。

    "_doc" 键及非字符串值会被过滤掉。调用方可将返回值与自身默认值合并使用。
    """
    routing = _load(_catalog_key(catalog_path)).get("run_type_routing", {})
    sanitized = _coerce_string_mapping(
        {k: v for k, v in routing.items() if k != "_doc"},
        normalize_key_case=True,
        normalize_value_case=True,
    )
    sanitized.pop("_doc", None)
    return sanitized


def lookup_model(model_id: str, *, catalog_path: Path | None = None) -> dict[str, Any] | None:
    """在所有 provider 里查找 model_id，返回模型元数据字典；未找到返回 None。"""
    for provider_name in list_providers(catalog_path=catalog_path):
        for m in list_provider_models(provider_name, catalog_path=catalog_path):
            if m.get("id") == model_id:
                return m
    return None


def lookup_model_ref(model_ref: str, *, catalog_path: Path | None = None) -> dict[str, Any] | None:
    """通过 provider/model-id 形式查模型；provider 不匹配时返回 None。"""
    provider_name, _, model_id = model_ref.partition("/")
    if not provider_name or not model_id:
        return None
    for m in list_provider_models(provider_name, catalog_path=catalog_path):
        if m.get("id") == model_id:
            return m
    return None


def model_supports(
    model_ref_or_id: str,
    *,
    capability: str | None = None,
    input_modality: str | None = None,
    catalog_path: Path | None = None,
) -> bool:
    """判断模型是否具备指定 capability / 输入模态。"""
    spec = (
        lookup_model_ref(model_ref_or_id, catalog_path=catalog_path)
        if "/" in model_ref_or_id
        else lookup_model(model_ref_or_id, catalog_path=catalog_path)
    )
    if not spec:
        return False
    if capability:
        caps = spec.get("capabilities")
        if capability not in caps if isinstance(caps, list) else True:
            return False
    if input_modality:
        inputs = spec.get("input")
        if input_modality not in inputs if isinstance(inputs, list) else True:
            return False
    return True


def find_model_ref_for_capability(
    *,
    capability: str | None = None,
    input_modality: str | None = None,
    preferred_provider: str | None = None,
    catalog_path: Path | None = None,
) -> str | None:
    """按 provider/model-id 返回首个满足能力要求的模型。

    选择顺序：优先当前 provider，其次其它 provider；每个 provider 内保持 models.json 的声明顺序。
    """
    catalog = _load(_catalog_key(catalog_path))
    provider_names = [name for name in catalog if "models" in catalog[name]]
    if preferred_provider and preferred_provider in provider_names:
        provider_names = [preferred_provider] + [name for name in provider_names if name != preferred_provider]

    for provider_name in provider_names:
        for spec in list_provider_models(provider_name, catalog_path=catalog_path):
            model_ref = f"{provider_name}/{spec.get('id', '')}"
            if not model_ref.endswith("/") and model_supports(
                model_ref,
                capability=capability,
                input_modality=input_modality,
                catalog_path=catalog_path,
            ):
                return model_ref
    return None


def list_providers(*, catalog_path: Path | None = None) -> list[str]:
    """返回 models.json 中所有 provider 名称（过滤掉 _doc 等非 provider 键）。"""
    return [k for k, v in _load(_catalog_key(catalog_path)).items() if "models" in v]


def list_provider_models(provider_name: str, *, catalog_path: Path | None = None) -> list[dict[str, Any]]:
    """返回指定 provider 的模型列表； provider 不存在时返回空列表。"""
    provider_data = _load(_catalog_key(catalog_path)).get(provider_name, {})
    models = provider_data.get("models", [])
    if not isinstance(models, list):
        return []
    defaults = provider_data.get("model_defaults", {})
    if not isinstance(defaults, dict):
        return [m for m in models if isinstance(m, dict)]
    return [_merged_model(defaults, m) for m in models if isinstance(m, dict)]


def is_reasoning_model(model_id: str, *, catalog_path: Path | None = None) -> bool:
    """返回该模型是否具备推理能力（reasoning=true 元数据标记）。

    copilot 的 gpt-5.4 / o3 等模型由顶层 thinking 自动映射到 reasoning_effort，
    bailian 的 Qwen3 模型通过 thinking.budget_tokens 控制。
    两套机制不同，此函数仅供展示/路由层判断，不影响 payload 构造逻辑。
    """
    m = lookup_model(model_id, catalog_path=catalog_path)
    return bool(m and m.get("reasoning"))


def resolve_context_window(model_id: str, override: int | None, *, catalog_path: Path | None = None) -> int | None:
    """返回模型的上下文窗口大小（tokens）。

    优先级：
      1. lingzhou.json 中的 context_window_tokens（显式覆盖）
      2. models.json 内置目录中的 context_window
      3. None（由调用方决定 fallback）
    """
    if override is not None:
        return override
    hinted = _context_window_hints.get(model_id)
    m = lookup_model(model_id, catalog_path=catalog_path)
    if m:
        context_window = m.get("context_window")
        if isinstance(context_window, int) and context_window > 0 and hinted is not None:
            return min(context_window, hinted)
        if isinstance(context_window, int) and context_window > 0:
            return context_window
    if hinted is not None:
        return hinted
    return None


def set_context_window_hint(model_id: str, context_window: int) -> None:
    """记录运行期观测到的更严格 context_window 上限。

    该 hint 仅在当前进程生效，用于在线收敛预算，避免后续请求重复触发超限。
    """
    if context_window <= 0:
        return
    prev = _context_window_hints.get(model_id)
    if prev is None or context_window < prev:
        _context_window_hints[model_id] = context_window
