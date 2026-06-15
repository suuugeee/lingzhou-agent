"""provider/capabilities.py — 模型能力查询（tools 层入口，不直接依赖 catalog 细节）。"""
from __future__ import annotations

from typing import TYPE_CHECKING

from provider.catalog import find_model_ref_for_capability, model_supports

if TYPE_CHECKING:
    from core.config import Config


def resolve_model_ref_for_input(
    cfg: Config,
    *,
    capability: str,
    input_modality: str,
    current_model_ref: str | None = None,
) -> str:
    """返回满足 capability + input_modality 的 model_ref；无候选时抛 RuntimeError。"""
    configured_vision_ref = str(getattr(cfg, "vision_model", "") or "").strip()
    configured_provider = configured_vision_ref.partition("/")[0]
    use_configured_vision = (
        current_model_ref is None
        and configured_vision_ref
        and configured_provider in getattr(cfg, "providers", {})
        and capability == "vision"
        and input_modality == "image"
    )
    if use_configured_vision:
        model_ref = configured_vision_ref
    else:
        model_ref = (current_model_ref or cfg.model).strip()
    if model_supports(model_ref, capability=capability, input_modality=input_modality):
        return model_ref

    if use_configured_vision:
        raise RuntimeError(
            f"配置的 vision_model {model_ref} 不支持 {input_modality} 输入或缺少 {capability} 能力"
        )

    fallback_ref = find_model_ref_for_capability(
        capability=capability,
        input_modality=input_modality,
        preferred_provider=cfg.active_provider_name,
    )
    if fallback_ref:
        return fallback_ref

    raise RuntimeError(
        f"当前模型 {model_ref} 不支持 {input_modality} 输入，"
        f"且运行时目录中未找到具备 {capability} 能力的候选模型"
    )
