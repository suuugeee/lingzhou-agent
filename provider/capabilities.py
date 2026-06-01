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
    model_ref = (current_model_ref or cfg.model).strip()
    if model_supports(model_ref, capability=capability, input_modality=input_modality):
        return model_ref

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
