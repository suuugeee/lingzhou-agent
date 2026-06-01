"""provider/__init__.py — Provider 工厂。

新增 wire protocol：
  1. 在 provider/ 下新建实现文件（如 anthropic.py），实现 base.Provider Protocol。
  2. 在下方 match 里加一行 case "xxx": return XxxProvider(cfg)。
  其余代码零改动。

已支持的 type：
  "openai_compat" — OpenAI 兼容 API（百炼/DeepSeek/标准 OpenAI/GitHub Copilot）
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config import Config
    from provider.base import Provider


def create_provider(cfg: Config) -> Provider:
    provider_def = cfg.active_provider
    match provider_def.type:
        case "openai_compat":
            from provider.openai_compat import OpenAICompatProvider
            return OpenAICompatProvider(cfg)
        case _:
            raise ValueError(
                f"未知 provider 类型: {provider_def.type!r}。\n"
                f"已支持: openai_compat。\n"
                f"新增协议请参考 provider/__init__.py 顶部注释。"
            )


def create_provider_with_model(cfg: Config, model_ref: str) -> Provider:
    """用指定 model_ref 替换 cfg.model 创建 provider（routing 路由用）。"""
    routing_cfg = cfg.model_copy(update={"model": model_ref})
    routing_cfg._base_dir = cfg._base_dir
    return create_provider(routing_cfg)
