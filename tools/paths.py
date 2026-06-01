"""tools/paths.py — 工具层共享路径解析（workspace / skills 等）。"""
from __future__ import annotations

from pathlib import Path

from tools.registry import ToolContext


def workspace_dir_from_ctx(ctx: ToolContext) -> Path | None:
    """解析 workspace 根目录。

    生产 ``Config`` 使用 ``workspace_dir`` 属性（含相对路径 resolve）；
    测试 fixture 仅有 ``loop.workspace_dir`` 时回退 expanduser。
    """
    cfg = ctx.config
    workspace = getattr(cfg, "workspace_dir", None)
    if workspace is not None:
        return Path(workspace)
    raw = getattr(getattr(cfg, "loop", None), "workspace_dir", "")
    if not raw:
        return None
    try:
        return Path(str(raw)).expanduser()
    except (TypeError, ValueError):
        return None


def skills_dir_from_ctx(ctx: ToolContext) -> Path:
    workspace = workspace_dir_from_ctx(ctx)
    if workspace is None:
        raise ValueError("workspace_dir 未配置，无法加载 skills")
    return workspace / "skills"
