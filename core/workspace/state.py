"""core/workspace_state.py — 工作区 bootstrap 状态持久化。

对齐 OpenClaw workspace-state.json 机制：
  - bootstrapSeededAt : BOOTSTRAP.md 首次写入时间 (ISO 8601)
  - setupCompletedAt  : bootstrap 流程完成时间 (ISO 8601)

状态文件路径: <workspace_dir>/.lingzhou-state.json

Bootstrap 状态机：
  pending  → BOOTSTRAP.md 存在且 setupCompletedAt 未写入
  complete → setupCompletedAt 已写入 OR BOOTSTRAP.md 不存在

Bootstrap 注入模式：
  "full"    — bootstrap 待完成 + 交互式运行；完整注入 BOOTSTRAP.md 到 system prompt
  "limited" — bootstrap 待完成但本次不适合执行（预留，当前未启用）
  "none"    — bootstrap 已完成或后台任务；不注入 BOOTSTRAP.md
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

_log = logging.getLogger("lingzhou.workspace_state")

_STATE_FILENAME = ".lingzhou-state.json"

BootstrapMode = Literal["full", "limited", "none"]


@dataclass
class WorkspaceState:
    bootstrap_seeded_at: str | None = None  # BOOTSTRAP.md 首次写入时间
    setup_completed_at: str | None = None   # bootstrap 完成时间


def _state_path(workspace_dir: Path) -> Path:
    return workspace_dir / _STATE_FILENAME


def _bootstrap_path(workspace_dir: Path) -> Path:
    return workspace_dir / "BOOTSTRAP.md"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_from_json(raw: dict[str, object]) -> WorkspaceState:
    return WorkspaceState(
        bootstrap_seeded_at=raw.get("bootstrapSeededAt") or None,
        setup_completed_at=raw.get("setupCompletedAt") or None,
    )


def _state_to_json(state: WorkspaceState) -> dict[str, str]:
    data: dict[str, str] = {}
    if state.bootstrap_seeded_at:
        data["bootstrapSeededAt"] = state.bootstrap_seeded_at
    if state.setup_completed_at:
        data["setupCompletedAt"] = state.setup_completed_at
    return data


def read_workspace_state(workspace_dir: Path) -> WorkspaceState:
    """从 .lingzhou-state.json 读取工作区状态；文件缺失或解析失败返回空状态。"""
    path = _state_path(workspace_dir)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return _state_from_json(raw)
    except Exception:
        return WorkspaceState()


def write_workspace_state(workspace_dir: Path, state: WorkspaceState) -> None:
    """将工作区状态写入 .lingzhou-state.json。"""
    path = _state_path(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_state_to_json(state), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bootstrap_status(
    workspace_dir: Path,
    state: WorkspaceState,
) -> Literal["pending", "complete"]:
    """判断 bootstrap 是否已完成。

    complete 条件（任一满足）：
      1. setupCompletedAt 已写入（持久完成标志）
      2. BOOTSTRAP.md 文件不存在（文件即标志，缺失 = 完成）
    """
    if state.setup_completed_at:
        return "complete"
    if not _bootstrap_path(workspace_dir).exists():
        return "complete"
    return "pending"


def resolve_bootstrap_mode(
    bootstrap_pending: bool,
    run_kind: Literal["interactive", "heartbeat", "cron"] = "interactive",
) -> BootstrapMode:
    """决定本次启动的 bootstrap 注入模式。

    - none    : bootstrap 已完成，或后台任务；不注入 BOOTSTRAP.md
    - full    : bootstrap 待完成 + 交互式运行；完整注入 BOOTSTRAP.md
    - limited : bootstrap 待完成但本次运行不适合执行初始化（预留）
    """
    if not bootstrap_pending:
        return "none"
    if run_kind in ("heartbeat", "cron"):
        return "none"
    return "full"


def reconcile_bootstrap_completion(workspace_dir: Path) -> WorkspaceState:
    """检测 BOOTSTRAP.md 是否已被删除，若是则写入 setupCompletedAt。

    调用时机：
      1. 每次 run() 启动时（检测上次 run 末尾是否删除了 BOOTSTRAP.md）
      2. 每个 tick 结束后（in-session 即时感知）

    对齐 OpenClaw reconcileWorkspaceBootstrapCompletion 机制。
    """
    state = read_workspace_state(workspace_dir)
    bootstrap_path = _bootstrap_path(workspace_dir)

    # 已有完成标志，无需重复处理
    if state.setup_completed_at:
        return state

    # bootstrapSeededAt 已记录（曾写入过 BOOTSTRAP.md）且文件已消失 → 标记完成
    if state.bootstrap_seeded_at and not bootstrap_path.exists():
        now = _now_iso()
        completed_state = WorkspaceState(
            bootstrap_seeded_at=state.bootstrap_seeded_at,
            setup_completed_at=now,
        )
        write_workspace_state(workspace_dir, completed_state)
        _log.info(
            "[workspace_state] BOOTSTRAP.md 已删除，标记 setupCompletedAt=%s", now
        )
        return completed_state

    return state
