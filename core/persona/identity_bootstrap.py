"""core/persona/identity_bootstrap.py — 身份启动管理器。

职责（严格限定于身份文件与启动注入）：
  - workspace_dir 下 SOUL.md / IDENTITY.md / BOOTSTRAP.md 等身份文件的首次写入
  - bootstrap：将所有身份文件注入 WM（启动身份注入机制）

不负责：
  - models.json 生成 → 由 provider.models_gen.ensure_models_json() 在启动时处理

原位置：CognitionLoop._build_soul_content / _init_soul_files / _sync_soul_md / _bootstrap
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from core.workspace.defaults import (
    BOOTSTRAP_MD,
    CONSTITUTION_MD,
    HEARTBEAT_MD,
    IDENTITY_MD,
    MEMORY_MD,
    TOOLS_MD,
    USER_MD,
)
from core.workspace.state import (
    BootstrapMode,
    _now_iso,
    bootstrap_status,
    read_workspace_state,
    reconcile_bootstrap_completion,
    resolve_bootstrap_mode,
    write_workspace_state,
)

if TYPE_CHECKING:
    from core.config import Config
    from core.judgment import JudgmentLayer
    from memory.working import WorkingMemory
    from store.task import TaskStore

_log = logging.getLogger("lingzhou.soul")

# 工作区默认文件列表（名称 → 模板内容），按写入顺序
_WORKSPACE_FILES: list[tuple[str, str]] = [
    ("IDENTITY.md",     IDENTITY_MD),
    ("BOOTSTRAP.md",    BOOTSTRAP_MD),
    ("USER.md",         USER_MD),
    ("TOOLS.md",        TOOLS_MD),
    ("HEARTBEAT.md",    HEARTBEAT_MD),
    ("MEMORY.md",       MEMORY_MD),
    ("CONSTITUTION.md", CONSTITUTION_MD),  # 宪法器官（A3）：首次初始化写入，后续不覆盖
]

# 冷启动时注入 WM 的文件顺序（越靠前越优先被 LLM 读到）
_BOOTSTRAP_FILES = ("BOOTSTRAP.md", "IDENTITY.md", "SOUL.md", "USER.md", "TOOLS.md", "HEARTBEAT.md", "MEMORY.md")
# DREAMS.md 单独注入：给 LLM 看自己的长期志向演化记录（不进 system prompt 前缀，只进 WM）
_DREAMS_FILE = "DREAMS.md"


def _bootstrap_file_should_be_written(workspace: Any) -> bool:
    state = read_workspace_state(workspace)
    if state.setup_completed_at:
        return False
    if not state.bootstrap_seeded_at:
        state.bootstrap_seeded_at = _now_iso()
        write_workspace_state(workspace, state)
        _log.debug("[workspace_state] bootstrapSeededAt 已写入")
    return True


def _render_workspace_file(content: str, soul_name: str) -> str:
    return content.replace("{name}", soul_name)


def _read_existing_text(path: Any, *, strip: bool = False) -> str | None:
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return None
    return content.strip() if strip else content


def _bootstrap_identity_content(fname: str, content: str) -> str:
    return f"[{fname}]\n{content}"


def _should_add_identity_prefix(fname: str, mode: BootstrapMode) -> bool:
    return fname == "IDENTITY.md" or (mode == "full" and fname == "BOOTSTRAP.md")


def _is_legacy_heartbeat_signal(signal: dict[str, Any]) -> bool:
    return signal.get("payload", {}).get("source") == "heartbeat"


class IdentityBootstrapManager:
    """管理身份文件，并在冷启动时将身份材料注入 WM。

    人格层委托给 PersonaEngine；SOUL.md 镜像委托给 SoulEngine。
    """

    def __init__(
        self,
        cfg: Config,
        task_store: TaskStore,
        wm: WorkingMemory,
    ) -> None:
        from core.persona import PersonaEngine
        from core.soul import SoulEngine
        self._cfg = cfg
        self._task_store = task_store
        self._wm = wm
        self._persona = PersonaEngine(cfg, task_store)
        self._soul_engine = SoulEngine(cfg, self._persona)

    async def init_files(self) -> None:
        """冷启动：确保 workspace_dir 中有所有必要的 Soul 文件。

        优先级：facts DB（持久化 EMA 值）> cfg.soul 配置默认值（全新启动）。
        文件一旦写入后归用户/agent 管理，不会被再次覆盖。
        """
        workspace = self._cfg.workspace_dir
        workspace.mkdir(parents=True, exist_ok=True)

        from core.skill import seed_workspace_skills

        seed_workspace_skills(workspace)

        soul_name = await self._persona.soul_name()

        await self._soul_engine.init_md()

        for fname, content in _WORKSPACE_FILES:
            fpath = workspace / fname
            if not fpath.exists():
                # BOOTSTRAP.md 在 bootstrap 完成后不应被重建：
                # init_files 重建会使 reconcile_bootstrap_completion 永远感知不到"已删除"，
                # 导致每次启动都重新进入 full bootstrap 模式。
                if fname == "BOOTSTRAP.md" and not _bootstrap_file_should_be_written(workspace):
                    continue  # bootstrap 已完成，跳过重建
                fpath.write_text(_render_workspace_file(content, soul_name), encoding="utf-8")
                _log.info("%s 初始化: 已写入 %s", fname, fpath)

    async def sync_md(self) -> None:
        """将 facts DB 中最新 EMA ethos 值同步写回 SOUL.md（人类可读镜像）。

        委托给 SoulEngine.sync_md()。
        """
        await self._soul_engine.sync_md()

    async def bootstrap(
        self,
        judgment: JudgmentLayer | None = None,
        run_kind: Literal["interactive", "heartbeat", "cron"] = "interactive",
    ) -> BootstrapMode:
        """冷启动：Soul 文件初始化 + WM 身份注入 + system prompt 前缀注入。

        对齐 OpenClaw bootstrap 三模式机制：
        - "full"  : bootstrap 待完成（BOOTSTRAP.md 存在）且交互式运行
                   → BOOTSTRAP.md + 所有身份文件注入 system prompt 和 WM
        - "limited": bootstrap 待完成但非交互式（预留）
        - "none"  : bootstrap 已完成（BOOTSTRAP.md 已删除或 setupCompletedAt 已写入）
                   → 跳过 BOOTSTRAP.md，仅注入其余身份文件

        返回本次计算的 BootstrapMode，供 CognitionLoop 存储在 _bootstrap_mode 上。
        """
        from memory.working import WMItem

        await self.init_files()
        workspace = self._cfg.workspace_dir

        # ① 检测上次 run 末尾是否已删除 BOOTSTRAP.md → 持久化 setupCompletedAt
        state = reconcile_bootstrap_completion(workspace)

        # ② 计算本次 bootstrap 模式
        pending = bootstrap_status(workspace, state) == "pending"
        mode = resolve_bootstrap_mode(pending, run_kind)
        _log.info("[boot] bootstrap_mode=%s (pending=%s run_kind=%s)", mode, pending, run_kind)

        injected: list[str] = []
        identity_parts: list[str] = []
        for fname in _BOOTSTRAP_FILES:
            # "none" 模式：跳过 BOOTSTRAP.md（已完成初始化，不再注入启动指令）
            if mode == "none" and fname == "BOOTSTRAP.md":
                continue
            fpath = workspace / fname
            content = _read_existing_text(fpath)
            if content is None:
                continue
            wrapped_content = _bootstrap_identity_content(fname, content)
            self._wm.add(WMItem(
                kind="bootstrap_identity",
                content=wrapped_content,
                priority=self._cfg.thresholds.wm_pri_identity,
            ))
            injected.append(fname)
            # 核心身份文件 → system prompt 前缀（永久，不随 WM 驱逐）
            # "none" 模式只保留 IDENTITY.md；"full" 模式保留两者
            if _should_add_identity_prefix(fname, mode):
                identity_parts.append(wrapped_content)
        if injected:
            _log.info("[boot] 身份注入: %s", " ".join(injected))
        if judgment is not None and identity_parts:
            judgment.set_identity_prefix("\n\n".join(identity_parts))

        # DREAMS.md：长期志向，进 WM 供 LLM 感知自己的成长轨迹（priority 略低于身份文件）
        dreams_path = workspace / _DREAMS_FILE
        dreams_content = _read_existing_text(dreams_path, strip=True)
        if dreams_content:
            self._wm.add(WMItem(
                kind="bootstrap_identity",
                content=_bootstrap_identity_content(_DREAMS_FILE, dreams_content),
                priority=self._cfg.thresholds.wm_pri_identity * 0.9,
            ))

        # 清理旧版 heartbeat cron 信号（旧实现将 heartbeat 存入 signals 表；
        # 新版改为 monotonic 时间戳机制，DB 中遗留的 source=heartbeat 条目应移除）
        old_hb = [
            s for s in await self._task_store.list_signals(limit=200)
            if _is_legacy_heartbeat_signal(s)
        ]
        for s in old_hb:
            await self._task_store.cancel_signal(s["id"])

        return mode

    async def refresh_identity(self, judgment: JudgmentLayer | None = None) -> None:
        """重读身份文件，更新 system prompt 前缀。

        在 evolution 运行后调用：evolution 可能已通过 file.write 修改
        BOOTSTRAP.md / IDENTITY.md，身份前缀需要在当次会话内生效，
        不应等到下次重启。只更新 system prompt 前缀，不重复注入 WM。
        """
        if judgment is None:
            return
        workspace = self._cfg.workspace_dir
        identity_parts: list[str] = []
        for fname in ("BOOTSTRAP.md", "IDENTITY.md"):
            fpath = workspace / fname
            content = _read_existing_text(fpath)
            if content is not None:
                identity_parts.append(_bootstrap_identity_content(fname, content))
        if identity_parts:
            judgment.set_identity_prefix("\n\n".join(identity_parts))
            _log.debug("[soul] 身份前缀已刷新（evolution 后更新）")
