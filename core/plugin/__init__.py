"""core/plugin.py — 灵舟插件系统。

面向灵舟运行时的轻量插件管理器。

插件结构:
  plugins/<name>/
    plugin.json      — 元数据
    __init__.py      — 入口（导出 register/unregister 函数）

生命周期:
  discover → validate → register → start → [运行] → stop → unregister
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

_log = logging.getLogger("lingzhou.plugin")


@dataclass
class PluginManifest:
    """插件元数据。"""
    id: str
    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    dependencies: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    configSchema: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class PluginInstance:
    manifest: PluginManifest
    path: Path
    module: Any = None
    state: str = "discovered"  # discovered → loaded → registered → running → stopped

    def __repr__(self) -> str:
        return f"Plugin({self.manifest.id} v{self.manifest.version} [{self.state}])"


class PluginManager:
    """插件管理器：发现、加载、注册、启停。"""

    def __init__(self, plugins_dir: str | Path = "plugins"):
        self._dir = Path(plugins_dir)
        self._plugins: dict[str, PluginInstance] = {}
        self._hooks: dict[str, list[Callable]] = {
            "on_start": [],
            "on_stop": [],
            "on_tick": [],
        }

    # ── 发现 ────────────────────────────────────────────────────────────────

    def discover(self) -> list[PluginManifest]:
        """扫描 plugins/ 目录，返回有效的插件清单。"""
        manifests: list[PluginManifest] = []
        if not self._dir.exists():
            return manifests

        for item in sorted(self._dir.iterdir()):
            if not item.is_dir():
                continue
            manifest_path = item / "plugin.json"
            if not manifest_path.exists():
                continue

            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest = PluginManifest(
                    id=data.get("id", item.name),
                    name=data.get("name", item.name),
                    version=data.get("version", "0.1.0"),
                    description=data.get("description", ""),
                    author=data.get("author", ""),
                    dependencies=data.get("dependencies", []),
                    channels=data.get("channels", []),
                    tools=data.get("tools", []),
                    configSchema=data.get("configSchema", {}),
                    enabled=data.get("enabled", True),
                )
                instance = PluginInstance(manifest=manifest, path=item)
                self._plugins[manifest.id] = instance
                manifests.append(manifest)
                _log.info("[plugin] 发现: %s v%s", manifest.id, manifest.version)
            except Exception as e:
                _log.warning("[plugin] 解析失败 %s: %s", manifest_path, e)

        return manifests

    # ── 加载 ────────────────────────────────────────────────────────────────

    def load(self, plugin_id: str) -> bool:
        """加载插件模块。"""
        instance = self._plugins.get(plugin_id)
        if not instance:
            _log.warning("[plugin] 未发现: %s", plugin_id)
            return False

        try:
            init_file = instance.path / "__init__.py"
            if not init_file.exists():
                _log.warning("[plugin] 无 __init__.py: %s", plugin_id)
                return False

            spec = importlib.util.spec_from_file_location(
                f"lingzhou_plugin_{plugin_id}", init_file
            )
            if not spec or not spec.loader:
                return False

            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)

            instance.module = mod
            instance.state = "loaded"
            _log.info("[plugin] 已加载: %s", plugin_id)
            return True
        except Exception as e:
            _log.error("[plugin] 加载失败 %s: %s", plugin_id, e)
            return False

    def load_all(self) -> dict[str, bool]:
        """加载所有已发现的插件。"""
        return {pid: self.load(pid) for pid in self._plugins}

    # ── 注册 ────────────────────────────────────────────────────────────────

    def register(self, plugin_id: str, tool_registry=None, channel_registry=None) -> bool:
        """注册插件：调用插件的 register() 函数，注入工具和通道。"""
        instance = self._plugins.get(plugin_id)
        if not instance or not instance.module:
            return False

        try:
            register_fn = getattr(instance.module, "register", None)
            if register_fn:
                ctx = PluginContext(
                    manifest=instance.manifest,
                    tool_registry=tool_registry,
                    channel_registry=channel_registry,
                )
                register_fn(ctx)
            instance.state = "registered"
            _log.info("[plugin] 已注册: %s", plugin_id)
            return True
        except Exception as e:
            _log.error("[plugin] 注册失败 %s: %s", plugin_id, e)
            return False

    def register_all(self, tool_registry=None, channel_registry=None) -> dict[str, bool]:
        return {pid: self.register(pid, tool_registry, channel_registry) for pid in self._plugins}

    # ── 启停 ────────────────────────────────────────────────────────────────

    def start(self, plugin_id: str) -> bool:
        instance = self._plugins.get(plugin_id)
        if not instance or not instance.module:
            return False
        try:
            start_fn = getattr(instance.module, "start", None)
            if start_fn:
                start_fn()
            instance.state = "running"
            _log.info("[plugin] 已启动: %s", plugin_id)
            return True
        except Exception as e:
            _log.error("[plugin] 启动失败 %s: %s", plugin_id, e)
            return False

    def stop(self, plugin_id: str) -> bool:
        instance = self._plugins.get(plugin_id)
        if not instance or not instance.module:
            return False
        try:
            stop_fn = getattr(instance.module, "stop", None)
            if stop_fn:
                stop_fn()
            unregister_fn = getattr(instance.module, "unregister", None)
            if unregister_fn:
                unregister_fn()
            instance.state = "stopped"
            _log.info("[plugin] 已停止: %s", plugin_id)
            return True
        except Exception as e:
            _log.error("[plugin] 停止失败 %s: %s", plugin_id, e)
            return False

    def start_all(self) -> dict[str, bool]:
        return {pid: self.start(pid) for pid in self._plugins}

    def stop_all(self) -> dict[str, bool]:
        return {pid: self.stop(pid) for pid in self._plugins}

    # ── 查询 ────────────────────────────────────────────────────────────────

    def list_plugins(self) -> list[dict]:
        return [
            {
                "id": p.manifest.id,
                "name": p.manifest.name,
                "version": p.manifest.version,
                "state": p.state,
                "description": p.manifest.description,
            }
            for p in self._plugins.values()
        ]

    def get(self, plugin_id: str) -> PluginInstance | None:
        return self._plugins.get(plugin_id)


@dataclass
class PluginContext:
    """传递给插件 register() 的上下文。"""
    manifest: PluginManifest
    tool_registry: Any = None
    channel_registry: Any = None
