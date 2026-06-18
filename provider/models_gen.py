"""provider/models_gen.py — 运行时 models.json 生成器。

设计思路：

  当前实现：
    - 内置目录（provider/models.json）存储模型元数据（context_window / max_tokens /
      thinking），随包发布，仅用作种子数据。
    - 运行时目录（workspace_dir/models.json）由本模块在每次启动时生成：
        * 以内置目录为基础（保留所有模型元数据）
        * 用 lingzhou.json providers 中的连接参数（base_url / mode / api_key_env）覆盖
        * 用户在 lingzhou.json providers.<name>.models 中自定义的模型追加合并
        * 嵌入 _fingerprint 字段，下次启动时比对，未变化则 skip
    - 三态（skip / noop / write）：
        skip  — 指纹命中内存缓存，直接返回
        noop  — 指纹未缓存但文件内容与生成结果一致，仅更新缓存
        write — 内容不同，写入文件
    - 只负责生成 workspace_dir/models.json；catalog 消费方自行显式传路径。

调用点：
    CognitionLoop.run() / CognitionLoop.open() 在 task_store.open() 之后、
    soul.bootstrap() 之前调用 ensure_models_json(cfg)。
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pathlib import Path

    from core.config import Config

from provider import catalog as _catalog

_log = logging.getLogger("lingzhou.models_gen")

# 内存指纹缓存：str(workspace_models_path) → fingerprint
_READY_CACHE: dict[str, str] = {}
_READY_CACHE_MAX = 32

# 从 ProviderDefinition 中写入 models.json 的连接字段（过滤掉敏感字段如 api_key）
_PROVIDER_CATALOG_FIELDS = ("base_url", "mode", "type", "api_key_env")


def _remember_ready_fingerprint(cache_key: str, fingerprint: str) -> None:
    _READY_CACHE.pop(cache_key, None)
    _READY_CACHE[cache_key] = fingerprint
    while len(_READY_CACHE) > _READY_CACHE_MAX:
        _READY_CACHE.pop(next(iter(_READY_CACHE)))


def _deep_merge_dict(target: dict[str, Any], patch: dict[str, Any]) -> None:
    """将 patch 递归合并进 target，标量与数组直接覆盖。"""
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge_dict(cast("dict[str, Any]", target[key]), value)
            continue
        target[key] = value


def _stable_json(value: Any) -> str:
    """确定性 JSON 序列化（键递归排序），用于 fingerprint 计算。"""
    if value is None or not isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        items = cast("list[Any]", value)
        return "[" + ",".join(_stable_json(item) for item in items) + "]"
    obj = cast("dict[str, Any]", value)
    entries: list[tuple[str, Any]] = sorted(obj.items())
    return "{" + ",".join(f"{json.dumps(k)}:{_stable_json(v)}" for k, v in entries) + "}"


def _compute_fingerprint(providers_cfg: dict[str, Any], builtin_bytes: bytes) -> str:
    """fingerprint = SHA-256(stable_json(providers_cfg) + builtin_catalog_bytes)。

    providers_cfg 包含 lingzhou.json providers 各字段（不含 api_key 等敏感值）。
    builtin_bytes 为 provider/models.json 原始字节，内置目录变化时触发重新生成。
    """
    h = hashlib.sha256()
    h.update(_stable_json(providers_cfg).encode())
    h.update(b"\x00")  # 分隔符，防止 hash 碰撞
    h.update(builtin_bytes)
    return h.hexdigest()


def _copy_provider_entry(pdata: dict[str, Any]) -> dict[str, Any]:
    entry: dict[str, Any] = {k: v for k, v in pdata.items() if k != "models"}
    if "models" in pdata:
        entry["models"] = [dict(m) for m in pdata["models"]]
    return entry


def _merge_custom_models(entry: dict[str, Any], models: Any) -> None:
    existing_by_id: dict[str, dict[str, Any]] = {
        str(m.get("id", "")): m for m in entry.get("models", []) if isinstance(m, dict) and m.get("id")
    }
    for custom in models:
        if not isinstance(custom, dict):
            continue
        model_id = str(custom.get("id", "")).strip()
        if not model_id:
            continue
        if model_id in existing_by_id:
            _deep_merge_dict(existing_by_id[model_id], dict(custom))
            continue
        new_entry = dict(custom)
        entry.setdefault("models", []).append(new_entry)
        existing_by_id[model_id] = new_entry


def _merge(providers_cfg: dict[str, Any], builtin: dict[str, Any]) -> dict[str, Any]:
    """生成运行时 models.json 内容（不含 _fingerprint / _doc，由调用方注入）。

    合并规则：
      1. 以内置目录所有非 "_" 前缀的 provider 条目为基础（模型元数据来源）。
      2. 对 lingzhou.json 中出现的 provider，用 config 值覆盖连接参数
         （base_url / mode / type / api_key_env）。
      3. lingzhou.json provider 中若存在 "models" 列表，将内置目录中没有的条目追加。
      4. lingzhou.json 中新增（内置目录中不存在）的 provider，创建新条目。
    """
    out: dict[str, Any] = {}

    # 步骤 1：复制内置目录（深复制，避免修改 lru_cache 缓存对象）
    for pname, pdata in builtin.items():
        if pname.startswith("_"):
            continue
        out[pname] = _copy_provider_entry(pdata)

    # 步骤 2-4：用 lingzhou.json config 覆盖/补充
    for pname, pcfg in providers_cfg.items():
        if pname not in out:
            out[pname] = {}
        entry = out[pname]

        # 覆盖连接参数（catalog-relevant fields only）
        for field in _PROVIDER_CATALOG_FIELDS:
            if field in pcfg:
                entry[field] = pcfg[field]

        # 合并用户模型元数据：同 id 覆盖内置字段；新 id 追加。
        if "models" in pcfg:
            _merge_custom_models(entry, pcfg["models"])

    return out


@dataclass(frozen=True)
class EnsureResult:
    """ensure_models_json 的返回值。"""

    wrote: bool    # True=写入了新文件或更新了文件，False=skip/noop
    path: Path     # workspace_dir/models.json 的实际路径


async def ensure_models_json(cfg: Config) -> EnsureResult:
    """确保 workspace_dir/models.json 是基于当前 config 生成的最新版本。

        三态行为（skip/noop/write）：
            skip  — 指纹命中内存缓存，直接返回
            noop  — 指纹未缓存，但生成内容与磁盘文件一致，仅更新缓存
      write — 生成内容与磁盘不同（或文件不存在），写入文件
    """
    workspace = cfg.workspace_dir
    workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / "models.json"

    # 读取内置目录原始字节（lru_cache 不跨进程，直接读文件）
    builtin_bytes = _catalog.BUILTIN_CATALOG_PATH.read_bytes()
    builtin: dict[str, Any] = json.loads(builtin_bytes)

    # 构造用于指纹计算的 providers 视图（只含 catalog-relevant 字段）
    providers_view: dict[str, Any] = {}
    for pname, pdef in cfg.providers.items():
        provider_view = {f: v for f, v in pdef.model_dump().items() if f in _PROVIDER_CATALOG_FIELDS}
        if pdef.models:
            provider_view["models"] = pdef.models
        providers_view[pname] = provider_view
    fp = _compute_fingerprint(providers_view, builtin_bytes)
    cache_key = str(target)

    # ── skip ──────────────────────────────────────────────────────────────
    if _READY_CACHE.get(cache_key) == fp:
        _log.debug("[models_gen] skip — 指纹命中缓存")
        return EnsureResult(wrote=False, path=target)

    # ── 生成新内容 ────────────────────────────────────────────────────────
    generated = _merge(providers_view, builtin)
    generated["_fingerprint"] = fp
    generated["_doc"] = (
        "运行时生成文件（由 provider.models_gen.ensure_models_json 管理）。"
        "如需自定义模型，请在 lingzhou.json providers.<name>.models 中添加，"
        "重启后自动合并进此文件。"
    )
    new_content = json.dumps(generated, ensure_ascii=False, indent=2)

    # ── noop ──────────────────────────────────────────────────────────────
    if target.exists() and target.read_text(encoding="utf-8") == new_content:
        _remember_ready_fingerprint(cache_key, fp)
        _log.debug("[models_gen] noop — 文件内容未变: %s", target)
        return EnsureResult(wrote=False, path=target)

    # ── write ─────────────────────────────────────────────────────────────
    was_new = not target.exists()
    target.write_text(new_content, encoding="utf-8")
    _remember_ready_fingerprint(cache_key, fp)
    _log.info("[models_gen] %s: %s", "已创建" if was_new else "已更新", target)
    return EnsureResult(wrote=True, path=target)
