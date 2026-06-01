"""cli/diag.py — version / doctor 命令（诊断与版本信息）。"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.panel import Panel

from cli.common import DEFAULT_CONFIG_PATH, PROJECT_ROOT, console, load_cfg, resolve_config_path
from store.auth import (
    AUTH_PROFILES_PATH,
    get_auth_profile,
    load_legacy_credentials,
    resolve_copilot_token,
)

_ENV_VAR_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")

# ── Rich 标记常量 ──────────────────────────────────────────────────────────
_OK   = "[bold green]✓[/bold green]"
_FAIL = "[bold red]✗[/bold red]"
_WARN = "[bold yellow]![/bold yellow]"


@dataclass
class _CheckResult:
    """单项诊断结果。ok=True 表示通过；issue 非 None 时会加入汇总问题列表。"""
    ok: bool
    message: str        # 完整 Rich 格式化行，直接 console.print
    issue: str | None = None


# ── 工具函数 ──────────────────────────────────────────────────────────────

def _mask_token(token: str) -> str:
    if len(token) <= 12:
        return "*" * len(token)
    return f"{token[:6]}...{token[-3:]}"


def _resolve_openai_provider_api_key(provider: object) -> tuple[str | None, str | None]:
    api_key_ref = str(getattr(provider, "api_key_env", "") or "").strip()
    if not api_key_ref:
        return None, None
    if not _ENV_VAR_NAME_RE.fullmatch(api_key_ref):
        return api_key_ref, "literal"

    env_token = os.environ.get(api_key_ref, "").strip()
    if env_token:
        return env_token, f"env:{api_key_ref}"

    legacy = load_legacy_credentials()
    legacy_token = str(legacy.get(api_key_ref, "")).strip()
    if legacy_token:
        return legacy_token, f"legacy-credentials:{api_key_ref}"

    profile_id = str(getattr(provider, "auth_profile_id", "") or "").strip()
    if profile_id:
        profile = get_auth_profile(profile_id)
        if isinstance(profile, dict):
            profile_token = str(profile.get("token", "")).strip()
            if profile_token:
                return profile_token, f"auth-profile:{profile_id}"

    return None, None


# ── 独立检查函数（可单独单元测试）────────────────────────────────────────

def _check_python_version() -> _CheckResult:
    import sys

    from core.version import __min_python__
    py = sys.version_info
    py_str = f"{py.major}.{py.minor}.{py.micro}"
    if py[:3] >= __min_python__:
        return _CheckResult(ok=True, message=f"  {_OK} Python {py_str}")
    need = ".".join(str(x) for x in __min_python__)
    return _CheckResult(
        ok=False,
        message=f"  {_FAIL} Python {py_str}  (需要 ≥ {need})",
        issue=f"Python 版本过低: {py_str}",
    )


def _check_dependencies() -> list[_CheckResult]:
    import importlib
    results: list[_CheckResult] = []
    for dep in ["pydantic", "httpx", "aiosqlite", "typer", "rich"]:
        try:
            importlib.import_module(dep)
            results.append(_CheckResult(ok=True, message=f"  {_OK} {dep}"))
        except ImportError:
            results.append(_CheckResult(
                ok=False,
                message=f"  {_FAIL} {dep}  [dim]未安装[/dim]",
                issue=f"缺少依赖: {dep}",
            ))
    return results


def _check_config_file(resolved_config: Path) -> _CheckResult:
    import json as _json
    if resolved_config.exists():
        try:
            _json.loads(resolved_config.read_text(encoding="utf-8"))
            return _CheckResult(ok=True, message=f"  {_OK} 配置文件: {resolved_config}")
        except Exception as e:
            return _CheckResult(
                ok=False,
                message=f"  {_FAIL} 配置文件解析失败: {e}",
                issue=f"配置文件无效: {e}",
            )
    return _CheckResult(
        ok=False,
        message=f"  {_WARN} 配置文件不存在: {resolved_config}  [dim]运行 lingzhou onboard[/dim]",
        issue=f"配置文件缺失: {resolved_config}",
    )


def _load_cfg_silently(config: Path) -> Any | None:
    try:
        return load_cfg(config)
    except Exception:
        return None


def _check_api_key(cfg: Any) -> list[_CheckResult]:
    results: list[_CheckResult] = []
    _api_key_env: str | None = None
    _provider_name: str | None = None
    try:
        _provider_name = cfg.active_provider_name
        _api_key_env = cfg.active_provider.api_key_env
    except Exception:
        pass

    if not _api_key_env:
        return results

    if getattr(cfg.active_provider, 'mode', '') == 'copilot':
        resolved = resolve_copilot_token(_api_key_env)
        if resolved:
            if resolved.source.startswith('env:'):
                env_name = resolved.source.split(':', 1)[1]
                masked = _mask_token(resolved.token)
                results.append(_CheckResult(
                    ok=True,
                    message=f"  {_OK} Copilot token ({env_name}): {masked}",
                ))
            elif resolved.source == 'auth-profile':
                results.append(_CheckResult(
                    ok=True,
                    message=f"  {_OK} Copilot token: 来自 auth profile store  [dim]{AUTH_PROFILES_PATH}[/dim]",
                ))
            else:
                results.append(_CheckResult(
                    ok=True,
                    message=f"  {_OK} Copilot token: 来自 legacy credentials 文件",
                ))
        else:
            results.append(_CheckResult(
                ok=False,
                message=f"  {_FAIL} Copilot token: 未设置",
                issue="Copilot token 未配置: lingzhou auth login-copilot",
            ))
    else:
        _resolved_key, _resolved_source = _resolve_openai_provider_api_key(cfg.active_provider)
        if _resolved_key:
            if _resolved_source == "literal":
                msg = f"  {_OK} API key (literal): {_mask_token(_resolved_key)}"
            elif _resolved_source and _resolved_source.startswith("env:"):
                env_name = _resolved_source.split(":", 1)[1]
                msg = f"  {_OK} API key ({env_name}): {_mask_token(_resolved_key)}"
            elif _resolved_source and _resolved_source.startswith("legacy-credentials:"):
                env_name = _resolved_source.split(":", 1)[1]
                msg = f"  {_OK} API key ({env_name}): 来自 credentials 文件"
            elif _resolved_source and _resolved_source.startswith("auth-profile:"):
                profile_id = _resolved_source.split(":", 1)[1]
                msg = f"  {_OK} API key ({_api_key_env}): 来自 auth profile  [dim]{profile_id}[/dim]"
            else:
                msg = f"  {_OK} API key ({_api_key_env}): 已解析"
            results.append(_CheckResult(ok=True, message=msg))
        else:
            if _ENV_VAR_NAME_RE.fullmatch(_api_key_env):
                _provider_hint = _provider_name or "<provider>"
                issue = f"API key 未配置: export {_api_key_env}=your_key 或 lingzhou auth set-token --provider {_provider_hint}"
            else:
                issue = "API key 未配置: lingzhou auth set-token --provider <provider>"
            results.append(_CheckResult(
                ok=False,
                message=f"  {_FAIL} API key ({_api_key_env}): 未设置",
                issue=issue,
            ))
    return results


def _check_database(cfg: Any) -> _CheckResult:
    db_path = cfg.db_path
    if db_path.exists():
        try:
            from store.task.ingress import IngressStore
            tables = IngressStore(db_path).list_tables()
            return _CheckResult(
                ok=True,
                message=f"  {_OK} 数据库: {db_path}  [dim]表: {', '.join(tables) or '(空)'}[/dim]",
            )
        except Exception as e:
            return _CheckResult(
                ok=False,
                message=f"  {_FAIL} 数据库异常: {e}",
                issue=f"DB 异常: {e}",
            )
    return _CheckResult(
        ok=False,
        message=f"  {_WARN} 数据库未初始化: {db_path}  [dim]运行 lingzhou onboard[/dim]",
    )


def _check_config_schema() -> list[_CheckResult]:
    results: list[_CheckResult] = []
    try:
        from core.loop.runtime.startup import (
            _MEMORY_FIELD_PATCHES,
            _THRESHOLDS_FIELD_PATCHES,
            _missing_config_schema_fields,
        )

        missing_all = _missing_config_schema_fields()
        for _cls_name, _patches in [
            ("ThresholdsConfig", _THRESHOLDS_FIELD_PATCHES),
            ("MemoryConfig", _MEMORY_FIELD_PATCHES),
        ]:
            _still_missing = missing_all.get(_cls_name, [])
            if _still_missing:
                results.append(_CheckResult(
                    ok=False,
                    message=f"  {_FAIL} {_cls_name} 缺少字段: {_still_missing}  [dim](请 git pull 升级 core.config_models)[/dim]",
                    issue=f"core.config_models 版本过旧，{_cls_name} 缺少: {_still_missing}",
                ))
            else:
                results.append(_CheckResult(
                    ok=True,
                    message=f"  {_OK} {_cls_name} schema  [dim]({len(_patches)} 个关键字段均存在)[/dim]",
                ))
    except Exception as e:
        results.append(_CheckResult(
            ok=False,
            message=f"  {_FAIL} Config schema 检查失败: {e}",
            issue=f"Config schema 检查异常: {e}",
        ))
    return results


def _check_directories(cfg: Any) -> list[_CheckResult]:
    import tempfile
    results: list[_CheckResult] = []
    _dirs_to_check = [
        ("memory_dir", cfg.memory_dir),
        ("workspace_dir", cfg.workspace_dir),
        ("db_parent", Path(cfg.db_path).parent),
    ]
    for _label, _dpath in _dirs_to_check:
        _dpath = Path(_dpath).expanduser()
        if not _dpath.exists():
            try:
                _dpath.mkdir(parents=True, exist_ok=True)
                results.append(_CheckResult(
                    ok=False,
                    message=f"  {_WARN} {_label}: {_dpath}  [dim]不存在，已创建[/dim]",
                ))
            except Exception as _e:
                results.append(_CheckResult(
                    ok=False,
                    message=f"  {_FAIL} {_label}: {_dpath}  无法创建: {_e}",
                    issue=f"{_label} 无法创建: {_e}",
                ))
        else:
            try:
                with tempfile.NamedTemporaryFile(dir=_dpath, delete=True):
                    pass
                results.append(_CheckResult(
                    ok=True,
                    message=f"  {_OK} {_label}: {_dpath}  [dim]可写[/dim]",
                ))
            except Exception:
                results.append(_CheckResult(
                    ok=False,
                    message=f"  {_FAIL} {_label}: {_dpath}  [dim]无写权限[/dim]",
                    issue=f"{_label} 目录无写权限: {_dpath}",
                ))
    return results


def _check_db_schema(cfg: Any) -> _CheckResult:
    _REQUIRED_TABLES = {
        "tasks", "failures", "facts", "signals",
        "chat_messages", "runs", "meta_reflections",
    }
    if not Path(cfg.db_path).exists():
        return _CheckResult(
            ok=False,
            message=f"  {_WARN} DB schema: 跳过（数据库未初始化）",
        )
    try:
        from store.task.ingress import IngressStore
        _tables = set(IngressStore(cfg.db_path).list_tables())
        _missing = _REQUIRED_TABLES - _tables
        if _missing:
            return _CheckResult(
                ok=False,
                message=f"  {_WARN} DB schema: 缺少表 {sorted(_missing)}  [dim]lingzhou run 首次启动时会自动建表[/dim]",
            )
        return _CheckResult(
            ok=True,
            message=f"  {_OK} DB schema: 关键表均存在  [dim]{sorted(_REQUIRED_TABLES)}[/dim]",
        )
    except Exception as _e:
        return _CheckResult(
            ok=False,
            message=f"  {_FAIL} DB schema 检查失败: {_e}",
            issue=f"DB schema 异常: {_e}",
        )


def _check_plugins() -> _CheckResult:
    try:
        from core.plugin import PluginManager
        _pm = PluginManager(Path(PROJECT_ROOT / "plugins").expanduser())
        _pm.discover()
        _loaded = _pm.list_plugins()
        if _loaded:
            names = ', '.join(p['name'] for p in _loaded[:4])
            suffix = '...' if len(_loaded) > 4 else ''
            return _CheckResult(
                ok=True,
                message=f"  {_OK} 插件: {len(_loaded)} 个已发现  [dim]{names}{suffix}[/dim]",
            )
        return _CheckResult(
            ok=True,
            message=f"  {_OK} 插件: 无已安装插件  [dim](plugins/ 目录为空)[/dim]",
        )
    except Exception as _e:
        return _CheckResult(ok=False, message=f"  {_WARN} 插件检查失败: {_e}")


def _check_tool_registry() -> _CheckResult:
    try:
        from tools.registry import ToolRegistry
        reg = ToolRegistry()
        reg.discover(PROJECT_ROOT / "tools")
        tool_ids = [m.name for m in reg.list_manifests()]
        names = ', '.join(tool_ids[:6])
        suffix = '...' if len(tool_ids) > 6 else ''
        return _CheckResult(
            ok=True,
            message=f"  {_OK} 工具注册: {len(tool_ids)} 个  [dim]{names}{suffix}[/dim]",
        )
    except Exception as e:
        return _CheckResult(
            ok=False,
            message=f"  {_FAIL} 工具注册失败: {e}",
            issue=f"工具注册异常: {e}",
        )


def _probe_model(cfg: Any) -> _CheckResult:
    try:
        import asyncio as _asyncio

        from provider import create_provider

        async def _ping_and_close() -> tuple[bool, int, str | None]:
            _prov_inst = create_provider(cfg)
            try:
                return await _prov_inst.ping()
            finally:
                await _prov_inst.close()

        _model_id = cfg.active_model_id
        _ok, _ms, _err = _asyncio.run(_ping_and_close())
        if _ok:
            return _CheckResult(ok=True, message=f"  {_OK} 模型探针: {_model_id} 响应 {_ms}ms")
        if _err and any(x in _err for x in ("认证", "401", "403")):
            return _CheckResult(
                ok=False,
                message=f"  {_FAIL} 模型探针: {_model_id} {_err}",
                issue=f"API key 认证失败: {_err}",
            )
        return _CheckResult(
            ok=False,
            message=f"  {_WARN} 模型探针: {_model_id} {_err or 'unknown'} ({_ms}ms)",
        )
    except Exception as _e:
        return _CheckResult(ok=False, message=f"  {_WARN} 模型探针: 跳过 ({_e})")


# ── CLI 命令 ─────────────────────────────────────────────────────────────

def version() -> None:
    """显示版本信息。"""
    import sys

    from core.version import __codename__, __min_python__, __version__
    console.print(f"[bold]lingzhou[/bold] v{__version__}  代号: {__codename__}")
    console.print(f"  Python {sys.version.split()[0]}  (要求 ≥ {'.'.join(str(x) for x in __min_python__)})")


def doctor(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
) -> None:
    """自检：诊断运行环境、配置、API key 和数据库状态。"""
    from core.version import __version__

    issues: list[str] = []

    def _emit(*results: _CheckResult) -> None:
        for r in results:
            console.print(r.message)
            if r.issue:
                issues.append(r.issue)

    console.print(Panel(
        f"[bold]lingzhou doctor[/bold]  v{__version__}",
        border_style="cyan",
    ))

    _emit(_check_python_version())
    _emit(*_check_dependencies())

    resolved_config = resolve_config_path(config)
    _emit(_check_config_file(resolved_config))

    cfg = _load_cfg_silently(config) if resolved_config.exists() else None

    if cfg is not None:
        api_key_results = _check_api_key(cfg)
        if api_key_results:
            _emit(*api_key_results)
        _emit(_check_database(cfg))
    else:
        console.print(f"  {_WARN} API key: 跳过（配置文件不可用）")
        console.print(f"  {_WARN} 数据库: 跳过（配置文件不可用）")

    _emit(*_check_config_schema())

    if cfg is not None:
        _emit(*_check_directories(cfg))
        _emit(_check_db_schema(cfg))
    else:
        console.print(f"  {_WARN} 目录权限: 跳过（配置文件不可用）")
        console.print(f"  {_WARN} DB schema: 跳过（数据库未初始化）")

    _emit(_check_plugins())
    _emit(_check_tool_registry())

    if cfg is not None:
        _emit(_probe_model(cfg))
    else:
        console.print(f"  {_WARN} 模型探针: 跳过（配置文件不可用）")

    # ── 汇总 ────────────────────────────────────────────────────────────
    console.print("")
    if not issues:
        console.print("[bold green]所有检查通过。[/bold green] 可以运行 [bold]lingzhou run[/bold]")
    else:
        console.print(f"[bold red]发现 {len(issues)} 个问题：[/bold red]")
        for i, issue in enumerate(issues, 1):
            console.print(f"  {i}. {issue}")
        raise typer.Exit(1)

