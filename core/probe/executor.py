"""core/probe/executor.py — 探针执行器（shell / http / python）。

每种执行器都是无状态的，接收 ProbeConfig 返回 (output, error)。
所有阻塞操作通过 asyncio.to_thread 隔离，不阻塞事件循环。
"""
from __future__ import annotations

import asyncio
import io
import logging
import subprocess
from contextlib import redirect_stdout
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.contracts.probe import ProbeConfig

_log = logging.getLogger("lingzhou.probe")

# 探针执行超时（秒）
DEFAULT_TIMEOUT_SEC = 30


async def execute_probe(cfg: ProbeConfig, timeout: int = DEFAULT_TIMEOUT_SEC) -> tuple[str, str | None]:  # noqa: ASYNC109
    """执行探针，返回 (output, error)。output 为空字符串表示无输出。"""
    try:
        if cfg.kind == "shell":
            return await _run_shell(cfg.spec, timeout)
        if cfg.kind == "http":
            return await _run_http(cfg.spec, timeout)
        if cfg.kind == "python":
            return await _run_python(cfg.spec, timeout)
        if cfg.kind == "builtin":
            return await _run_builtin(cfg.spec)
        return "", f"未知探针类型: {cfg.kind}"
    except TimeoutError:
        return "", f"超时（>{timeout}s）"
    except Exception as exc:
        return "", str(exc)


async def _run_shell(cmd: str, timeout: int) -> tuple[str, str | None]:  # noqa: ASYNC109
    def _blocking() -> tuple[str, str | None]:
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = (result.stdout or "").strip()
            error = (result.stderr or "").strip() if result.returncode != 0 else None
            return output, error
        except subprocess.TimeoutExpired:
            raise TimeoutError() from None
        except Exception as exc:
            return "", str(exc)

    return await asyncio.to_thread(_blocking)


async def _run_http(url: str, timeout: int) -> tuple[str, str | None]:  # noqa: ASYNC109
    try:
        import httpx  # type: ignore[import]
    except ImportError:
        return "", "httpx 未安装，无法使用 http 类型探针"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            return resp.text.strip(), None
    except httpx.HTTPError as exc:
        return "", str(exc)


async def _run_builtin(spec: str) -> tuple[str, str | None]:
    """执行内置探针（格式: 'module.path:func_name'，零参数，返回 str）。

    函数在当前进程中运行（非沙盒），可访问进程内全局缓存。
    通过 asyncio.to_thread 隔离文件 I/O。
    """
    def _blocking() -> tuple[str, str | None]:
        try:
            module_path, func_name = spec.rsplit(":", 1)
        except ValueError:
            return "", f"builtin spec 格式错误（需 'module:func'）: {spec!r}"
        try:
            import importlib
            mod = importlib.import_module(module_path)
            func = getattr(mod, func_name)
            result = func()
            return str(result) if result is not None else "", None
        except Exception as exc:
            return "", str(exc)

    return await asyncio.to_thread(_blocking)


async def _run_python(code: str, timeout: int) -> tuple[str, str | None]:  # noqa: ASYNC109
    """在受限沙盒中执行 Python 代码片段。stdout 作为输出。

    沙盒限制：只开放 print / len / range / int / float / str / list / dict /
    math / datetime 等安全内置。不允许 import os/sys/subprocess 等危险模块。
    """

    def _blocking() -> tuple[str, str | None]:
        import datetime as _dt
        import math

        safe_globals: dict = {
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "int": int,
                "float": float,
                "str": str,
                "bool": bool,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "abs": abs,
                "min": min,
                "max": max,
                "sum": sum,
                "round": round,
                "sorted": sorted,
                "reversed": reversed,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "isinstance": isinstance,
                "type": type,
                "repr": repr,
                "math": math,
                "datetime": _dt,
            }
        }
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(compile(code, "<probe>", "exec"), safe_globals)
            return buf.getvalue().strip(), None
        except Exception as exc:
            return buf.getvalue().strip(), str(exc)

    return await asyncio.wait_for(asyncio.to_thread(_blocking), timeout=timeout)
