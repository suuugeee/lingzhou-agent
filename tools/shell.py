"""tools/shell.py — shell.run 工具。"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import tempfile
from pathlib import Path
from typing import Any

from tools.registry import ToolContext, ToolManifest, ToolParam, ToolResult, tool, tool_metadata

_DEFAULT_TIMEOUT = 30.0
_MAX_SUMMARY_CHARS = 4096
_MAX_OUTPUT_PREVIEW_CHARS = 2048
_MANIFEST = ToolManifest(
    name="shell.run",
    description=(
        "在当前宿主环境中执行一次性 shell 命令（非持久会话）。"
        "返回 stdout+stderr 合并输出预览，保留头尾与长度信息避免上下文污染。"
        "高风险命令会自动触发沙箱隔离并向工作记忆注入危险感知信号。"
    ),
    progress_category="mutation",
    capabilities=("completion_verify",),
    params=[
        ToolParam("command", "string", "要执行的 bash 命令", required=True),
        ToolParam("timeout", "number", "超时秒数，默认 30", required=False),
        ToolParam("workdir", "string", "工作目录，默认项目根目录", required=False),
        ToolParam("sandbox", "boolean", "是否在隔离沙箱中运行（临时目录 + 受限 PATH）；危险命令自动启用", required=False),
    ],
)

# ── 危险命令感知 ────────────────────────────────────────────────────────────────

_RISKY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # rm 递归/强制删除（根路径、家目录、相对上级）
    (re.compile(r'\brm\b.*-[^\s]*[rR][^\s]*\s+(/|~/|--\s*/)', re.S), "rm 递归删除系统/根路径"),
    (re.compile(r'\brm\b.*-[^\s]*[fF][^\s]*\s+(/|~/)', re.S), "rm 强制删除根路径"),
    # dd 磁盘写入
    (re.compile(r'\bdd\b.+\bif=/dev/', re.S), "dd 直接读写磁盘设备"),
    (re.compile(r'\bdd\b.+\bof=/dev/', re.S), "dd 直接写入磁盘设备"),
    # 磁盘格式化
    (re.compile(r'\b(mkfs|fdisk|parted)\b'), "磁盘格式化/分区操作"),
    (re.compile(r'\bdiskutil\s+(erase|reformat)\b'), "diskutil 抹盘"),
    # 从网络下载并直接执行
    (re.compile(r'(curl|wget)\b.+\|\s*(ba?sh|sh|zsh|python\d*)\b', re.S), "从网络下载并管道执行脚本"),
    # 写入裸设备文件
    (re.compile(r'>\s*/dev/(sd[a-z]|nvme|disk\d)'), "直接写入块设备"),
    # 系统关机/重启
    (re.compile(r'\b(shutdown|reboot|poweroff|halt|init\s+0)\b'), "系统关机或重启"),
    # fork 炸弹
    (re.compile(r':\s*\(\s*\)\s*\{'), "fork 炸弹特征"),
    # 修改根目录或系统路径权限
    (re.compile(r'\bchmod\b.*\s(/|/etc|/usr|/bin)\b'), "修改系统目录权限"),
    # 清空 cron
    (re.compile(r'\bcrontab\s+-r\b'), "清空所有 cron 任务"),
    # 清空文件系统层文件
    (re.compile(r'\bshred\b.+(/etc|/usr|/bin|/lib)'), "shred 销毁系统文件"),
    # 高风险数据库操作（宽泛匹配，非精确 SQL 语法）
    (re.compile(r'\b(DROP\s+DATABASE|DROP\s+TABLE|TRUNCATE\s+TABLE)\b', re.IGNORECASE), "数据库破坏性 DDL"),
]


def check_command_risk(command: str) -> tuple[bool, str]:
    """检测命令是否含有高风险模式。返回 (is_risky, reason)。公开契约，供 smoke/诊断使用。"""
    if not isinstance(command, str):
        raise TypeError(f"command 应为字符串，实际收到 {type(command).__name__}")
    for pattern, reason in _RISKY_PATTERNS:
        if pattern.search(command):
            return True, reason
    return False, ""


def _check_risky(command: str) -> tuple[bool, str]:
    return check_command_risk(command)

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_workdir(raw: Any, ctx: ToolContext | None = None) -> Path:
    if raw is None or raw == "":
        repo_root = _repo_root()
        if repo_root.exists():
            return repo_root
        return Path.cwd()
    p = Path(str(raw)).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"workdir 不存在: {p}")
    return p


def _threshold_value(ctx: ToolContext, attr: str, default: Any) -> Any:
    config = getattr(ctx, "config", None)
    thresholds = getattr(config, "thresholds", None)
    return getattr(thresholds, attr, default)


def _decode_output(data: bytes | None) -> str:
    if not data:
        return ""
    return data.decode("utf-8", errors="replace")


def _shorten_output(text: str, *, max_chars: int = _MAX_OUTPUT_PREVIEW_CHARS) -> str:
    """对超长输出保留头尾 + 省略说明，尽量保留关键上下文信号。"""
    value = str(text or "")
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    keep_each = max(64, (max_chars - 64) // 2)
    omitted = len(value) - keep_each * 2
    return (
        value[:keep_each]
        + f"\n...[output omitted {omitted} chars]...\n"
        + value[-keep_each:]
    )


def _build_summary(status: str, workdir: Path, *, use_sandbox: bool, is_risky: bool, risk_reason: str, output_preview: str) -> str:
    summary = f"{status} cwd={workdir}"
    if use_sandbox:
        summary = f"[sandbox] {summary}"
    if is_risky:
        summary = f"[risky:{risk_reason}] {summary}"
    output_part = output_preview if output_preview else "(无输出)"
    if len(output_part) > _MAX_SUMMARY_CHARS:
        output_part = _shorten_output(output_part, max_chars=_MAX_SUMMARY_CHARS)
    return f"{summary} | {output_part}"


def _fingerprint(command: str, workdir: Path, returncode: int, output: str) -> str:
    digest = hashlib.sha256()
    digest.update(command.encode("utf-8", errors="replace"))
    digest.update(b"\0")
    digest.update(str(workdir).encode("utf-8", errors="replace"))
    digest.update(b"\0")
    digest.update(str(returncode).encode("utf-8", errors="replace"))
    digest.update(b"\0")
    digest.update(output.encode("utf-8", errors="replace"))
    return f"shell:{digest.hexdigest()[:16]}"


@tool(_MANIFEST)
async def shell_run(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    _raw_cmd = params.get("command")
    # 宽松类型转换：自动将非字符串转为字符串，避免 command必须为字符串 报错
    command = str(_raw_cmd).strip() if _raw_cmd is not None else ""
    if not command:
        return ToolResult(summary="命令为空", skipped=True)

    try:
        return await _shell_run_impl(params, ctx, command)
    except FileNotFoundError as e:
        return ToolResult(summary=str(e), error="WorkdirNotFound")
    except (ValueError, TypeError) as e:
        return ToolResult(summary=f"参数错误: {e}", error="InvalidParam")
    except OSError as e:
        return ToolResult(summary=f"启动进程失败: {e}", error="OSError")
    except Exception as e:
        return ToolResult(summary=f"工具执行异常: {type(e).__name__}: {e}", error=type(e).__name__)


async def _shell_run_impl(params: dict[str, Any], ctx: ToolContext, command: str) -> ToolResult:
    timeout_raw = params.get("timeout")
    timeout = float(
        _threshold_value(ctx, "shell_timeout", _DEFAULT_TIMEOUT)
        if timeout_raw is None
        else timeout_raw
    )

    # ── 危险感知 ─────────────────────────────────────────────────────────────
    is_risky, risk_reason = _check_risky(command)
    sandbox_param = params.get("sandbox")
    # sandbox=True if: (a) 显式请求，或 (b) 命令高风险且用户未明确关闭
    use_sandbox = bool(sandbox_param) if sandbox_param is not None else is_risky

    if is_risky:
        # 向工作记忆注入危险感知信号（不阻断执行，让灵舟自己判断）
        try:
            from memory.working import WMItem
            ctx.wm.add(WMItem(
                kind="caution",
                content=f"shell.run 危险感知（{risk_reason}）: {command}",
                priority=0.96,
            ))
        except Exception:
            pass  # WM 不可用时不阻断

    workdir_raw = params.get("workdir")

    # ── 沙箱模式：临时目录 + 受限 PATH ────────────────────────────────────────
    _sandbox_dir: str | None = None
    if use_sandbox:
        _sandbox_dir = tempfile.mkdtemp(prefix="lz_sandbox_")
        workdir = Path(_sandbox_dir)
    else:
        workdir = _resolve_workdir(workdir_raw, ctx)  # FileNotFoundError 由外层捕获

    # 最小化 env：过滤含 API_KEY / TOKEN / SECRET / PASSWORD / CREDENTIAL / AUTH 的变量
    # 防止提示注入攻击通过 printenv / curl 等命令外泄 API 密钥
    _SECRET_KWORDS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")
    safe_env = {
        k: v for k, v in os.environ.items()
        if not any(kw in k.upper() for kw in _SECRET_KWORDS)
    }
    if use_sandbox:
        # 沙箱模式：只保留最小 PATH，将 HOME 重定向到沙箱目录
        safe_env["PATH"] = "/usr/bin:/bin:/usr/local/bin"
        safe_env["HOME"] = _sandbox_dir or str(workdir)
        safe_env["TMPDIR"] = _sandbox_dir or str(workdir)

    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=str(workdir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=safe_env,
        executable=(os.environ.get("SHELL") or shutil.which("bash") or "/bin/sh"),
        start_new_session=True,  # 新建进程组，超时时可整组终止
    )

    timed_out = False
    stdout_b = b""
    stderr_b = b""

    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        timed_out = True
        # 杀整个进程组（包括 git 派生的 ssh 等子进程），防止子进程持续占用管道
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            with contextlib.suppress(Exception):
                proc.kill()
        # 洗残据输出，加短超时防止管道排水第二次挂起
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        except (TimeoutError, Exception):
            stdout_b = b""
            stderr_b = b""

    returncode = proc.returncode if proc.returncode is not None else -1
    stdout = _decode_output(stdout_b)
    stderr = _decode_output(stderr_b)
    combined = stdout
    if stdout and stderr:
        combined += "\n"
    combined += stderr
    status = "timeout" if timed_out else f"exit={returncode}"
    output_text = combined or "(无输出)"
    output_preview = _shorten_output(output_text)
    summary = _build_summary(
        status,
        workdir,
        use_sandbox=use_sandbox,
        is_risky=is_risky,
        risk_reason=risk_reason,
        output_preview=output_preview,
    )

    log_summary = (
        f"shell.run {'timeout' if timed_out else f'exit={returncode}'} chars={len(combined)}"
    )
    evidence_preview = _shorten_output(output_text, max_chars=_MAX_OUTPUT_PREVIEW_CHARS)
    stdout_preview = _shorten_output(stdout or "")
    stderr_preview = _shorten_output(stderr or "")
    payload = tool_metadata(
        "shell.run",
        log_summary,
        command=command,
        workdir=str(workdir),
        timeout_sec=timeout,
        timed_out=timed_out,
        returncode=returncode,
        stdout_chars=len(stdout),
        stderr_chars=len(stderr),
        output_chars=len(combined),
        output_preview=evidence_preview,
        stdout_preview=stdout_preview,
        stderr_preview=stderr_preview,
        output_preview_chars=len(output_preview),
        output_omitted_chars=max(0, len(output_text) - len(evidence_preview)),
        sandbox=use_sandbox,
        sandbox_dir=_sandbox_dir,
        risky=is_risky,
        risk_reason=risk_reason,
    )

    return ToolResult(
        summary=summary,
        evidence=json.dumps(payload, ensure_ascii=False),
        resource_key=str(workdir),
        fingerprint=_fingerprint(command, workdir, returncode, combined),
        metadata=payload,
        error="timeout" if timed_out else (f"exit={returncode}" if returncode != 0 else None),
        state_delta={
            "process": "finished",
            "exit_code": returncode,
            "timed_out": timed_out,
            "sandbox": use_sandbox,
            "risky": is_risky,
        },
    )
