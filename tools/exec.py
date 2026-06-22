"""tools/exec.py — exec/process 工具。

目标：
- exec：启动 shell 命令，支持前台/后台、PTY、超时、工作目录、环境变量
- process：管理已启动的后台进程（list/poll/log/write/kill）

注意：
- 不引入重型审批/安全抽象；这里先补能力本体
- 进程状态当前为进程内内存态，runtime 重启后不会恢复（后续可持久化）
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from core.resource_guard import (
    build_memory_limit_preexec,
    local_embedding_memory_preflight,
    memory_guard_settings,
)
from tools.exec_helpers import (
    ProcessInfo,
    ProcessManager,
    _append_output,
    _build_capabilities,
    _preview,
    _spawn_pty_process,
    _terminate_info,
    _watch_pty_process,
)
from tools.registry import ToolContext, ToolManifest, ToolParam, ToolResult, tool, tool_metadata

_log = logging.getLogger("lingzhou.tools.exec")


_MANAGER = ProcessManager()


_CAP_MANIFEST = ToolManifest(
    name="shell.capabilities",
    description="返回 shell 执行能力画像（可用命令、默认限制、环境语义、exec/process 支持）",
    params=[],
    prefer_tier="reader",
    progress_category="info",
    capabilities=("plan_bootstrap_exempt", "plan_alignment_exempt", "completion_info_only"),
)


@tool(_CAP_MANIFEST)
async def shell_capabilities(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    workdir = params.get("workdir", str(Path.cwd()))
    caps = _build_capabilities(workdir)
    summary = (
        f"shell.capabilities: sandbox={caps['sandbox']} "
        f"background={caps['has_background_exec']} "
        f"process_mgmt={caps['has_process_management']} "
        f"process_write={caps['has_process_write']} "
        f"pty={caps['has_pty']} "
        f"cmds={len(caps['available_commands'])}"
    )
    return ToolResult(
        summary=summary,
        evidence=json.dumps(caps, ensure_ascii=False),
        resource_key=workdir,
        fingerprint=f"caps:{len(caps['available_commands'])}:{int(caps['has_pty'])}",
        metadata=tool_metadata("shell.capabilities", summary, caps=caps),
    )


# ── exec：启动命令 ───────────────────────────────────────────────────────────

_EXEC_MANIFEST = ToolManifest(
    name="exec",
    description=(
        "启动 shell 命令。支持前台阻塞执行或后台运行。"
        "前台模式：等待命令完成，返回完整输出（受 timeout 限制）。"
        "后台模式：立即返回 process_id，后续通过 process.poll/log/write/kill 管理。"
        "支持 pty=true 运行需要 TTY 的交互式程序（如 python -i、vim）。"
    ),
    params=[
        ToolParam("command", "string", "要执行的 shell 命令", required=True),
        ToolParam("background", "boolean", "是否后台运行（默认 false）", required=False),
        ToolParam("pty", "boolean", "是否使用 PTY（适合交互式程序）", required=False),
        ToolParam("timeout", "number", "超时秒数，默认 30（前台）或 300（后台）", required=False),
        ToolParam("workdir", "string", "工作目录，默认当前目录", required=False),
        ToolParam("env", "object", "环境变量字典（可选）", required=False),
    ],
    progress_category="mutation",
    capabilities=("run_spawn",),
)


@tool(_EXEC_MANIFEST)
async def exec_run(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    command = (params.get("command") or "").strip()
    if not command:
        return ToolResult(summary="命令为空", skipped=True, error="EmptyCommand")

    background = bool(params.get("background", False))
    use_pty = bool(params.get("pty", False))
    timeout = float(params.get("timeout") or (300.0 if background else 30.0))
    workdir = str(params.get("workdir") or Path.cwd())
    env_overrides = params.get("env")

    if ctx.dry_run:
        return ToolResult(
            summary=f"[dry-run] exec: {command}",
            evidence=json.dumps({
                "dry_run": True,
                "command": command,
                "timeout": timeout,
                "workdir": workdir,
                "background": background,
                "pty": use_pty,
            }, ensure_ascii=False),
            skipped=True,
        )

    guard_enabled, required_mib = memory_guard_settings(getattr(ctx, "config", None))
    guard = local_embedding_memory_preflight(
        command=command,
        min_available_mib=required_mib,
        guard_enabled=guard_enabled,
    )
    memory_preexec = build_memory_limit_preexec(guard.limit_mib if guard.matched and not guard.ok else None)

    exec_env = os.environ.copy()
    if env_overrides and isinstance(env_overrides, dict):
        exec_env.update({str(k): str(v) for k, v in env_overrides.items()})

    session_id = _MANAGER.next_id()
    info = ProcessInfo(
        session_id=session_id,
        command=command,
        started_at=time.time(),
        background=background,
        workdir=workdir,
        timeout_seconds=timeout,
        pty=use_pty,
    )
    _MANAGER.register(info)

    try:
        if use_pty:
            proc, master_fd = _spawn_pty_process(command, workdir, exec_env, preexec_fn=memory_preexec)
            info.proc = proc
            info.pid = proc.pid
            info.master_fd = master_fd
            if background:
                info.watch_task = asyncio.create_task(_watch_pty_process(info))
                payload = {
                    "process_id": session_id,
                    "pid": proc.pid,
                    "command": command,
                    "timeout": timeout,
                    "workdir": workdir,
                    "background": True,
                    "pty": True,
                    "resource_guard": guard.as_metadata() if guard.matched else None,
                }
                return ToolResult(
                    summary=f"后台 PTY 进程已启动: process_id={session_id}, pid={proc.pid}",
                    evidence=json.dumps(payload, ensure_ascii=False),
                    resource_key=session_id,
                    artifact_paths=[info.meta_path, info.log_path],
                    state_delta={"process": "started", "background": True, "pty": True},
                    metadata=tool_metadata(
                        "exec",
                        f"exec background pty pid={proc.pid}",
                        **payload,
                    ),
                )
            await _watch_pty_process(info)
        else:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=workdir,
                env=exec_env,
                start_new_session=True,  # 超时/中断时可 killpg，避免子进程残留
                preexec_fn=memory_preexec,
            )
            info.proc = proc
            info.pid = proc.pid
            if background:
                info.watch_task = asyncio.create_task(_watch_pipe_process(info))
                payload = {
                    "process_id": session_id,
                    "pid": proc.pid,
                    "command": command,
                    "timeout": timeout,
                    "workdir": workdir,
                    "background": True,
                    "pty": False,
                    "resource_guard": guard.as_metadata() if guard.matched else None,
                }
                return ToolResult(
                    summary=f"后台进程已启动: process_id={session_id}, pid={proc.pid}",
                    evidence=json.dumps(payload, ensure_ascii=False),
                    resource_key=session_id,
                    artifact_paths=[info.meta_path, info.log_path],
                    state_delta={"process": "started", "background": True, "pty": False},
                    metadata=tool_metadata(
                        "exec",
                        f"exec background pid={proc.pid}",
                        **payload,
                    ),
                )
            await _watch_pipe_process(info)

        if info.return_code is None:
            latest = _MANAGER.get(session_id)
            if latest and latest.return_code is not None:
                info.return_code = latest.return_code
            elif not info.timed_out and not info.error:
                info.return_code = 0
                ProcessManager._persist(info)

        output = info.stdout.strip()
        if info.timed_out:
            payload = {
                "timeout": timeout,
                "command": command,
                "workdir": workdir,
                "timed_out": True,
                "pty": use_pty,
                "process_id": session_id,
            }
            return ToolResult(
                summary=f"执行超时（{timeout}s）: {command}",
                evidence=json.dumps(payload, ensure_ascii=False),
                error="TimeoutError",
                skipped=True,
                resource_key=session_id,
                artifact_paths=[info.meta_path, info.log_path],
                state_delta={"process": "timed_out"},
                metadata=tool_metadata(
                    "exec",
                    f"exec timeout={timeout}s command={command!r}",
                    **payload,
                ),
            )

        output_text = _preview(output, 4000) if output else "(无输出)"
        evidence = json.dumps({
            "command": command,
            "exit_code": info.return_code,
            "timeout": timeout,
            "workdir": workdir,
            "output_chars": len(output),
            "preview_chars": len(output_text),
            "pty": use_pty,
            "resource_guard": guard.as_metadata() if guard.matched else None,
        }, ensure_ascii=False)
        payload = json.loads(evidence)
        payload.update({"process_id": session_id, "meta_path": info.meta_path, "log_path": info.log_path})
        log_summary = f"exec exit={info.return_code} chars={payload['output_chars']}"
        exec_meta = tool_metadata("exec", log_summary, **payload)
        if info.return_code == 0:
            return ToolResult(
                summary=f"命令完成 (exit=0):\n{output_text}",
                evidence=json.dumps(payload, ensure_ascii=False),
                resource_key=session_id,
                fingerprint=f"exec:{info.return_code}:{payload['output_chars']}",
                artifact_paths=[info.meta_path, info.log_path],
                state_delta={"process": "finished", "exit_code": info.return_code},
                metadata=exec_meta,
            )
        return ToolResult(
            summary=f"执行出错 (exit={info.return_code}):\n{output_text}",
            evidence=json.dumps(payload, ensure_ascii=False),
            error=(output_text if output else (info.error or f"exit={info.return_code}")),
            resource_key=session_id,
            fingerprint=f"exec:{info.return_code}:{payload['output_chars']}",
            artifact_paths=[info.meta_path, info.log_path],
            state_delta={"process": "finished", "exit_code": info.return_code},
            metadata=exec_meta,
        )
    except Exception as exc:
        info.error = str(exc)
        _MANAGER.mark_finished(session_id, -1)
        _log.exception("exec 失败: %s", command)
        ProcessManager._persist(info)
        return ToolResult(
            summary=f"执行异常: {exc}",
            error=str(exc),
            resource_key=session_id,
            artifact_paths=[info.meta_path, info.log_path] if info.meta_path or info.log_path else [],
            state_delta={"process": "failed_to_start"},
            metadata=tool_metadata(
                "exec",
                f"exec failed_to_start command={command!r}",
                process_id=session_id,
                command=command,
                workdir=workdir,
            ),
        )


async def _watch_pipe_process(info: ProcessInfo) -> None:
    proc = info.proc
    assert proc is not None

    async def _reader() -> None:
        if proc.stdout is None:
            return
        while True:
            chunk = await proc.stdout.read(1024)
            if not chunk:
                break
            _append_output(info, chunk.decode(errors="replace"))

    reader_task = asyncio.create_task(_reader())
    try:
        await asyncio.wait_for(proc.wait(), timeout=info.timeout_seconds)
    except TimeoutError:
        _terminate_info(info)
        await asyncio.sleep(0.1)
        _terminate_info(info, force=True)
        info.error = "TimeoutError"
        _MANAGER.mark_finished(info.session_id, -1, timed_out=True)
        _MANAGER._persist(info)
    except Exception as e:
        info.error = str(e)
        _MANAGER.mark_finished(info.session_id, -1)
        _MANAGER._persist(info)
    else:
        _MANAGER.mark_finished(info.session_id, proc.returncode if proc.returncode is not None else -1)
        _MANAGER._persist(info)
    finally:
        try:
            await asyncio.wait_for(reader_task, timeout=1.0)
        except Exception:
            reader_task.cancel()
        try:
            if proc.stdin and not proc.stdin.is_closing():
                proc.stdin.close()
            if proc.stdout:
                await proc.stdout.read()
        except Exception:
            pass
        info.proc = None


# ── process：管理后台进程 ────────────────────────────────────────────────────

_PROCESS_MANIFEST_LIST = ToolManifest(
    name="process.list",
    description="列出所有通过 exec 启动的进程。可过滤 running/finished/all。",
    params=[ToolParam("status", "string", "过滤：running/finished/all（默认 all）", required=False)],
    progress_category="info",
    capabilities=("run_spawn",),
)

_PROCESS_MANIFEST_POLL = ToolManifest(
    name="process.poll",
    description="检查指定进程的状态。返回是否已完成、退出码、运行时间等。",
    params=[ToolParam("process_id", "string", "exec 后台启动时返回的 process_id", required=True)],
    progress_category="info",
    capabilities=("run_spawn",),
)

_PROCESS_MANIFEST_LOG = ToolManifest(
    name="process.log",
    description="获取指定进程的标准输出。支持 offset/limit 分段读取（首次不传 offset 就是从头读）。",
    params=[
        ToolParam("process_id", "string", "exec 后台启动时返回的 process_id", required=True),
        ToolParam("offset", "number", "从第几个字符开始读，默认 0", required=False),
        ToolParam("limit", "number", "最多读多少字符，默认 2000", required=False),
    ],
    progress_category="info",
    capabilities=("run_spawn",),
)

_PROCESS_MANIFEST_WRITE = ToolManifest(
    name="process.write",
    description="向后台进程写入 stdin / PTY 输入。可选 eof=true 关闭输入。",
    params=[
        ToolParam("process_id", "string", "exec 后台启动时返回的 process_id", required=True),
        ToolParam("data", "string", "要写入的文本", required=False),
        ToolParam("eof", "boolean", "写入后是否关闭输入（默认 false）", required=False),
    ],
    progress_category="mutation",
    capabilities=("run_spawn",),
)

_PROCESS_MANIFEST_KILL = ToolManifest(
    name="process.kill",
    description="强制终止指定进程。",
    params=[ToolParam("process_id", "string", "exec 后台启动时返回的 process_id", required=True)],
    progress_category="mutation",
    capabilities=("run_spawn",),
)


@tool(_PROCESS_MANIFEST_LIST)
async def process_list(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    status_filter = (params.get("status") or "all").lower()
    procs = _MANAGER.list_all()
    if status_filter == "running":
        procs = [p for p in procs if not p.finished]
    elif status_filter == "finished":
        procs = [p for p in procs if p.finished]
    if not procs:
        return ToolResult(summary=f"无进程（filter={status_filter})")
    lines = []
    process_items = []
    for p in procs:
        process_items.append(p.to_dict())
        state = "running" if not p.finished else f"done(exit={p.return_code})"
        mode = "pty" if p.pty else "pipe"
        duration = time.time() - p.started_at
        lines.append(f"  {p.session_id}: {state} [{mode}] | {p.command} | {duration:.0f}s")
    return ToolResult(
        summary=f"进程列表 ({len(procs)} 个):\n" + "\n".join(lines),
        metadata=tool_metadata(
            "process.list",
            f"process.list count={len(procs)} filter={status_filter}",
            count=len(procs),
            status_filter=status_filter,
            items=process_items,
        ),
    )


@tool(_PROCESS_MANIFEST_POLL)
async def process_poll(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    session_id = params.get("process_id") or params.get("session_id", "")
    info = _MANAGER.get(session_id)
    if not info:
        return ToolResult(summary=f"进程不存在: {session_id}", error="ProcessNotFound")
    status = info.to_dict()
    return ToolResult(
        summary=json.dumps(status, ensure_ascii=False, indent=2),
        resource_key=session_id,
        fingerprint=f"poll:{status['status']}:{status['return_code']}",
        artifact_paths=[p for p in [info.meta_path, info.log_path] if p],
        metadata=tool_metadata(
            "process.poll",
            f"process.poll {session_id} status={status.get('status')}",
            **status,
        ),
    )


@tool(_PROCESS_MANIFEST_LOG)
async def process_log(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    session_id = params.get("process_id") or params.get("session_id", "")
    info = _MANAGER.get(session_id)
    if not info:
        return ToolResult(summary=f"进程不存在: {session_id}", error="ProcessNotFound")
    offset = int(params.get("offset") or 0)
    limit = max(1, min(12_000, int(params.get("limit") or 2000)))
    if info.log_path and await asyncio.to_thread(Path(info.log_path).exists):
        try:
            output = await asyncio.to_thread(Path(info.log_path).read_text, encoding="utf-8", errors="replace")
            info.stdout = output
        except Exception:
            output = info.stdout
    else:
        output = info.stdout
    if not output:
        return ToolResult(
            summary=f"进程 {session_id} 暂无输出（进程可能刚启动或尚未产生日志）",
            metadata=tool_metadata(
                "process.log",
                f"process.log {session_id} empty",
                session_id=session_id,
                total_output_chars=0,
            ),
        )
    if offset >= len(output):
        return ToolResult(summary=f"输出总长 {len(output)} 字符，offset={offset} 超出范围", skipped=True)
    chunk = output[offset:offset + limit]
    remaining = len(output) - offset - len(chunk)
    payload = {
        "session_id": session_id,
        "offset": offset,
        "limit": limit,
        "returned_chars": len(chunk),
        "remaining_chars": max(0, remaining),
        "total_output_chars": len(output),
        "log_path": info.log_path,
    }
    return ToolResult(
        summary=chunk,
        evidence=json.dumps(payload, ensure_ascii=False),
        resource_key=session_id,
        fingerprint=f"log:{offset}:{len(chunk)}",
        artifact_paths=[p for p in [info.meta_path, info.log_path] if p],
        metadata=tool_metadata(
            "process.log",
            f"process.log {session_id} chars={len(chunk)}",
            **payload,
        ),
    )


@tool(_PROCESS_MANIFEST_WRITE)
async def process_write(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    session_id = params.get("process_id") or params.get("session_id", "")
    data = str(params.get("data") or "")
    eof = bool(params.get("eof", False))
    info = _MANAGER.get(session_id)
    if not info:
        return ToolResult(summary=f"进程不存在: {session_id}", error="ProcessNotFound")
    if info.finished:
        return ToolResult(summary=f"进程 {session_id} 已结束，不能再写入", error="ProcessFinished", skipped=True)
    if info.handle_lost or info.proc is None and info.pid and info.restored:
        _log.info("[process.write] session=%s handle lost after restore; write unavailable", session_id)
        return ToolResult(
            summary=f"进程 {session_id} 来自重启前的持久状态，当前无法恢复 stdin/PTY 写入句柄；可继续 poll/log/kill。",
            error="ProcessHandleLost",
            skipped=True,
            resource_key=session_id,
            artifact_paths=[p for p in [info.meta_path, info.log_path] if p],
            metadata=tool_metadata(
                "process.write",
                f"process.write {session_id} handle_lost",
                handle_lost=True,
                restored=info.restored,
            ),
        )

    try:
        if info.pty and info.master_fd is not None:
            if data:
                os.write(info.master_fd, data.encode())
            if eof:
                os.write(info.master_fd, b"\x04")
        else:
            proc = info.proc
            if proc is None or proc.stdin is None:
                return ToolResult(summary=f"进程 {session_id} 没有可写 stdin", error="NoStdin")
            if data:
                proc.stdin.write(data.encode())
                await proc.stdin.drain()
            if eof:
                proc.stdin.close()
        _MANAGER._persist(info)
        return ToolResult(
            summary=f"已写入进程 {session_id}: {len(data)} 字符{' + EOF' if eof else ''}",
            resource_key=session_id,
            state_delta={"stdin_write_chars": len(data), "eof": eof},
            artifact_paths=[p for p in [info.meta_path, info.log_path] if p],
            metadata=tool_metadata(
                "process.write",
                f"process.write {session_id} chars={len(data)}",
                session_id=session_id,
                chars=len(data),
                eof=eof,
            ),
        )
    except Exception as e:
        info.error = str(e)
        _MANAGER._persist(info)
        return ToolResult(summary=f"写入失败: {e}", error=str(e), resource_key=session_id)


@tool(_PROCESS_MANIFEST_KILL)
async def process_kill(params: dict[str, Any], ctx: ToolContext) -> ToolResult:
    session_id = params.get("process_id") or params.get("session_id", "")
    info = _MANAGER.get(session_id)
    if not info:
        return ToolResult(summary=f"进程不存在: {session_id}", error="ProcessNotFound")
    if info.finished:
        return ToolResult(summary=f"进程 {session_id} 已结束 (exit={info.return_code})", skipped=True)
    try:
        _terminate_info(info)
        await asyncio.sleep(0.1)
        if not info.finished:
            _terminate_info(info, force=True)
        _MANAGER.mark_finished(session_id, -15)
        _MANAGER._persist(info)
        return ToolResult(
            summary=f"已终止进程 {session_id} (pid={info.pid})",
            resource_key=session_id,
            state_delta={"process": "killed"},
            artifact_paths=[p for p in [info.meta_path, info.log_path] if p],
            metadata=tool_metadata(
                "process.kill",
                f"process.kill {session_id} pid={info.pid}",
                session_id=session_id,
                pid=info.pid,
            ),
        )
    except Exception as e:
        info.error = str(e)
        return ToolResult(summary=f"终止失败: {e}", error=str(e))
