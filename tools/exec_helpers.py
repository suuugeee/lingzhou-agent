"""tools/exec_helpers.py - exec/process 内部辅助实现。"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import select
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProcessInfo:
    session_id: str
    command: str
    pid: int | None = None
    started_at: float = 0.0
    finished_at: float | None = None
    return_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    background: bool = False
    finished: bool = False
    timed_out: bool = False
    pty: bool = False
    workdir: str = ""
    timeout_seconds: float | None = None
    proc: Any | None = None
    master_fd: int | None = None
    watch_task: asyncio.Task | None = None
    log_path: str = ""
    meta_path: str = ""
    restored: bool = False
    handle_lost: bool = False
    _output_chunks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        now = time.time()
        duration = max(0.0, now - float(self.started_at or 0.0)) if self.started_at else 0.0
        interaction_available = bool(
            not self.finished and not self.handle_lost and (
                (self.pty and self.master_fd is not None) or (self.proc is not None)
            )
        )
        return {
            "session_id": self.session_id,
            "command": self.command,
            "status": "running" if not self.finished else "finished",
            "pid": self.pid,
            "pty": self.pty,
            "return_code": self.return_code,
            "duration_seconds": round(duration, 1),
            "output_length": len(self.stdout),
            "error": self.error,
            "timed_out": self.timed_out,
            "restored": self.restored,
            "handle_lost": self.handle_lost,
            "interaction_available": interaction_available,
            "meta_path": self.meta_path,
            "log_path": self.log_path,
        }


class ProcessManager:
    """追踪所有通过 exec 启动的进程，并把最小状态持久化到磁盘。"""

    _counter: int = 0
    _processes: dict[str, ProcessInfo] = {}
    _loaded: bool = False

    @classmethod
    def _state_root(cls) -> Path:
        root = Path(os.environ.get("LINGZHOU_PROCESS_STATE_DIR") or (Path.home() / ".lingzhou/state/processes"))
        root.mkdir(parents=True, exist_ok=True)
        return root

    @classmethod
    def _meta_path(cls, session_id: str) -> Path:
        return cls._state_root() / f"{session_id}.json"

    @classmethod
    def _log_path(cls, session_id: str) -> Path:
        return cls._state_root() / f"{session_id}.log"

    @classmethod
    def _pid_alive(cls, pid: int | None) -> bool:
        if not pid:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except Exception:
            return False

    @classmethod
    def _persist(cls, info: ProcessInfo) -> None:
        if not info.meta_path:
            info.meta_path = str(cls._meta_path(info.session_id))
        if not info.log_path:
            info.log_path = str(cls._log_path(info.session_id))
        payload = {
            "session_id": info.session_id,
            "command": info.command,
            "pid": info.pid,
            "started_at": info.started_at,
            "finished_at": info.finished_at,
            "return_code": info.return_code,
            "error": info.error,
            "background": info.background,
            "finished": info.finished,
            "timed_out": info.timed_out,
            "pty": info.pty,
            "workdir": info.workdir,
            "timeout_seconds": info.timeout_seconds,
            "log_path": info.log_path,
            "meta_path": info.meta_path,
            "restored": info.restored,
            "handle_lost": info.handle_lost,
        }
        path = Path(info.meta_path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def _load_stdout(cls, info: ProcessInfo) -> None:
        if not info.log_path:
            return
        path = Path(info.log_path)
        if not path.exists():
            return
        with contextlib.suppress(Exception):
            info.stdout = path.read_text(encoding="utf-8", errors="replace")

    @classmethod
    def _refresh_liveness(cls, info: ProcessInfo) -> None:
        if info.finished:
            return
        if info.proc is not None:
            return
        if info.pid and not cls._pid_alive(info.pid):
            info.finished = True
            info.finished_at = info.finished_at or time.time()
            if info.return_code is None:
                info.return_code = -1
            cls._persist(info)

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._loaded:
            return
        root = cls._state_root()
        for meta in sorted(root.glob("exec-*.json")):
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
            except Exception:
                continue
            sid = str(data.get("session_id") or meta.stem)
            info = ProcessInfo(
                session_id=sid,
                command=str(data.get("command") or ""),
                pid=data.get("pid"),
                started_at=float(data.get("started_at") or 0.0),
                finished_at=data.get("finished_at"),
                return_code=data.get("return_code"),
                error=data.get("error"),
                background=bool(data.get("background", False)),
                finished=bool(data.get("finished", False)),
                timed_out=bool(data.get("timed_out", False)),
                pty=bool(data.get("pty", False)),
                workdir=str(data.get("workdir") or ""),
                timeout_seconds=data.get("timeout_seconds"),
                log_path=str(data.get("log_path") or cls._log_path(sid)),
                meta_path=str(data.get("meta_path") or meta),
                restored=True,
                handle_lost=not bool(data.get("finished", False)),
            )
            cls._load_stdout(info)
            cls._refresh_liveness(info)
            cls._processes[sid] = info
        cls._loaded = True

    @classmethod
    def next_id(cls) -> str:
        cls._ensure_loaded()
        cls._counter += 1
        return f"exec-{int(time.time() * 1000)}-{cls._counter}"

    @classmethod
    def register(cls, info: ProcessInfo) -> str:
        cls._ensure_loaded()
        info.meta_path = str(cls._meta_path(info.session_id))
        info.log_path = str(cls._log_path(info.session_id))
        Path(info.log_path).touch(exist_ok=True)
        cls._processes[info.session_id] = info
        cls._persist(info)
        return info.session_id

    @classmethod
    def get(cls, session_id: str) -> ProcessInfo | None:
        cls._ensure_loaded()
        info = cls._processes.get(session_id)
        if info:
            cls._refresh_liveness(info)
        return info

    @classmethod
    def list_all(cls) -> list[ProcessInfo]:
        cls._ensure_loaded()
        for info in cls._processes.values():
            cls._refresh_liveness(info)
        return list(cls._processes.values())

    @classmethod
    def mark_finished(cls, session_id: str, return_code: int, timed_out: bool = False) -> None:
        cls._ensure_loaded()
        info = cls._processes.get(session_id)
        if info:
            info.finished = True
            info.finished_at = time.time()
            info.return_code = return_code
            info.timed_out = timed_out
            info.handle_lost = False
            cls._persist(info)

    @classmethod
    def clear(cls) -> None:
        for info in list(cls._processes.values()):
            if info.watch_task is not None:
                with contextlib.suppress(Exception):
                    info.watch_task.cancel()
            if info.proc is not None:
                with contextlib.suppress(Exception):
                    _terminate_info(info, force=True)
                with contextlib.suppress(Exception):
                    _close_process_handles(info)
            elif info.pid and cls._pid_alive(info.pid):
                with contextlib.suppress(Exception):
                    os.kill(info.pid, signal.SIGKILL)
        cls._processes.clear()
        cls._counter = 0
        cls._loaded = True
        root = cls._state_root()
        for p in root.glob("exec-*"):
            with contextlib.suppress(Exception):
                p.unlink()


def _append_output(info: ProcessInfo, text: str) -> None:
    if not text:
        return
    info._output_chunks.append(text)
    info.stdout += text
    if info.log_path:
        try:
            with open(info.log_path, "a", encoding="utf-8") as fh:
                fh.write(text)
        except Exception:
            pass


def _preview(text: str, limit: int) -> str:
    return text


def _terminate_info(info: ProcessInfo, *, force: bool = False) -> None:
    proc = info.proc
    try:
        if proc is None:
            return
        if isinstance(proc, asyncio.subprocess.Process):
            if proc.returncode is None:
                # 优先杀进程组，避免子进程残留（例如 bash -lc 拉起的子命令）
                if getattr(proc, "pid", None):
                    with contextlib.suppress(Exception):
                        os.killpg(proc.pid, signal.SIGKILL if force else signal.SIGTERM)
                (proc.kill if force else proc.terminate)()
        elif isinstance(proc, subprocess.Popen):
            if proc.poll() is None:
                if getattr(proc, "pid", None):
                    with contextlib.suppress(Exception):
                        os.killpg(proc.pid, signal.SIGKILL if force else signal.SIGTERM)
                (proc.kill if force else proc.terminate)()
        elif info.pid:
            os.kill(info.pid, signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        pass
    except Exception as e:
        info.error = str(e)


def _close_process_handles(info: ProcessInfo) -> None:
    proc = info.proc
    if proc is None:
        return
    try:
        if isinstance(proc, asyncio.subprocess.Process):
            stdin = getattr(proc, "stdin", None)
            if stdin is not None and not stdin.is_closing():
                stdin.close()
            transport = getattr(proc, "_transport", None)
            if transport is not None:
                transport.close()
        elif isinstance(proc, subprocess.Popen):
            stdin = getattr(proc, "stdin", None)
            if stdin is not None:
                stdin.close()
            stdout = getattr(proc, "stdout", None)
            if stdout is not None:
                stdout.close()
            stderr = getattr(proc, "stderr", None)
            if stderr is not None:
                stderr.close()
    except ProcessLookupError:
        pass
    except Exception:
        pass
    info.proc = None


def _build_capabilities(workdir: str) -> dict[str, Any]:
    common = (
        "python3", "python", "bash", "sh", "grep", "find", "ls", "cat",
        "sqlite3", "git", "sed", "awk", "jq", "rg",
    )
    available = [cmd for cmd in common if shutil.which(cmd)]
    try:
        import pty  # noqa: F401
        has_pty = True
    except Exception:
        has_pty = False
    return {
        "engine": "exec/process runtime",
        "execution_model": "foreground or background",
        "sandbox": False,
        "network_policy": "inherits-host-environment",
        "default_timeout_sec": 30,
        "default_output_preview_chars": 500,
        "workdir": workdir,
        "shell": os.environ.get("SHELL") or "/bin/sh",
        "available_commands": available,
        "has_background_exec": True,
        "has_process_management": True,
        "has_pty": has_pty,
        "has_process_write": True,
    }


def _spawn_pty_process(command: str, workdir: str, env: dict[str, str]) -> tuple[subprocess.Popen[Any], int]:
    import pty

    master_fd, slave_fd = pty.openpty()
    os.set_blocking(master_fd, False)
    proc = subprocess.Popen(
        [os.environ.get("SHELL") or "/bin/bash", "-lc", command],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        cwd=workdir,
        env=env,
        close_fds=True,
        start_new_session=True,  # 让 killpg 能终止整组子进程，避免 orphan
    )
    os.close(slave_fd)
    return proc, master_fd


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
        ProcessManager.mark_finished(info.session_id, -1, timed_out=True)
        ProcessManager._persist(info)
    except Exception as e:
        info.error = str(e)
        ProcessManager.mark_finished(info.session_id, -1)
        ProcessManager._persist(info)
    else:
        ProcessManager.mark_finished(info.session_id, proc.returncode if proc.returncode is not None else -1)
        ProcessManager._persist(info)
    finally:
        try:
            await asyncio.wait_for(reader_task, timeout=1.0)
        except Exception:
            reader_task.cancel()
        _close_process_handles(info)


def _run_pty_until_exit(info: ProcessInfo) -> tuple[int, bool, str | None]:
    proc = info.proc
    master_fd = info.master_fd
    assert proc is not None and master_fd is not None

    timed_out = False
    err: str | None = None
    start = time.time()
    try:
        while True:
            if info.timeout_seconds and (time.time() - start) > info.timeout_seconds and proc.poll() is None:
                proc.terminate()
                time.sleep(0.1)
                if proc.poll() is None:
                    proc.kill()
                timed_out = True

            try:
                ready, _, _ = select.select([master_fd], [], [], 0.1)
            except (OSError, ValueError):
                ready = []

            if ready:
                try:
                    chunk = os.read(master_fd, 1024)
                    if chunk:
                        _append_output(info, chunk.decode(errors="replace"))
                except BlockingIOError:
                    pass
                except OSError:
                    pass

            rc = proc.poll()
            if rc is not None:
                for _ in range(5):
                    try:
                        chunk = os.read(master_fd, 1024)
                        if not chunk:
                            break
                        _append_output(info, chunk.decode(errors="replace"))
                    except Exception:
                        break
                return rc, timed_out, err
    except Exception as e:
        err = str(e)
        return -1, timed_out, err
    finally:
        with contextlib.suppress(Exception):
            os.close(master_fd)
        info.master_fd = None


async def _watch_pty_process(info: ProcessInfo) -> None:
    rc, timed_out, err = await asyncio.to_thread(_run_pty_until_exit, info)
    if err:
        info.error = err
    info.proc = None
    ProcessManager.mark_finished(info.session_id, rc, timed_out=timed_out)
    ProcessManager._persist(info)
