"""cli/gateway.py — run / gateway 命令组（消息网关与认知循环启动）。"""
from __future__ import annotations

import asyncio
import fcntl
import json as _json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from channels import describe_channel_runtime, start_channel_runtime
from cli.bootstrap import onboarding_status
from cli.common import DEFAULT_CONFIG_PATH, console, load_cfg
from cli.logs import logs_crash, logs_errors, logs_files, logs_stats, logs_tail, logs_wechat
from cli.plugin import plugin_app

_PID_FILE = Path("~/.lingzhou/lingzhou.pid").expanduser()
_LOCK_FILE = Path("~/.lingzhou/lingzhou.lock").expanduser()
_LOCK_FD: int | None = None  # 保持打开的锁文件描述符，进程退出时自动释放锁


def _startup_config_log_line(
    cfg: Any,
    requested_config: Path,
    *,
    channel: str,
    daemon: bool,
) -> str:
    requested = requested_config.expanduser().resolve()
    base_dir = Path(getattr(cfg, "_base_dir", requested.parent)).expanduser().resolve()
    effective = (base_dir / "lingzhou.json").resolve()
    routing = getattr(cfg, "routing", {}) or {}
    routing_items = ", ".join(
        f"{tier}={model_ref}" for tier, model_ref in sorted(routing.items())
    ) if routing else "(none)"
    from core.log_fields import format_log_fields

    return "[startup] " + format_log_fields(
        channel=channel,
        daemon=daemon,
        requested_config=str(requested),
        effective_config=str(effective),
        model_ref=getattr(cfg, "model", ""),
        routing=routing_items,
    )


def _restart_mode_log_line(
    requested_config: Path,
    *,
    mode: str,
    channel: str | None,
) -> str:
    requested = requested_config.expanduser().resolve()
    return (
        "[restart] "
        f"mode={mode} "
        f"channel={channel or '(auto)'} "
        f"requested_config={requested}"
    )


def _load_lingzhou_dotenv() -> None:
    """加载 ~/.lingzhou/.env，让 daemon 与前台启动使用同一套凭证环境。"""
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]

        load_dotenv(Path("~/.lingzhou/.env").expanduser())
    except Exception:
        pass


def _provider_names_required_for_startup(cfg: Any) -> list[str]:
    providers = getattr(cfg, "providers", None)
    if not isinstance(providers, dict) or not providers:
        return []

    names: list[str] = []

    def _add_model_ref(model_ref: Any) -> None:
        ref = str(model_ref or "").strip()
        if "/" not in ref:
            return
        provider_name = ref.split("/", 1)[0].strip()
        if provider_name and provider_name not in names:
            names.append(provider_name)

    _add_model_ref(getattr(cfg, "model", ""))
    return names


def _gateway_provider_preflight_error(cfg: Any) -> str | None:
    """启动前检查主模型凭证，避免 daemon fork 后才 traceback。

    routing/model_fallbacks 不是启动必需路径：缺少 token 时运行时会跳过并回退主模型。
    """
    providers = getattr(cfg, "providers", None)
    if not isinstance(providers, dict) or not providers:
        return None

    errors: list[str] = []
    for provider_name in _provider_names_required_for_startup(cfg):
        provider = providers.get(provider_name)
        if provider is None:
            errors.append(
                f"provider {provider_name!r} 未在 providers 中定义。"
            )
            continue

        api_key_env = str(getattr(provider, "api_key_env", "") or "").strip()
        if getattr(provider, "mode", "") == "copilot":
            from store.auth import resolve_copilot_token

            if not resolve_copilot_token(api_key_env):
                errors.append(
                    f"provider {provider_name!r} 缺少 Copilot token。"
                    "请运行: lingzhou auth login-copilot"
                )
            continue

        if getattr(provider, "mode", "") == "codex":
            from provider.codex_oauth import resolve_codex_oauth_token

            auth_profile_id = str(getattr(provider, "auth_profile_id", "") or "").strip()
            resolved = (
                resolve_codex_oauth_token(profile_id=auth_profile_id)
                if auth_profile_id
                else resolve_codex_oauth_token()
            )
            if not resolved or not resolved.token:
                errors.append(
                    f"provider {provider_name!r} 缺少 Codex OAuth token。"
                    "请运行: lingzhou auth login-codex"
                )
            continue

        try:
            _ = provider.api_key
        except OSError as exc:
            errors.append(f"provider {provider_name!r} 凭证不可用: {exc}")

    if not errors:
        return None
    return "\n".join(errors)


def _ensure_singleton() -> None:
    """保证唯一实例。通过 flock 独占锁，与 wrapper.sh 共用同一个锁文件。

    如果已有实例持有锁，立即退出并提示。
    锁会在进程退出时自动释放（无论正常退出还是崩溃）。
    """
    global _LOCK_FD
    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock_fd: int | None = None
    try:
        lock_fd = os.open(_LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # 写入当前 PID 到锁文件（调试用）
        os.lseek(lock_fd, 0, os.SEEK_SET)
        os.ftruncate(lock_fd, 0)
        os.write(lock_fd, f"{os.getpid()}\n".encode())
        os.fsync(lock_fd)
        _LOCK_FD = lock_fd
    except BlockingIOError:
        try:
            holder = "未知"
            if lock_fd is not None:
                os.lseek(lock_fd, 0, os.SEEK_SET)
                holder = os.read(lock_fd, 1024).decode("utf-8", errors="ignore").strip()
                os.close(lock_fd)
                lock_fd = None
        except Exception:
            holder = "未知"
        console.print(f"[red]✗ lingzhou 已在运行[/red]  （锁文件: {_LOCK_FILE}）")
        if holder:
            console.print(f"  [dim]占用进程 PID: {holder}[/dim]")
        console.print("  [dim]停止: lingzhou gateway stop[/dim]")
        raise typer.Exit(1) from None
    except Exception as e:
        if lock_fd is not None:
            os.close(lock_fd)
            lock_fd = None
        console.print(f"[red] 获取锁失败: {e}[/red]")
        raise typer.Exit(1) from None


def _configure_lingzhou_logging(
    log_dir: Path,
    log_level: int,
    *,
    logger_name: str = "lingzhou",
) -> tuple[Path, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"lingzhou-{datetime.now().strftime('%Y-%m-%d')}.log"
    console_log_file = log_dir / "console.log"

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.propagate = False

    keep_handlers: list[logging.Handler] = []
    for handler in logger.handlers:
        base = getattr(handler, "baseFilename", "")
        if base in {str(log_file), str(console_log_file)}:
            handler.close()
            continue
        keep_handlers.append(handler)
    logger.handlers = keep_handlers

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(message)s", datefmt="%H:%M:%S"))
    file_handler.setLevel(log_level)

    console_handler = logging.FileHandler(console_log_file, mode="w", encoding="utf-8")
    console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s [%(name)s] %(message)s", datefmt="%H:%M:%S"))
    console_handler.setLevel(log_level)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return log_file, console_log_file


def _daemonize(argv: list[str]) -> None:
    """fork 子进程后台运行，父进程立即退出。"""
    pid = os.fork()
    if pid > 0:
        # 父进程：打印 PID 后退出
        console.print(f"[green]✓ 已后台启动[/green]  PID={pid}  日志: ~/.lingzhou/logs/")
        console.print("  停止: [bold]lingzhou stop[/bold]")
        raise typer.Exit(0)
    # 子进程：脱离终端
    os.setsid()
    # stdin → /dev/null, stdout/stderr → 崩溃日志（关键：保留 traceback）
    log_dir = Path("~/.lingzhou/logs").expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    crash_log = os.open(str(log_dir / "crash.log"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    tick_log = os.open(str(log_dir / "daemon-stdout.log"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)     # stdin → /dev/null
    os.dup2(tick_log, 1)    # stdout → daemon-stdout.log（tick 快照等）
    os.dup2(crash_log, 2)   # stderr → crash.log（只放真正的异常栈）
    os.close(crash_log)
    os.close(tick_log)
    os.close(devnull)
    # 写 PID
    _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text(str(os.getpid()), encoding="utf-8")


# channel名称 → (描述, 是否需要 setup 配置)
_GATEWAY_CHANNELS: dict[str, tuple[str, bool]] = {
    "local":    ("本地终端 — 直接在当前终端运行，无需额外配置", False),
    "webhook":  ("HTTP Webhook — 对外暴露 /message 端点（适合集成其他系统）", True),
    "telegram": ("Telegram Bot — 需要 BOT_TOKEN", True),
    "wechat":   ("微信公众号 / 企业微信 — 开发中", True),
    "qq":       ("QQ Bot — 开发中", True),
}

# 已实现的渠道
_GATEWAY_READY = {"local", "webhook", "wechat"}

gateway_app = typer.Typer(name="gateway", help="消息网关（Telegram、Webhook 等）", no_args_is_help=True, context_settings={"help_option_names": ["-h", "--help"]})

# ── 日志子命令（注册在 gateway 下）─────────────────────────────────────────────
logs_group = typer.Typer(name="logs", invoke_without_command=True, help="快速查看运行日志（直接调用: tail -f）")


@logs_group.callback()
def _logs_default(ctx: typer.Context) -> None:
    """无子命令时默认执行 tail -f（持续输出最新日志）。"""
    if ctx.invoked_subcommand is None:
        logs_tail(follow=True)


logs_group.command("tail")(logs_tail)
logs_group.command("errors")(logs_errors)
logs_group.command("crash")(logs_crash)
logs_group.command("wechat")(logs_wechat)
logs_group.command("files")(logs_files)
logs_group.command("stats")(logs_stats)
gateway_app.add_typer(logs_group)
gateway_app.add_typer(plugin_app)


def _kill_existing_loop(quiet: bool = False) -> None:
    """杀掉 PID 文件中记录的旧进程，等待它真正退出（最长 8s）。"""
    import time as _t
    if not _PID_FILE.exists():
        return
    pid_str = _PID_FILE.read_text(encoding="utf-8").strip()
    _PID_FILE.unlink(missing_ok=True)
    try:
        old_pid = int(pid_str)
        os.kill(old_pid, 0)  # 先检查进程是否存在
    except (ProcessLookupError, ValueError, PermissionError):
        return  # 进程已不存在，跳过

    if not quiet:
        console.print(f"[yellow]正在关闭旧进程[/yellow]  PID={old_pid} …")
    try:
        os.kill(old_pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    # 等待进程退出（最长 8s），避免新进程与旧进程 overlap
    deadline = _t.monotonic() + 8.0
    while _t.monotonic() < deadline:
        _t.sleep(0.3)
        try:
            os.kill(old_pid, 0)
        except ProcessLookupError:
            if not quiet:
                console.print(f"[green]旧进程已退出[/green]  PID={old_pid}")
            return

    # 超时：强制 SIGKILL
    try:
        os.kill(old_pid, signal.SIGKILL)
        if not quiet:
            console.print(f"[yellow]旧进程未响应 SIGTERM，已发送 SIGKILL[/yellow]  PID={old_pid}")
    except ProcessLookupError:
        pass


@gateway_app.command("channels")
def gateway_channels() -> None:
    """列出支持的消息渠道。"""
    console.print("[bold]支持的消息渠道[/bold]\n")
    for ch, (desc, needs_setup) in _GATEWAY_CHANNELS.items():
        if ch in _GATEWAY_READY:
            status = "[green]可用[/green]"
            setup_hint = f"  [dim]lingzhou gateway setup --channel {ch}[/dim]" if needs_setup else ""
        else:
            status = "[dim]开发中[/dim]"
            setup_hint = ""
        console.print(f"  [cyan]{ch:<10}[/cyan] {status}  {desc}{setup_hint}")
    console.print(
        "\n[dim]启动: lingzhou gateway start --channel <name>[/dim]\n"
        "[dim]配置: lingzhou gateway setup --channel <name>[/dim]"
    )


@gateway_app.command("setup")
def gateway_setup(
    channel: Annotated[str, typer.Option("--channel", "-ch", help="渠道名称")] = "webhook",
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
) -> None:
    """配置消息渠道（向导模式）。local 渠道无需配置。"""
    from rich.panel import Panel

    if channel not in _GATEWAY_CHANNELS:
        console.print(f"[red]未知渠道: {channel}。支持: {', '.join(_GATEWAY_CHANNELS)}[/red]")
        raise typer.Exit(1)

    if channel == "local":
        console.print("[green]local 渠道无需配置，直接运行: lingzhou[/green]")
        return

    if channel not in _GATEWAY_READY:
        console.print(f"[yellow]{channel} 渠道尚在开发中，暂不支持配置。[/yellow]")
        raise typer.Exit(1)

    gw_dir = Path("~/.lingzhou/gateway").expanduser()
    gw_dir.mkdir(parents=True, exist_ok=True)
    gw_cfg_path = gw_dir / f"{channel}.json"

    console.print(Panel(
        f"[bold]网关配置向导[/bold]  渠道: [cyan]{channel}[/cyan]",
        border_style="blue",
    ))

    if channel == "telegram":
        console.print("\n  获取 Bot Token: 与 @BotFather 对话 → /newbot → 复制 token")
        token = typer.prompt("  BOT_TOKEN").strip()
        allowed_raw = typer.prompt("  允许的用户 ID（逗号分隔，留空则允许所有人）", default="").strip()
        allowed = [int(x.strip()) for x in allowed_raw.split(",") if x.strip().isdigit()]
        gw_conf: dict[str, Any] = {"channel": "telegram", "bot_token": token, "allowed_user_ids": allowed}
        gw_cfg_path.write_text(_json.dumps(gw_conf, ensure_ascii=False, indent=2), encoding="utf-8")
        gw_cfg_path.chmod(0o600)
        console.print(f"\n[green]✓ Telegram 网关配置已保存: {gw_cfg_path}[/green]")

    elif channel == "wechat":
        console.print("\n  iLink Bot Token（微信开放平台 → bot 管理 → 复制 Token）")
        token = typer.prompt("  ILINK_TOKEN").strip()
        base_url = typer.prompt("  iLink API 地址", default="https://ilinkai.weixin.qq.com").strip()
        console.print("  [dim]若使用 hermesclaw 等代理轮询 iLink，填代理地址（如 http://127.0.0.1:19997）。直连 iLink 请留空。[/dim]")
        poll_base_url = typer.prompt("  轮询代理地址 poll_base_url", default="").strip()
        poll_sec = typer.prompt("  长轮询超时（秒）", default="35")
        gw_conf: dict[str, Any] = {
            "channel": "wechat",
            "token": token,
            "base_url": base_url,
            "poll_base_url": poll_base_url,
            "poll_sec": int(poll_sec),
            "reply_poll_sec": 3,
        }
        gw_cfg_path.write_text(_json.dumps(gw_conf, ensure_ascii=False, indent=2), encoding="utf-8")
        gw_cfg_path.chmod(0o600)
        console.print(f"\n[green]✓ 微信网关配置已保存: {gw_cfg_path}[/green]")

    elif channel == "webhook":
        host = typer.prompt("  监听地址", default="0.0.0.0")
        port = int(typer.prompt("  监听端口", default="8765"))
        secret = typer.prompt("  共享 secret（留空则无鉴权）", default="").strip() or None
        gw_conf: dict[str, Any] = {"channel": "webhook", "host": host, "port": port, "secret": secret}
        gw_cfg_path.write_text(_json.dumps(gw_conf, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"\n[green]✓ Webhook 网关配置已保存: {gw_cfg_path}[/green]")

    console.print(f"\n  启动: [bold]lingzhou gateway start --channel {channel}[/bold]")


@gateway_app.command("stop")
def gateway_stop() -> None:
    """停止后台运行的认知循环。"""
    if not _PID_FILE.exists():
        console.print("[yellow]未找到运行中的 lingzhou 进程（~/.lingzhou/lingzhou.pid 不存在）[/yellow]")
        raise typer.Exit(1)
    pid_str = _PID_FILE.read_text(encoding="utf-8").strip()
    try:
        pid = int(pid_str)
        os.kill(pid, signal.SIGTERM)
        _PID_FILE.unlink(missing_ok=True)
        console.print(f"[green]✓ 已发送停止信号[/green]  PID={pid}")
    except ProcessLookupError:
        console.print(f"[yellow]进程 {pid_str} 已不存在，清理 PID 文件[/yellow]")
        _PID_FILE.unlink(missing_ok=True)
        raise typer.Exit(1) from None
    except ValueError:
        console.print(f"[red]PID 文件内容无效: {pid_str!r}[/red]")


@gateway_app.command("status")
def gateway_status() -> None:
    """查看认知循环运行状态。"""
    import time as _t

    if not _PID_FILE.exists():
        console.print("[yellow]● 未运行[/yellow]  (PID 文件不存在)")
        raise typer.Exit(1)

    pid_str = _PID_FILE.read_text(encoding="utf-8").strip()
    try:
        pid = int(pid_str)
        os.kill(pid, 0)  # 仅探测进程是否存在，不发送真实信号
    except ProcessLookupError:
        console.print(f"[yellow]● 进程已退出[/yellow]  PID={pid_str}（PID 文件残留，可运行 lingzhou gateway stop 清理）")
        raise typer.Exit(1) from None
    except ValueError:
        console.print(f"[red]PID 文件内容无效: {pid_str!r}[/red]")
        raise typer.Exit(1) from None

    # 用 PID 文件 mtime 估算运行时长
    try:
        mtime = _PID_FILE.stat().st_mtime
        uptime_s = int(_t.time() - mtime)
        h, rem = divmod(uptime_s, 3600)
        m, s = divmod(rem, 60)
        uptime_str = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"
    except OSError:
        uptime_str = "未知"

    console.print(f"[green]● 运行中[/green]  PID={pid}  已运行 {uptime_str}")
    console.print("  日志: [dim]~/.lingzhou/logs/[/dim]")
    console.print("  停止: [dim]lingzhou gateway stop[/dim]")
    console.print("  重启: [dim]lingzhou gateway restart[/dim]")


def _is_systemd_managed() -> bool:
    """检查 lingzhou 是否由 systemd 管理。

    不仅检查 active 状态，还检查是否 enabled。
    这样即使服务正在重启或失败，也能检测到 systemd 管理。
    """
    try:
        # 先检查是否 enabled（持久配置）
        r = subprocess.run(
            ["systemctl", "is-enabled", "lingzhou.service"],
            capture_output=True, text=True, timeout=5
        )
        if r.stdout.strip() in ("enabled", "static"):
            return True
        # 再检查 active 状态（即使没 enabled，正在运行也算）
        r = subprocess.run(
            ["systemctl", "is-active", "lingzhou.service"],
            capture_output=True, text=True, timeout=5
        )
        return r.stdout.strip() in ("active", "activating")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _restart_via_systemd() -> bool:
    """通过 systemd 重启 lingzhou。返回 True 表示成功。

    systemd 本身保证 restart 的原子性，不需要额外的 flock。
    """
    try:
        console.print("[dim]检测到 systemd 管理，通过 systemctl 重启...[/dim]")
        subprocess.run(["systemctl", "restart", "lingzhou.service"], check=True, timeout=10)
        console.print("[green]✓ systemd 重启成功[/green]")
        return True
    except Exception as e:
        console.print(f"[yellow]systemd 重启失败: {e}，回退到 PID 文件模式[/yellow]")
        return False


@gateway_app.command("restart")
def gateway_restart(
    channel: Annotated[str | None, typer.Option("--channel", "-ch", help="消息渠道（默认从 lingzhou.json gateway.default_channel 读取）")] = None,
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    debug: Annotated[bool | None, typer.Option("--debug/--no-debug")] = None,
    dry_run: Annotated[bool | None, typer.Option("--dry-run/--act")] = None,
) -> None:
    """重启认知循环（stop + start）。

    优先通过 systemd 重启（如果 lingzhou 由 systemd 管理），
    避免 systemd 与 PID 文件双管理导致多实例竞争。
    """
    if _is_systemd_managed():
        console.print(f"[dim]{_restart_mode_log_line(config, mode='systemd', channel=channel)}[/dim]")
        if _restart_via_systemd():
            return  # systemd 已接管，不需要后续操作
    # 非 systemd 管理：使用原有的 PID 文件模式
    console.print(f"[dim]{_restart_mode_log_line(config, mode='pid', channel=channel)}[/dim]")
    _kill_existing_loop(quiet=False)
    gateway_start(channel=channel, config=config, debug=debug, dry_run=dry_run, daemon=True)


@gateway_app.command("start")
def gateway_start(
    channel: Annotated[str | None, typer.Option("--channel", "-ch", help="消息渠道（默认从 lingzhou.json gateway.default_channel 读取）")] = None,
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    debug: Annotated[bool | None, typer.Option("--debug/--no-debug")] = None,
    dry_run: Annotated[bool | None, typer.Option("--dry-run/--act")] = None,
    daemon: Annotated[bool, typer.Option("--daemon/--no-daemon", "-d/-f", help="后台运行，默认已开启；--no-daemon 前台运行")] = True,
) -> None:
    """启动认知循环 + 消息渠道（loop 是内核，channel 是 I/O 层）。

    local    — 本地终端，无需配置，直接运行
    webhook  — HTTP 接入，loop 与 webhook server 并行
    wechat   — 微信 iLink 通道
    telegram — Telegram Bot（开发中）

    如果检测到 systemd 管理且非 wrapper 调用，自动重定向到 systemctl restart。
    """
    # 检测是否已经运行在 systemd unit 内（INVOCATION_ID 由 systemd 启动时自动注入）
    # 或在 wrapper 脚本内调用（兼容旧 wrapper）
    # 两种情况都不应再重定向到 systemctl，否则会造成无限重启循环
    is_inside_systemd_unit = bool(os.environ.get("INVOCATION_ID"))
    is_inside_wrapper = os.environ.get("LINGZHOU_WRAPPER", "") == "1"

    # 如果 systemd 在管理且当前进程不是由 systemd/wrapper 直接启动，重定向到 systemctl
    if daemon and _is_systemd_managed() and not is_inside_wrapper and not is_inside_systemd_unit:
        console.print("[dim]检测到 systemd 管理，使用 systemctl restart...[/dim]")
        _restart_via_systemd()
        return

    ready, reason = onboarding_status(config)
    if not ready:
        console.print(f"[yellow]{reason}[/yellow]")
        console.print("[dim]先运行: lingzhou onboard[/dim]")
        raise typer.Exit(1)

    cfg = load_cfg(config)
    if channel is None:
        channel = str(cfg.gateway.default_channel or "local").strip() or "local"

    if channel not in _GATEWAY_CHANNELS:
        console.print(f"[yellow]{channel} 渠道尚在开发中。当前可用: {', '.join(_GATEWAY_READY)}[/yellow]")
        raise typer.Exit(1)

    _load_lingzhou_dotenv()
    provider_error = _gateway_provider_preflight_error(cfg)
    if provider_error:
        console.print(f"[red]Provider 凭证不可用，启动已停止。[/red]\n{provider_error}")
        console.print("[dim]可运行 `lingzhou dev doctor` 查看完整诊断。[/dim]")
        raise typer.Exit(1)

    if daemon and hasattr(os, "fork"):
        # 后台模式：先杀旧进程（释放锁），再获取锁 fork
        _kill_existing_loop(quiet=False)
        _ensure_singleton()
        _daemonize(sys.argv)
        # 子进程从这里继续执行（父进程已退出）
    else:
        # 前台模式：先杀旧进程，再获取锁
        _kill_existing_loop(quiet=True)
        _ensure_singleton()
        # 前台模式也写 PID，防止并发启动
        _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PID_FILE.write_text(str(os.getpid()), encoding="utf-8")

    # 非 local 渠道需要提前 setup
    gw_conf: dict[str, Any] = {}
    if channel != "local":
        gw_cfg_path = Path("~/.lingzhou/gateway").expanduser() / f"{channel}.json"
        if not gw_cfg_path.exists():
            console.print(
                f"[red]渠道 {channel!r} 尚未配置，请先运行: "
                f"lingzhou gateway setup --channel {channel}[/red]"
            )
            raise typer.Exit(1)
        gw_conf = _json.loads(gw_cfg_path.read_text(encoding="utf-8"))

    if debug is not None:
        cfg.loop.debug = debug
    if dry_run is not None:
        cfg.loop.act = not dry_run

    # 日志
    log_dir = Path(cfg.logging.dir).expanduser()
    log_level = logging.DEBUG if (debug or cfg.loop.debug) else logging.INFO
    log_file, console_log_file = _configure_lingzhou_logging(log_dir, log_level)
    console.print(f"[dim]渠道: [cyan]{channel}[/cyan]  日志: {log_file}  console: {console_log_file}[/dim]")
    logging.getLogger("lingzhou.gateway").info(
        _startup_config_log_line(cfg, config, channel=channel, daemon=daemon)
    )

    from core.loop import CognitionLoop
    gateway_logger = logging.getLogger("lingzhou.gateway")
    construct_started = time.monotonic()
    gateway_logger.info("[startup] loop construct start")
    loop_instance = CognitionLoop(cfg)
    gateway_logger.info("[startup] loop construct done dt=%.3fs", time.monotonic() - construct_started)
    if channel != "local":
        # 将 Config.gateway 的默认值注入 gw_conf（json 文件已有的 key 优先）
        if channel == "webhook":
            gw_conf.setdefault("host", cfg.gateway.webhook_host)
            gw_conf.setdefault("port", cfg.gateway.webhook_port)

        def _start_channel_when_ready() -> None:
            _start_external_channel_runtime(channel, gw_conf, db_path=cfg.db_path)

        loop_instance._runtime_ready_callback = _start_channel_when_ready
    try:
        gateway_logger.info("[startup] loop.run enter")
        asyncio.run(loop_instance.run())
    except KeyboardInterrupt:
        console.print("\n[dim]认知循环已停止。[/dim]")
    finally:
        # 前台模式：退出时清理 PID 文件
        if _PID_FILE.exists():
            try:
                if _PID_FILE.read_text(encoding="utf-8").strip() == str(os.getpid()):
                    _PID_FILE.unlink(missing_ok=True)
            except OSError:
                pass


def run(
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    channel: Annotated[str | None, typer.Option("--channel", "-ch", help="消息渠道（默认从 lingzhou.json gateway.default_channel 读取）")] = None,
    debug: Annotated[bool | None, typer.Option("--debug/--no-debug")] = None,
    dry_run: Annotated[bool | None, typer.Option("--dry-run/--act")] = None,
    daemon: Annotated[bool, typer.Option("--daemon/--no-daemon", "-d/-f", help="后台运行，默认已开启；--no-daemon 前台运行")] = True,
) -> None:
    """启动认知循环（等同于 gateway start）。"""
    gateway_start(channel=channel, config=config, debug=debug, dry_run=dry_run, daemon=daemon)


def stop() -> None:
    """停止后台运行的认知循环。"""
    if not _PID_FILE.exists():
        console.print("[yellow]未找到运行中的 lingzhou 进程（~/.lingzhou/lingzhou.pid 不存在）[/yellow]")
        raise typer.Exit(1)
    pid_str = _PID_FILE.read_text(encoding="utf-8").strip()
    try:
        pid = int(pid_str)
        os.kill(pid, signal.SIGTERM)
        _PID_FILE.unlink(missing_ok=True)
        console.print(f"[green]✓ 已发送停止信号[/green]  PID={pid}")
    except ProcessLookupError:
        console.print(f"[yellow]进程 {pid_str} 已不存在，清理 PID 文件[/yellow]")
        _PID_FILE.unlink(missing_ok=True)
        raise typer.Exit(1) from None
    except ValueError:
        console.print(f"[red]PID 文件内容无效: {pid_str!r}[/red]")
        raise typer.Exit(1) from None


def _start_external_channel_runtime(channel: str, gw_conf: dict[str, Any], *, db_path: str | Path) -> object:
    console.print(f"[dim]{describe_channel_runtime(channel, gw_conf)}[/dim]")
    return start_channel_runtime(channel, gw_conf, db_path)
