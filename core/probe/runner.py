"""core/probe/runner.py — 探针调度与执行引擎。

调度模型（参考 Prometheus scrape_interval）：
- interval:<N>  — 每 N 秒执行一次，独立异步 Task
- manual        — 仅在 probe.run 工具显式调用时执行

数据回传：
- wm            — interval 探针后台周期执行时推入 WorkingMemory，下一轮 tick LLM 可见
- none          — 仅记日志，不自动推送（LLM 通过 probe.run 主动获取）

LLM 可对 probe.run 返回结果自行决定如何处置。
"""
from __future__ import annotations

import ast
import asyncio
import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from core.contracts.probe import ProbeConfig, ProbeResult

from .executor import execute_probe

if TYPE_CHECKING:
    from .store import ProbeStore

_log = logging.getLogger("lingzhou.probe")

# 数据回传到 WM 的优先级
_WM_PRIORITY = 0.72
# 告警消息回传到 WM 的优先级
_ALERT_WM_PRIORITY = 0.90
_CONFIDENCE_WARN_PRIORITY = 0.86


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


_SAFE_ALERT_NODES = (
    ast.Expression, ast.BoolOp, ast.UnaryOp, ast.Compare,
    ast.And, ast.Or, ast.Not, ast.In, ast.NotIn,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Constant, ast.Name, ast.Load,
)


def _safe_eval_alert(expr: str, output: str) -> bool:
    """仅允许纯比较/成员/布尔表达式，拒绝 Attribute/Call/函数调用等危险节点。"""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_ALERT_NODES):
            raise ValueError(f"alert_expr 含不允许的节点: {type(node).__name__}")
    return bool(eval(compile(tree, "<alert_expr>", "eval"), {"output": output}, {}))


def _evaluate_alert(cfg: ProbeConfig, output: str) -> tuple[bool, str]:
    """执行 alert_expr，返回 (triggered, detail)。"""
    if not cfg.alert_expr:
        return False, ""
    try:
        triggered = _safe_eval_alert(cfg.alert_expr, output)
        if triggered:
            msg = (cfg.alert_message or f"[探针告警] {cfg.name}: {output}").replace(
                "{output}", output
            )
            return True, msg
    except Exception as exc:
        _log.debug("[probe] alert_expr 评估失败 probe=%s: %s", cfg.name, exc)
    return False, ""


def _assess_confidence(cfg: ProbeConfig, output: str, error: str | None, duration_ms: int) -> tuple[float, str, bool]:
    """评估本次探针读数可信度。返回 (score, reason, suspect_setup)。"""
    reasons: list[str] = []
    score = 0.85
    suspect_setup = False

    if error:
        score = 0.2
        reasons.append("执行报错，结果可信度低")
        lowered = error.lower()
        setup_markers = (
            "command not found",
            "no such file",
            "connection refused",
            "name or service not known",
            "could not resolve",
            "invalid url",
            "timed out",
            "timeout",
        )
        if any(marker in lowered for marker in setup_markers):
            suspect_setup = True
            reasons.append("疑似探针布放或目标地址配置问题")
        return max(0.0, min(1.0, score)), "；".join(reasons), suspect_setup

    text = (output or "").strip()
    if not text:
        score -= 0.35
        reasons.append("无有效输出")

    if duration_ms <= 5:
        score -= 0.1
        reasons.append("执行过快，需确认不是空跑")

    if cfg.kind == "shell":
        spec = (cfg.spec or "").lower()
        if "curl" in spec and "%{http_code}" in spec:
            # 健康检查常见形态：输出应是 3 位状态码（如 200）
            code_match = re.search(r"\b(\d{3})\b", text)
            if not code_match:
                score -= 0.4
                suspect_setup = True
                reasons.append("未解析到 HTTP 状态码，探针命令可能布放不当")
            else:
                code = int(code_match.group(1))
                if not (100 <= code <= 599):
                    score -= 0.35
                    suspect_setup = True
                    reasons.append("状态码越界，探针输出不符合预期")
                elif code >= 500:
                    score -= 0.3
                    reasons.append("返回 5xx，先复核探针目标/布放，再确认是否真实服务故障")

    if cfg.kind == "http" and text and len(text) < 2:
        score -= 0.1
        reasons.append("HTTP 响应过短，建议复核采样目标")

    score = max(0.0, min(1.0, score))
    if not reasons:
        reasons.append("读数形态与执行状态正常")
    return score, "；".join(reasons), suspect_setup


class ProbeRunner:
    """管理所有运行中探针的调度任务。

    由 ProbeManager 持有。loop._probe_manager.runner 可访问。
    """

    def __init__(self, store: ProbeStore) -> None:
        self._store = store
        self._tasks: dict[str, asyncio.Task[None]] = {}
        # 由 ProbeManager 在启动后注入（避免循环依赖）
        self._wm: Any = None
        self._loop_ref: Any = None
        self._alert_event: asyncio.Event | None = None

    def attach(self, wm: Any, loop_ref: Any | None = None) -> None:
        """注入运行时依赖（WM / Loop 引用）。"""
        self._wm = wm
        self._loop_ref = loop_ref
        self._alert_event = asyncio.Event()
        self._loop_ref = loop_ref

    async def start_all(self) -> None:
        """从数据库加载所有启用的探针，启动调度任务。"""
        probes = await self._store.list_all(enabled_only=True)
        for cfg in probes:
            self._schedule(cfg)
        _log.info("[probe] runner started, %d probe(s) loaded", len(probes))

    def _schedule(self, cfg: ProbeConfig) -> None:
        """为探针启动一个异步调度 Task（如需）。"""
        # 已有同名 Task 且未结束，先取消
        existing = self._tasks.get(cfg.name)
        if existing and not existing.done():
            existing.cancel()

        if not cfg.enabled:
            return

        trigger = (cfg.trigger or "").strip().lower()
        if trigger == "manual":
            return  # 仅手动触发，不建调度

        if trigger.startswith("interval:"):
            try:
                interval = int(trigger.split(":", 1)[1])
            except (ValueError, IndexError):
                _log.warning("[probe] 无效 trigger 格式: %s，跳过调度", cfg.trigger)
                return
            task = asyncio.create_task(
                self._interval_loop(cfg, interval),
                name=f"probe:{cfg.name}",
            )
            self._tasks[cfg.name] = task
        else:
            _log.warning("[probe] 不支持的 trigger 格式: %s（仅支持 interval:<s> 或 manual）", cfg.trigger)

    def unschedule(self, name: str) -> None:
        """停止指定探针的调度 Task。"""
        task = self._tasks.pop(name, None)
        if task and not task.done():
            task.cancel()

    async def run_now(self, cfg: ProbeConfig) -> ProbeResult:
        """立即执行探针并回传数据（probe.run 工具入口）。"""
        return await self._execute(cfg)

    async def _interval_loop(self, cfg: ProbeConfig, interval: int) -> None:
        """定时循环：先等一个间隔再执行（避免启动时立即打扰 LLM）。"""
        await asyncio.sleep(interval)
        while True:
            # 每次运行时重新从 DB 取最新配置（用户可能修改了）
            latest = await self._store.get(cfg.name)
            if latest is None or not latest.enabled:
                _log.info("[probe] 探针 %s 已被删除或禁用，停止调度", cfg.name)
                return
            await self._execute(latest)
            await asyncio.sleep(interval)

    async def _execute(self, cfg: ProbeConfig) -> ProbeResult:
        """执行探针主体，处理数据回传与告警。"""
        started = datetime.now(UTC)
        output, error = await execute_probe(cfg)
        elapsed_ms = int((datetime.now(UTC) - started).total_seconds() * 1000)
        now_iso = _now_iso()

        alerted, alert_detail = _evaluate_alert(cfg, output)
        confidence, confidence_reason, suspect_setup = _assess_confidence(cfg, output, error, elapsed_ms)

        result = ProbeResult(
            probe_name=cfg.name,
            output=output,
            error=error,
            triggered_at=now_iso,
            duration_ms=elapsed_ms,
            alerted=alerted,
            alert_detail=alert_detail if alerted else None,
            confidence=confidence,
            confidence_reason=confidence_reason,
            deployment_suspect=suspect_setup,
        )

        # 持久化最近结果
        await self._store.update_run_result(
            cfg.name,
            last_run_at=now_iso,
            last_result=output if output else None,
            last_error=error if error else None,
            last_confidence=confidence,
            last_confidence_reason=confidence_reason,
            last_suspect=suspect_setup,
            last_alerted=alerted,
            last_alert_detail=alert_detail if alerted else None,
        )

        _log.info(
            "[probe] ran probe=%s kind=%s elapsed=%dms error=%s alerted=%s confidence=%.2f suspect=%s",
            cfg.name, cfg.kind, elapsed_ms, bool(error), alerted, confidence, suspect_setup,
        )

        await self._deliver(cfg, result)
        return result

    async def _deliver(self, cfg: ProbeConfig, result: ProbeResult) -> None:
        """按 data_back 策略回传探针结果。"""
        if result.alerted and result.alert_detail:
            await self._push_wm(f"[🔔 探针告警] {result.alert_detail}", priority=_ALERT_WM_PRIORITY)
            if self._alert_event is not None:
                self._alert_event.set()
        if result.confidence < 0.6 or result.deployment_suspect:
            hint = (
                f"[🧪 探针可信度预警] {cfg.name} confidence={result.confidence:.2f}。"
                f"{result.confidence_reason}。"
                "在基于该读数决策前，先复核探针 spec/target/trigger 布放是否正确。"
            )
            await self._push_wm(hint, priority=_CONFIDENCE_WARN_PRIORITY)

        if cfg.data_back == "wm":
            summary = _format_summary(cfg, result)
            await self._push_wm(summary, priority=_WM_PRIORITY)
        # "none" — 仅日志，已在上面记录

    async def _push_wm(self, content: str, priority: float = _WM_PRIORITY) -> None:
        if self._wm is None:
            return
        from memory.working import WMItem  # 延迟 import 避免循环
        self._wm.add(WMItem(kind="probe_result", content=content, priority=priority))

    def status(self) -> dict[str, str]:
        """返回所有调度 Task 的状态摘要。"""
        return {
            name: ("running" if not t.done() else ("cancelled" if t.cancelled() else "done"))
            for name, t in self._tasks.items()
        }


def _format_summary(cfg: ProbeConfig, result: ProbeResult) -> str:
    purpose_hint = f" [{cfg.purpose}]" if getattr(cfg, "purpose", "") else ""
    header = f"[探针 {cfg.name}]{purpose_hint} {result.triggered_at} ({result.duration_ms}ms)"
    confidence = f"confidence={result.confidence:.2f}"
    if result.error:
        return f"{header}\n❌ 错误: {result.error}\n{confidence} ({result.confidence_reason})"
    body = result.output if result.output else "(无输出)"
    return f"{header}\n{body}\n{confidence} ({result.confidence_reason})"
