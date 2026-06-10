from __future__ import annotations

from core.contracts.probe import ProbeConfig, ProbeResult
from core.judgment.context.skills import _fmt_blind_spots, _fmt_probe_sensors
from core.probe.runner import _assess_confidence, _format_summary


def _curl_probe() -> ProbeConfig:
    return ProbeConfig(
        name="hermesclaw_health",
        kind="shell",
        spec="curl -s -o /dev/null -w %{http_code} http://127.0.0.1:19997/",
        trigger="interval:300",
        purpose="微信代理健康检查",
    )


def test_confidence_for_http_code_probe_success_is_high() -> None:
    cfg = _curl_probe()
    score, reason, suspect = _assess_confidence(cfg, "200", None, 120)

    assert score >= 0.8
    assert suspect is False
    assert "正常" in reason or "读数" in reason


def test_confidence_for_http_code_probe_invalid_output_is_low_and_suspect() -> None:
    cfg = _curl_probe()
    score, reason, suspect = _assess_confidence(cfg, "ok", None, 120)

    assert score <= 0.55
    assert suspect is True
    assert "状态码" in reason or "布放" in reason


def test_confidence_for_error_marks_suspect_when_connection_issue() -> None:
    cfg = _curl_probe()
    score, reason, suspect = _assess_confidence(cfg, "", "Connection refused", 55)

    assert score <= 0.25
    assert suspect is True
    assert "配置" in reason or "布放" in reason


def test_probe_sensor_panel_contains_confidence_and_guardrail() -> None:
    cfg = _curl_probe()
    cfg.last_run_at = "2026-05-22T10:53:43+00:00"
    cfg.last_result = "501"
    cfg.last_confidence = 0.42
    cfg.last_confidence_reason = "未解析到稳定状态，需复核布放"
    cfg.last_suspect = True

    panel = _fmt_probe_sensors([cfg])

    assert "confidence=0.42" in panel
    assert "布放可疑" in panel
    assert "先校验探针布放" in panel


def test_blind_spots_use_explicit_coverage_tags_not_free_text() -> None:
    cfg = _curl_probe()
    cfg.purpose = "微信代理健康检查"
    cfg.coverage_tags = []

    blind_spots_without_tag = _fmt_blind_spots([cfg])
    assert "关键外部通道健康未监控" in blind_spots_without_tag

    cfg.coverage_tags = ["ops:channel_health"]
    blind_spots_with_tag = _fmt_blind_spots([cfg])
    assert "关键外部通道健康未监控" not in blind_spots_with_tag


def test_probe_wm_summary_compacts_normal_large_output() -> None:
    cfg = _curl_probe()
    result = ProbeResult(
        probe_name=cfg.name,
        output="recent_hits=" + "X" * 5000,
        error=None,
        triggered_at="2026-06-10T19:34:00+00:00",
        duration_ms=120,
        alerted=False,
        confidence=0.85,
        confidence_reason="读数形态与执行状态正常",
        deployment_suspect=False,
    )

    summary = _format_summary(cfg, result)

    assert "正常读数已压缩" in summary
    assert "output_chars=" in summary
    assert "X" * 100 not in summary


def test_probe_wm_summary_keeps_abnormal_output_but_clips() -> None:
    cfg = _curl_probe()
    result = ProbeResult(
        probe_name=cfg.name,
        output="recent_hits=" + "Y" * 5000,
        error=None,
        triggered_at="2026-06-10T19:34:00+00:00",
        duration_ms=120,
        alerted=True,
        confidence=0.85,
        confidence_reason="读数形态与执行状态正常",
        deployment_suspect=False,
    )

    summary = _format_summary(cfg, result)

    assert "recent_hits=" in summary
    assert "探针输出已截断" in summary
    assert len(summary) < 1800
