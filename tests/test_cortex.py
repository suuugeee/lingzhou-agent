from __future__ import annotations

from core.cortex import (
    build_action_first_cortex_patch,
    build_auto_cortex_patch,
    build_cortex_workspace,
    build_problem_solving_guard,
    extract_action_first_signal,
    format_cortex_workspace,
    format_problem_solving_guard,
)
from store.task import Failure, Run, Task


def test_verification_state_from_cortex_respects_control_marker():
    from core.cortex import intent as cortex_intent

    text = cortex_intent.control_next_verification(
        "运行 pytest 后再 verify blocker details."
    )

    assert cortex_intent.has_actionable_next_verification(text) is False
    assert cortex_intent.verification_state_from_cortex({"next_verification": text}) == (
        "resolved",
        "运行 pytest 后再 verify blocker details.",
    )


def test_build_verification_state_classifies_next_verification_once():
    from core.cortex import intent as cortex_intent

    pending = cortex_intent.build_verification_state(
        {"next_verification": "运行 pytest 验证 workbench 下一步。"}
    )
    resolved = cortex_intent.build_verification_state(
        {"next_verification": "none：当前证据已足够完成。"}
    )
    control = cortex_intent.build_verification_state(
        {
            "next_verification": cortex_intent.control_next_verification(
                "运行 pytest 后再 verify blocker details."
            )
        }
    )

    assert pending == {
        "status": "pending",
        "goal": "运行 pytest 验证 workbench 下一步。",
        "source": "workbench",
    }
    assert resolved == {
        "status": "resolved",
        "goal": "none：当前证据已足够完成。",
        "source": "workbench",
    }
    assert control == {
        "status": "resolved",
        "goal": "运行 pytest 后再 verify blocker details.",
        "source": "workbench",
    }


def test_next_verification_completion_text_classification():
    from core.cortex import intent as cortex_intent

    assert cortex_intent.has_actionable_next_verification("确认已完成测试并读取最新 pytest 结果") is True
    assert cortex_intent.has_actionable_next_verification("已完成，无需继续验证") is False
    assert cortex_intent.has_actionable_next_verification("already done; no need to rerun") is False
    assert cortex_intent.has_actionable_next_verification("none：当前证据已足够完成；除非后续 schema 迁移或查询失败，否则不再重复读取同一路径。") is False
    assert cortex_intent.has_actionable_next_verification("[[control-next-verification]] Try task.complete to close and then verify blocker details.") is False
    assert cortex_intent.has_actionable_next_verification("[[control-next-verification]] 运行 pytest 后再 verify blocker details.") is False


def test_successful_verification_run_tool_classification():
    from types import SimpleNamespace

    from core.cortex import intent as cortex_intent

    assert cortex_intent.is_successful_verification_run(
        SimpleNamespace(status="succeeded", tool_name="shell.run")
    )
    assert not cortex_intent.is_successful_verification_run(
        SimpleNamespace(status="succeeded", tool_name="task.workbench")
    )
    assert not cortex_intent.is_successful_verification_run(
        SimpleNamespace(status="failed", tool_name="shell.run")
    )
    assert not cortex_intent.is_successful_verification_run(
        SimpleNamespace(status="succeeded", tool_name="memory.add_semantic")
    )
    assert cortex_intent.is_successful_verification_run(
        SimpleNamespace(status="succeeded", tool_name="memory.add_semantic"),
        next_verification="沉淀一条语义记忆：记录本次恢复规则。",
    )


def test_cortex_workspace_derives_task_context_from_existing_artifacts():
    task = Task(
        id=42,
        title="完善大脑皮层",
        status="in_progress",
        priority="normal",
        created_at="2026-06-09T00:00:00+00:00",
        goal="让长任务不只依赖短期工作记忆",
        current_step="接入 judgment context",
        next_step="补测试并验证",
        extras={
            "plan": [
                {"step": "审查 WM 和 judgment 链路", "status": "completed"},
                {"step": "加入任务级 cortex workspace", "status": "in_progress"},
            ]
        },
    )
    recent_runs = [
        Run(
            id=7,
            task_id=42,
            run_type="tool",
            worker_type="reasoner",
            status="completed",
            created_at="2026-06-09T00:01:00+00:00",
            tool_name="exec",
            progress="已找到 context 注入点",
        )
    ]
    failures = [
        Failure(
            id=3,
            kind="context_too_cramped",
            dismissed=False,
            created_at="2026-06-09T00:02:00+00:00",
            summary="近期上下文无法稳定承载任务状态",
            task_id="42",
        )
    ]

    workspace = build_cortex_workspace(
        task=task,
        recent_runs=recent_runs,
        context_facts=[("task:42:proxy", "授权代理已接入配置")],
        failures=failures,
    )
    text = format_cortex_workspace(workspace)

    assert "task_id=42 status=in_progress" in text
    assert "goal=让长任务不只依赖短期工作记忆" in text
    assert "1. [completed] 审查 WM 和 judgment 链路" in text
    assert "task:42:proxy: 授权代理已接入配置" in text
    assert "run#7 [completed] exec: 已找到 context 注入点" in text
    assert "context_too_cramped: 近期上下文无法稳定承载任务状态" in text


def test_cortex_workspace_prefers_explicit_cortex_result_state():
    task = Task(
        id=9,
        title="任务",
        status="running",
        priority="normal",
        created_at="2026-06-09T00:00:00+00:00",
        result_json={
            "cortex": {
                "plan": [{"step": "显式计划", "status": "active"}],
                "evidence": ["显式证据"],
                "progress": ["显式进展"],
                "failures": ["显式失败"],
                "open_questions": ["显式问题"],
            }
        },
        extras={"plan": [{"step": "派生计划", "status": "pending"}]},
    )

    text = format_cortex_workspace(build_cortex_workspace(task=task))

    assert "[active] 显式计划" in text
    assert "显式证据" in text
    assert "显式进展" in text
    assert "显式失败" in text
    assert "显式问题" in text
    assert "派生计划" not in text


def test_cortex_workspace_formats_general_problem_solving_workbench():
    task = Task(
        id=11,
        title="通用排障",
        status="in_progress",
        priority="normal",
        created_at="2026-06-09T00:00:00+00:00",
        result_json={
            "cortex": {
                "domain": "network_proxy",
                "intent": "switch_outbound_node_and_retry_push",
                "hypothesis": "当前出站节点导致 GitHub TLS 中断",
                "capabilities": [
                    {"name": "mihomo external-controller", "status": "available"},
                ],
                "experiments": [
                    {"target": "github.com", "status": "failed", "error": "gnutls_handshake"},
                ],
                "recovery_state": "enumerating_alternatives",
                "next_verification": "切换候选节点后执行 git ls-remote",
                "completion_checks": [
                    {"text": "git push 成功", "status": "pending"},
                ],
            }
        },
    )

    text = format_cortex_workspace(build_cortex_workspace(task=task))

    assert "problem_solving:" in text
    assert "domain=network_proxy" in text
    assert "intent=switch_outbound_node_and_retry_push" in text
    assert "hypothesis=当前出站节点导致 GitHub TLS 中断" in text
    assert "capability_map:" in text
    assert "[available] mihomo external-controller" in text
    assert "experiment_log:" in text
    assert "[failed] target=github.com error=gnutls_handshake" in text
    assert "recovery_state=enumerating_alternatives" in text
    assert "next_verification=切换候选节点后执行 git ls-remote" in text
    assert "completion_checks:" in text


def test_action_first_signal_extracts_strong_inputs_and_execute_intent():
    signal = extract_action_first_signal(
        "用这个url的配置https://example.com/sub?clash=1，下载后写入 /Users/suge/.config/clash/config.yaml"
    )

    assert signal.intent == "execute"
    assert signal.must_act is True
    assert "strong_input" in signal.markers
    assert {"kind": "url", "value": "https://example.com/sub?clash=1"} in signal.captured_inputs
    assert {
        "kind": "path",
        "value": "/Users/suge/.config/clash/config.yaml",
    } in signal.captured_inputs
    assert "最小可验证动作" in signal.minimum_next_action


def test_action_first_signal_keeps_analysis_questions_as_analysis():
    signal = extract_action_first_signal("为什么 Lingzhou 的动手能力弱，分析一下架构问题")

    assert signal.intent == "analyze"
    assert signal.must_act is False
    assert "analysis_marker" in signal.markers


def test_action_first_signal_keeps_identity_birthday_question_as_recall():
    signal = extract_action_first_signal("你的生日是什么时候")

    assert signal.intent == "converse"
    assert signal.must_act is False
    assert "question_like" in signal.markers
    assert signal.captured_inputs == []


def test_action_first_signal_does_not_force_conversational_memory_hint_without_path():
    signal = extract_action_first_signal("你或许可以去旧记忆上面找一下。")

    assert signal.must_act is False
    assert "soft_inspection_marker" in signal.markers
    assert signal.captured_inputs == []


def test_action_first_signal_forces_soft_inspection_with_current_strong_input():
    signal = extract_action_first_signal("看一下 /Users/suge/Downloads/daemon-stdout.log")

    assert signal.intent == "execute"
    assert signal.must_act is True
    assert "soft_inspection_marker" in signal.markers
    assert {"kind": "path", "value": "/Users/suge/Downloads/daemon-stdout.log"} in signal.captured_inputs


def test_action_first_signal_does_not_force_example_feedback():
    signal = extract_action_first_signal("这个是宽泛的例子，我只是把案例问题告诉你。")

    assert signal.intent == "converse"
    assert signal.must_act is False
    assert "feedback_like" in signal.markers
    assert signal.captured_inputs == []


def test_action_first_cortex_patch_persists_inputs_without_dropping_existing_state():
    patch = build_action_first_cortex_patch(
        existing_cortex={
            "domain": "network",
            "captured_inputs": [{"kind": "path", "value": "/tmp/old.yaml"}],
        },
        user_message="下载 https://example.com/a.yaml 后测试",
    )

    cortex = patch["cortex"]
    assert cortex["domain"] == "network"
    assert cortex["action_first"]["intent"] == "execute"
    assert cortex["action_first"]["must_act"] is True
    assert {"kind": "url", "value": "https://example.com/a.yaml"} in cortex["captured_inputs"]
    assert {"kind": "path", "value": "/tmp/old.yaml"} in cortex["captured_inputs"]


def test_action_first_cortex_patch_clears_stale_inputs_for_conversational_feedback():
    patch = build_action_first_cortex_patch(
        existing_cortex={
            "domain": "logs",
            "captured_inputs": [{"kind": "path", "value": "/Users/suge/Downloads/daemon-stdout.log"}],
            "action_first": {
                "intent": "execute",
                "must_act": True,
                "minimum_next_action": "先对用户给定的强输入做最小可验证动作",
            },
        },
        user_message="这个是宽泛的例子，我只是把案例问题告诉你。",
    )

    cortex = patch["cortex"]
    assert "captured_inputs" not in cortex
    assert cortex["action_first"]["intent"] == "converse"
    assert cortex["action_first"]["must_act"] is False
    assert "minimum_next_action" not in cortex["action_first"]


def test_cortex_workspace_formats_action_first_state():
    task = Task(
        id=21,
        title="应用代理配置",
        status="in_progress",
        priority="normal",
        created_at="2026-06-09T00:00:00+00:00",
        result_json={
            "cortex": {
                "action_first": {
                    "intent": "execute",
                    "must_act": True,
                    "markers": ["execute_marker", "strong_input"],
                    "minimum_next_action": "先下载并校验配置",
                },
                "captured_inputs": [
                    {"kind": "url", "value": "https://example.com/sub?clash=1"},
                ],
            }
        },
    )

    text = format_cortex_workspace(build_cortex_workspace(task=task))

    assert "action_first:" in text
    assert "- intent=execute" in text
    assert "- must_act=yes" in text
    assert "先下载并校验配置" in text
    assert "captured_inputs:" in text
    assert "url=https://example.com/sub?clash=1" in text


def test_problem_solving_guard_requires_workbench_on_user_correction():
    task = Task(
        id=12,
        title="处理节点问题",
        status="in_progress",
        priority="normal",
        created_at="2026-06-09T00:00:00+00:00",
        next_step="切换节点并验证",
    )
    workspace = build_cortex_workspace(task=task)

    guard = build_problem_solving_guard(
        task=task,
        workspace=workspace,
        user_message="我指的是代理节点，不是模型节点",
    )
    text = format_problem_solving_guard(guard)

    assert guard.active is True
    assert "user_correction" in guard.signals
    assert "workbench_incomplete" in guard.signals
    assert "missing_fields=domain, intent, hypothesis" in text
    assert "task.amend" in text
    assert "task.workbench" in text


def test_problem_solving_guard_idles_when_workbench_is_complete():
    task = Task(
        id=13,
        title="通用修复",
        status="in_progress",
        priority="normal",
        created_at="2026-06-09T00:00:00+00:00",
        result_json={
            "cortex": {
                "domain": "git",
                "intent": "retry push",
                "hypothesis": "transport failure",
                "capabilities": ["git ls-remote 可用"],
                "experiments": ["git ls-remote ok"],
                "next_verification": "git push",
                "completion_checks": ["remote main updated"],
            }
        },
    )
    workspace = build_cortex_workspace(task=task)

    guard = build_problem_solving_guard(
        task=task,
        workspace=workspace,
        user_message="继续解决失败",
        failures=[Failure(id=1, kind="git_push", dismissed=False, created_at="now")],
    )

    assert guard.active is False
    assert "visible_failures" in guard.signals
    assert guard.missing_fields == []


def test_problem_solving_guard_does_not_block_on_completion_only_gaps():
    task = Task(
        id=14,
        title="验证修复",
        status="in_progress",
        priority="normal",
        created_at="2026-06-09T00:00:00+00:00",
        result_json={
            "cortex": {
                "domain": "runtime persistence",
                "intent": "verify compaction",
                "hypothesis": "store layer keeps previews instead of full raw output",
                "evidence": ["pytest 已覆盖大输出持久化"],
                "next_verification": "运行 pytest 验证回归。",
            }
        },
    )
    workspace = build_cortex_workspace(task=task)

    guard = build_problem_solving_guard(
        task=task,
        workspace=workspace,
        user_message="继续修复并验证",
    )

    assert guard.active is False
    assert "completion_checks" in guard.missing_fields
    assert "capabilities" in guard.missing_fields
    assert "workbench_incomplete" not in guard.signals


def test_auto_cortex_patch_appends_run_outcomes_without_overwriting_intent():
    patch = build_auto_cortex_patch(
        existing_cortex={
            "domain": "git",
            "intent": "retry push",
            "hypothesis": "transport failure",
            "experiments": [{"run_id": "1", "tool": "git", "status": "failed"}],
        },
        run_id=2,
        task_id=13,
        tool_name="shell.run",
        status="failed",
        summary="gnutls_handshake failed",
        error="TLSFailure",
        evidence="",
        progress="",
        state_delta={"exit_code": 128},
        artifact_paths=[],
    )

    cortex = patch["cortex"]
    assert cortex["domain"] == "git"
    assert cortex["intent"] == "retry push"
    assert cortex["hypothesis"] == "transport failure"
    assert cortex["experiments"][0]["run_id"] == "2"
    assert cortex["experiments"][0]["status"] == "failed"
    assert cortex["experiments"][1]["run_id"] == "1"
    assert "run#2 shell.run failed" in cortex["failures"][0]
    assert cortex["recovery_state"] == "recovering_from_run_failure"
    assert cortex["problem_runtime"]["phase"] == "recovering"
    assert cortex["problem_runtime"]["failure_streak"] == 1
    assert "next_verification" in cortex


def test_auto_cortex_patch_marks_successful_non_task_run_as_verification_collected():
    patch = build_auto_cortex_patch(
        existing_cortex={
            "action_first": {"intent": "execute", "must_act": True},
            "problem_runtime": {"phase": "acting", "failure_streak": 2},
        },
        run_id=5,
        task_id=13,
        tool_name="web.fetch",
        status="succeeded",
        summary="获取成功",
    )

    cortex = patch["cortex"]
    assert cortex["problem_runtime"]["phase"] == "verification_collected"
    assert cortex["problem_runtime"]["failure_streak"] == 0
    assert cortex["problem_runtime"]["last_success_run_id"] == "5"
    assert cortex["action_first"]["last_verifiable_action_run_id"] == "5"


def test_auto_cortex_patch_promotes_tool_input_recovery_hint():
    patch = build_auto_cortex_patch(
        existing_cortex={"domain": "runtime", "intent": "fix tool call"},
        run_id=7,
        task_id=13,
        tool_name="task.workbench",
        status="failed",
        summary="工具参数缺失: task.workbench requires workbench",
        error="ToolInputInvalid",
        state_delta={
            "tool_input_invalid": True,
            "missing_params": ["workbench"],
            "recovery_next_step": "按 task.workbench 的 manifest 重新调用工具；补齐必填参数 workbench。",
            "retry_params_template": {"workbench": {}},
        },
    )

    cortex = patch["cortex"]
    assert cortex["recovery_state"] == "recovering_from_run_failure"
    assert cortex["problem_runtime"]["phase"] == "recovering"
    assert cortex["next_verification"] == "按 task.workbench 的 manifest 重新调用工具；补齐必填参数 workbench。"
    assert cortex["experiments"][0]["run_id"] == "7"


def test_auto_cortex_patch_promotes_completion_block_recovery_hint():
    patch = build_auto_cortex_patch(
        existing_cortex={"domain": "self_evolution", "intent": "self_drive_growth"},
        run_id=8,
        task_id=13,
        tool_name="task.complete",
        status="failed",
        summary="任务皮层仍有未验证的下一步",
        error="WorkbenchVerificationPending",
        state_delta={
            "completion_blocked": True,
            "completion_blocker": "WorkbenchVerificationPending",
            "recovery_next_step": "读取最新 loop 日志并确认 active_idle_gap 是否生效。",
            "suggested_tools": ["file.read", "shell.run"],
        },
    )

    cortex = patch["cortex"]
    assert cortex["problem_runtime"]["phase"] == "recovering"
    assert cortex["next_verification"] == "读取最新 loop 日志并确认 active_idle_gap 是否生效。"
    assert "run#8 task.complete failed" in cortex["failures"][0]


def test_auto_cortex_patch_runtime_guard_failure_overrides_stale_next_verification():
    patch = build_auto_cortex_patch(
        existing_cortex={
            "domain": "runtime",
            "intent": "fix tool call",
            "next_verification": "继续读取旧日志。",
        },
        run_id=10,
        task_id=13,
        tool_name="task.workbench",
        status="failed",
        summary="工具参数缺失: task.workbench requires workbench",
        error="ToolInputInvalid",
        state_delta={
            "tool_input_invalid": True,
            "recovery_next_step": "按 task.workbench 的 manifest 重新调用工具；补齐必填参数 workbench。",
        },
    )

    cortex = patch["cortex"]
    assert cortex["problem_runtime"]["phase"] == "recovering"
    assert cortex["recovery_state"] == "recovering_from_run_failure"
    assert cortex["next_verification"] == "按 task.workbench 的 manifest 重新调用工具；补齐必填参数 workbench。"


def test_auto_cortex_patch_preserves_completion_blocker_when_run_succeeded():
    patch = build_auto_cortex_patch(
        existing_cortex={"domain": "self_evolution", "intent": "self_drive_growth"},
        run_id=9,
        task_id=13,
        tool_name="task.workbench",
        status="succeeded",
        summary="",
        state_delta={
            "completion_blocked": True,
            "completion_blocker": "WorkbenchVerificationPending",
            "next_verification": "先执行 probe.run 确认 reasoner_route_drift_watch。",
            "recovery_next_step": "先执行 probe.run 确认 reasoner_route_drift_watch。",
        },
    )

    cortex = patch["cortex"]
    assert cortex["problem_runtime"]["phase"] == "recovering"
    assert cortex["recovery_state"] == "recovering_from_runtime_guard"
    assert cortex["next_verification"] == "先执行 probe.run 确认 reasoner_route_drift_watch。"


def test_auto_cortex_patch_skips_task_workbench_to_avoid_self_noise():
    assert build_auto_cortex_patch(
        existing_cortex={"domain": "git"},
        run_id=3,
        task_id=13,
        tool_name="task.workbench",
        status="succeeded",
        summary="工作台已更新",
    ) == {}
