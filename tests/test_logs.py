"""logs 命令回归测试。"""


def test_logs_files_identifies_daily_and_stdout_roles(monkeypatch, tmp_path):
    from cli import logs as logs_mod

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    daily = log_dir / "lingzhou-2026-06-23.log"
    stdout = log_dir / "daemon-stdout.log"
    daily.write_text("run=1 status=succeeded\n", encoding="utf-8")
    stdout.write_text("stdout only\n", encoding="utf-8")

    monkeypatch.setattr(logs_mod, "LOG_DIR", log_dir)
    monkeypatch.setattr(logs_mod, "_latest_log", lambda: daily)

    printed: list[str] = []

    def _capture_print(message):
        printed.append(str(message))

    monkeypatch.setattr(logs_mod.console, "print", _capture_print)

    logs_mod.logs_files()

    output = "\n".join(printed)
    assert "daily" in output
    assert str(daily) in output
    assert "gateway logs/tail 默认读取这里" in output
    assert "stdout" in output
    assert str(stdout) in output
    assert "不保证包含结构化工具结果" in output



def test_logs_stats_includes_judgment_overflow_metrics(monkeypatch, tmp_path):
    from cli import logs as logs_mod

    log_file = tmp_path / "lingzhou-2026-05-31.log"
    log_file.write_text(
        "\n".join(
            [
                "[boot] start",
                "WARNING warning line",
                "ERROR error line",
                "[loop] tick decision=act",
                "[loop] tick decision=wait",
                "[chat] user hello",
                "[judgment] LLM 调用失败，2.20s 后重试: overflow_kind=output retry_after_seconds=2.00s backoff_seconds=2.20 err=429",
                "[judgment] LLM prompt_overflow x messages_omitted=true overflow_kind=prompt",
                "[judgment] LLM output_overflow x messages_omitted=false overflow_kind=output",
                "[wechat] chat_msg",
                "[wechat] 回复成功",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(logs_mod, "_latest_log", lambda: log_file)

    printed: list[str] = []

    def _capture_print(message):
        printed.append(str(message))

    monkeypatch.setattr(logs_mod.console, "print", _capture_print)

    logs_mod.logs_stats()

    output = "\n".join(printed)
    assert "overflow:  prompt=1 output=2" in output
    assert "超窗省略:  omitted=1 skipped=1" in output
    assert "backoff:   1 次 (avg=2.20s)" in output
    assert "LLM失败:" in output
