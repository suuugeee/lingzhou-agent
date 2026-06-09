"""core.config_models.base — provider 与 loop 配置模型。"""
from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ProviderDefinition(BaseModel):
    """单个 provider 的连接配置（与模型无关）。"""

    type: str = Field(
        default="openai_compat",
        description=(
            "wire protocol 选择器，决定使用哪个 Provider 实现类。\n"
            '  "openai_compat": OpenAI 兼容 API（百炼/DeepSeek/标准 OpenAI/Copilot 等）。\n'
            "添加新协议：新建 provider/*.py 实现 Provider 协议，"
            "在 provider/__init__.py 工厂的 match 里注册即可。"
        ),
    )
    base_url: str
    api_key_env: str = Field(
        default="OPENAI_API_KEY",
        description=(
            "openai 模式：存放 API Key 的环境变量名（如 DASHSCOPE_API_KEY）。\n"
            "copilot 模式：存放 GitHub PAT 的环境变量名。\n"
            "codex 模式：可选 OPENAI_CODEX_ACCESS_TOKEN fallback；推荐使用 auth profile。\n"
            "  推荐用专用变量 COPILOT_GITHUB_TOKEN，避免与 gh CLI / GitHub Actions\n"
            "  的通用 GITHUB_TOKEN 混用。若已执行 lingzhou auth login-copilot，\n"
            "  token 存于 auth-profiles.json，此字段仅作 env fallback。"
        ),
    )
    auth_profile_id: str = Field(
        default="",
        description="可选 auth profile id（如 'bailian:default'），用于从 ~/.lingzhou/auth-profiles.json 读取 token",
    )
    mode: str = Field(
        default="openai",
        description=(
            '仅对 type="openai_compat" 有效，描述认证后端与请求数据格式的差异。\n'
            '"openai"：标准 Bearer token + chat/completions API（Qwen/bailian 等）；\n'
            '"copilot"： GitHub token → Copilot token exchange + responses API。\n'
            '"codex"：OpenAI Codex OAuth + ChatGPT/Codex responses backend。\n'
            "其他协议（如未来的 anthropic/gemini）在各自 Provider 类里处理，不依赖此字段。"
        ),
    )
    extra_body: dict[str, Any] = Field(
        default_factory=dict,
        description="合并到每次请求 payload 的额外参数（escape hatch，最后覆盖）",
    )
    oauth_client_id: str = Field(
        default="",
        description="GitHub OAuth App Client ID（仅显式使用 --method device 时生效）",
    )
    models: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "可选模型元数据覆盖/补充列表。每项至少包含 id；"
            "同 id 条目会覆盖 provider/models.json 的内置字段，"
            "未出现的 id 则作为新增模型追加到运行时 models.json。"
        ),
    )

    @property
    def api_key(self) -> str:
        import re as _re
        # 若 api_key_env 不符合环境变量命名规则（如直接填入了 key 值），当作 literal key
        if self.api_key_env and not _re.match(r'^[A-Z_][A-Z0-9_]*$', self.api_key_env):
            return self.api_key_env
        # 1. 优先读环境变量
        key = os.environ.get(self.api_key_env, "").strip()
        if key:
            return key
        # 2. 回退：auth-profiles.json（需显式配置 auth_profile_id）
        if self.auth_profile_id:
            try:
                from store.auth import get_auth_profile

                profile = get_auth_profile(self.auth_profile_id)
                if isinstance(profile, dict):
                    token = str(profile.get("token", "")).strip()
                    if token:
                        return token
            except Exception:
                pass

        if self.mode == "copilot":
            raise OSError(
                f"未找到 {self.api_key_env!r} 的 GitHub token。\n"
                "请执行以下任一操作：\n"
                "  lingzhou auth login-copilot\n"
                f"  export {self.api_key_env}=your_token"
            )

        raise OSError(
            f"未找到 {self.api_key_env!r}（环境变量/auth-profile 均为空）。\n"
            f"请执行以下任一操作：\n"
            f"  export {self.api_key_env}=your_token\n"
            f"  lingzhou auth set-token --provider <provider>"
        )


class LoopConfig(BaseModel):
    max_concurrent_ticks: int = Field(
        default=4,
        ge=1,
        description=(
            "tick 并发上限。1=严格串行；>1 时启用分链并发："
            "同一 chain 内按顺序执行，不同 chain 可并行。"
            "默认 4：支持多用户同时对话（每位用户独立 chain）+ 自驱 auto tick 并行。"
        ),
    )
    max_tick_queue: int = Field(
        default=100,
        ge=1,
        description=(
            "等待中的 tick 全局队列上限。超过上限时新 tick 会被拒绝（chat 会返回繁忙提示）。"
        ),
    )
    max_tool_chain_workers: int = Field(
        default=8,
        ge=1,
        description=(
            "tool-chain-worker 的并发上限。控制普通工具链调用的同时执行数；"
            "超过后进入 worker 内部等待队列。"
        ),
    )
    max_exec_workers: int = Field(
        default=4,
        ge=1,
        description=(
            "exec-worker 的并发上限。控制前台/后台进程类工具的同时启动数。"
        ),
    )
    max_multimodal_workers: int = Field(
        default=2,
        ge=1,
        description=(
            "multimodal-worker 的并发上限。控制图像/多模态工具的同时执行数。"
        ),
    )
    max_llm_workers: int = Field(
        default=4,
        ge=1,
        description=(
            "llm-worker 的并发上限。控制带 fact/run_monitor 的 LLM 驱动工具的同时执行数。"
        ),
    )
    tick_job_timeout: float | None = Field(
        default=None,
        gt=0,
        description=(
            "单个 tick job 的外层保护超时（秒）。None=不加 dispatcher 外层超时，"
            "由 LLM/provider/tool 自身超时自然结束；设为正数时才启用外层保护。"
        ),
    )
    # ── 事件驱动时序（替代固定 interval 概念）───────────────────────────────
    # 设计依据：
    #   Friston Active Inference (2010/2017)：认知循环由自由能（预测误差）阈值驱动，
    #   而非时钟节拍。Global Workspace Theory (Baars 1988)：注意广播是事件触发，
    #   非周期轮询。SOAR / ACT-R：产生式持续激活，只在无匹配时挂起。
    #   → 灵舟应以"事件到达"而非"定时唤醒"作为认知节律的主驱动。
    min_act_gap: float = Field(
        default=500, ge=1,
        description=(
            "act 决策 + 有活跃任务后的最短间隔（毫秒）。让工具副作用短暂落地再进入下一认知轮，"
            "避免无限紧循环。不等固定 interval，执行中的任务可连续推进。"
        ),
    )
    active_idle_gap: float = Field(
        default=15000, ge=100,
        description=(
            "有活跃任务但 decision=wait/pause 时的默认等待上限（毫秒）。"
            "LLM 未表达偏好时使用此值作为备用；如设了 idle_with_task_bounds 面板 LLM 可覆盖的范围。"
        ),
    )
    max_idle_gap: float = Field(
        default=60000, ge=100,
        description=(
            "无活跃任务时的默认等待上限（毫秒）。"
            "chat 消息、task 状态变化任一事件即立即唤醒，不等满此值。"
            "LLM 未表达偏好时使用；如设了 idle_no_task_bounds 面板 LLM 可覆盖的范围。"
        ),
    )
    idle_with_task_bounds: list[float] = Field(
        default=[100, 30000],
        description=(
            "[min, max]：LLM 通过 next_idle_gap_secs / next_idle_gap_ms 在有活跃任务时可指定的等待时长范围（毫秒）。"
            "对 min_act_gap 后的短等待同样起下限保护作用（防止紧循环）。"
            "示例：[500, 60000] 表示 LLM 最快 500ms 最慢 60s；[100, 30000] 最快 100ms。"
        ),
    )
    idle_no_task_bounds: list[float] = Field(
        default=[2000, 300000],
        description=(
            "[min, max]：LLM 通过 next_idle_gap_secs / next_idle_gap_ms 在无活跃任务时可指定的等待时长范围（毫秒）。"
            "示例：[2000, 300000] 表示空闲时 LLM 最快 2s 触发下一探索 tick、最慢 300s。"
        ),
    )

    @model_validator(mode="after")
    def _validate_bounds(self) -> LoopConfig:
        for name in ("idle_with_task_bounds", "idle_no_task_bounds"):
            val = getattr(self, name)
            if len(val) != 2:
                raise ValueError(
                    f"{name} 必须是长度为 2 的列表 [min, max]，当前长度 {len(val)}"
                )
            if val[0] < 0 or val[1] < 0:
                raise ValueError(
                    f"{name} 的值不能为负数，当前 [{val[0]}, {val[1]}]"
                )
            if val[0] >= val[1]:
                raise ValueError(
                    f"{name}[0]={val[0]} 必须小于 [1]={val[1]}"
                )
        return self

    wake_poll_interval: float = Field(default=200, ge=1, description="事件轮询粒度（毫秒），越小响应越快但 CPU 开销越高")
    arousal_min_factor: float = Field(default=0.8, ge=0.0, le=1.0, description="arousal 调制因子下限（高唤醒时等待间隔最多缩短至此比例）")
    arousal_sensitivity: float = Field(default=0.4, ge=0.0, le=2.0, description="arousal 调制灵敏度（偏离中性每单位对应的因子变化量）")
    arousal_neutral: float = Field(default=0.5, ge=0.0, le=1.0, description="arousal 中性基准值（对应调制因子 = 1.0）")
    wake_on_task_change: bool = Field(default=True, description="任务状态变化时是否提前唤醒")
    chat_thinking: str = Field(
        default="low",
        description=(
            "chat/interact 模式（有用户消息）时的 thinking 等级覆盖。"
            "可选: off | minimal | low | medium | high。"
            "独立配置使 chat 模式在保留基本推理的同时大幅降低首次 decide() 耗时（40-60s → 3-10s）。"
        ),
    )
    autonomous_thinking: str = Field(
        default="medium",
        description=(
            "自主认知循环（无用户消息）时的 thinking 等级覆盖。"
            "可选: off | minimal | low | medium | high。"
            "日志分析显示 thinking=high 导致每 tick 耗时 40-90s，严重阻塞 chat 响应（用户消息"
            "必须等当前 LLM 调用完成才能被处理）。medium 约 10-20s，在推理质量与响应性间取得平衡。"
            "复杂推理（evolution、ethos 反思）仍可单独配置为 high。"
        ),
    )
    # 运行时目录默认生产布局：推荐落在 ~/.lingzhou 下，
    # 默认不写回源码仓目录；源码树通常只承载代码与文档，不承载 runtime data。
    db_path: str = "~/.lingzhou/state/runtime.db"
    memory_dir: str = "~/.lingzhou/memory"
    state_dir: str = "~/.lingzhou/state"
    workspace_dir: str = "~/.lingzhou/workspace"  # 人类可读镜像层（SOUL.md 等）
    act: bool = Field(default=True, description="True=真实执行，False=dry-run")
    debug: bool = False
    consolidate_every: int = Field(default=10, ge=1, description="每 N 轮 WM→语义整合")
    evolve_every: int = Field(default=30, ge=1, description="每 N 轮自进化检查")
    max_consecutive_errors: int = Field(default=5, ge=1, description="连续错误上限后暂停")
    heartbeat_interval: int = Field(default=300, ge=60, description="心跳自检信号触发间隔（秒，默认 5 分钟）")
    judge_every: int = Field(
        default=1, ge=1,
        description=(
            "按请求计费优化：空闲（无活跃任务、无用户消息）时每 N 轮才真正调用 LLM 判断。"
            "有任务或用户消息时忽略此设置，始终调用。默认 1 = 每轮都调（无聚合）。"
        ),
    )
    wait_streak_notify: list[int] = Field(
        default=[3, 6],
        description=(
            "连续 wait/pause 轮数触发自我感知通知的阈值列表（升序）。"
            "每个阈值首次达到时向工作记忆注入一条状态汇报，由 LLM 自主决定是否继续等待。"
            "空列表 [] 表示禁用此机制。示例：[3] 仅轻提示；[3,6,10] 三级递进。"
        ),
    )
    behavior_streak_threshold: int = Field(
        default=3, ge=1,
        description=(
            "行为重复探测窗口大小：连续 N 次相同 (action/read/list) 时向 WM 注入自我感知信号。"
            "同时是 action_streak / read_streak / list_streak 的滑动窗口大小。"
            "默认 3 与 wait_streak_notify 基准对齐；调大可降低多次重复的噪声，调小则更早袋扔循环。"
        ),
    )
    continue_reasoner_after_n_tools: int = Field(
        default=4, ge=1,
        description=(
            "continue 阶段工具历史达到 N 条后自动升级为 reasoner tier（处理更多工具历史需要更强直觉）。"
            "LLM 已显式设置 next_phase_tier 时优先级更高。"
        ),
    )
    wechat_coalesce_delay: float = Field(
        default=3.0, ge=0.0,
        description=(
            "wechat 通道图文合并等待窗口（秒）。"
            "用户连续发文字+图片时，iLink 将两条消息独立下发，图片需要下载解密才能写入 DB。"
            "drain 前等待此时长，让同批次图片消息有机会落库后再合并进同一 LLM 轮次。"
            "设为 0 可禁用（不等待）；建议范围 0.5-3.0。"
        ),
    )


class LoggingConfig(BaseModel):
    dir: str = Field(default="~/.lingzhou/logs", description="运行日志目录")
    daily_enabled: bool = Field(default=True, description="是否写入按日期分割的 lingzhou 日志")
    startup_enabled: bool = Field(default=True, description="是否写入启动摘要日志")
    startup_prefix: str = Field(default="console", description="启动/console 日志文件名前缀")
    console_enabled: bool = Field(default=True, description="是否写入 console 日志")
    console_file: str = Field(default="console.log", description="console 日志文件名")
