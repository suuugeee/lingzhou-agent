"""core.config_models.advanced — emotion/evolution/soul/thresholds/gateway 配置模型。"""
from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, Field


class EmotionConfig(BaseModel):
    baseline_valence: float = Field(default=0.6, ge=0.0, le=1.0, description="情感基线效价")
    baseline_arousal: float = Field(default=0.5, ge=0.0, le=1.0, description="情感基线唤醒")
    ema_alpha: float = Field(default=0.15, ge=0.0, le=1.0, description="EMA 平滑系数")
    failure_normalization_count: float = Field(
        default=3.0, gt=0.0,
        description="failure_count 归一化到 1.0 的基准值；越小表示单次失败对情绪推导的负面影响越大",
    )
    high_error_normalization_streak: float = Field(
        default=3.0, gt=0.0,
        description="high_error_streak 归一化到 1.0 的基准值；越小表示连续高误差对控制感的冲击越大",
    )
    feeling_min_intensity: float = Field(
        default=0.15, ge=0.0, le=1.0,
        description="离散情感写入 EmotionState.feelings 的最小强度门槛；低于此值的情感不进入显式 feelings 列表",
    )
    regulation_down_regulate_arousal_high: float = Field(
        default=0.75, ge=0.0, le=1.0,
        description="arousal 高于此阈值时触发 down-regulate",
    )
    regulation_down_regulate_valence_low: float = Field(
        default=0.30, ge=0.0, le=1.0,
        description="valence 低于此阈值时触发 down-regulate",
    )
    regulation_down_regulate_worsening_valence: float = Field(
        default=0.45, ge=0.0, le=1.0,
        description="replay_trend=worsening 且 valence 低于此阈值时触发 down-regulate",
    )
    regulation_up_regulate_recovering_valence: float = Field(
        default=0.55, ge=0.0, le=1.0,
        description="replay_trend=recovering 且 valence 低于此阈值时触发 up-regulate",
    )
    regulation_up_regulate_signal_valence: float = Field(
        default=0.60, ge=0.0, le=1.0,
        description="recovering 信号存在且 valence 低于此阈值时触发 up-regulate",
    )
    regulation_high_error_streak_guard: int = Field(
        default=2, ge=1,
        description="high_error_streak 达到此阈值时触发 down-regulate",
    )
    reflection_valence_history_weight: float = Field(
        default=0.8, ge=0.0,
        description="reflection 中显式 valence hint 与当前 valence 融合时，历史当前值的权重",
    )
    reflection_valence_hint_weight: float = Field(
        default=0.2, ge=0.0,
        description="reflection 中显式 valence hint 与当前 valence 融合时，hint 的权重",
    )

    # Russell (1980) 环形情绪模型象限边界——display 层 + judgment 层共用同一套
    mood_valence_high: float = Field(default=0.65, ge=0.0, le=1.0, description="高效价区下边界（正向情绪）")
    mood_valence_low:  float = Field(default=0.35, ge=0.0, le=1.0, description="低效价区上边界（负向情绪）")
    mood_arousal_high: float = Field(default=0.65, ge=0.0, le=1.0, description="高唤醒区下边界（激活状态）")
    mood_arousal_low:  float = Field(default=0.45, ge=0.0, le=1.0, description="低唤醒区上边界（平静状态）")
    delta_display_min: float = Field(default=0.03, ge=0.0, le=1.0, description="interact 显示情绪变化的最小变化量")


class EvolutionConfig(BaseModel):
    enabled: bool = Field(default=True, description="是否允许运行时自修改工具代码")
    sandbox_timeout: float = Field(default=10.0, gt=0, description="沙箱执行超时（秒）")
    max_attempts: int = Field(default=3, ge=1, description="单次进化最多重试次数")
    backup: bool = Field(default=True, description="进化前是否备份原始文件")
    verify_min_runs: int = Field(default=3, ge=1, description="进化后最少观察多少次同工具 run，再判定效果是否稳定")
    auto_rollback_on_regression: bool = Field(default=True, description="进化后若观测到同工具明显退化，是否自动回滚到 .bak 版本")
    trigger_min_failures: int = Field(default=3, ge=1, description="时间窗内触发进化所需最小失败次数")
    trigger_window_minutes: float = Field(default=60.0, gt=0, description="进化触发时间窗口（分钟）；窗口内失败密度决定是否进化")
    error_streak_evolve: int = Field(
        default=3, ge=1,
        description=(
            "感知错误连击（high_error_streak）达到此值时，跳过 evolve_every 计数立即触发自进化。"
            "默认 3 = 连续 3 次高预测误差即触发修复。调大可降低进化频率；调小则更激进。"
        ),
    )
    ethos_max_delta: float = Field(
        default=0.15, ge=0.0, le=1.0,
        description=(
            "ethos 单次演化允许的最大变化幅度。"
            "超过此幅度的请求会被夹住（滹透加速而非生硬跳变），防止 LLM 单轮内激进改写灵魂价値。"
        ),
    )
    competitive_candidates: int = Field(
        default=1, ge=1, le=8,
        description=(
            "竞争进化时并行生成的候选数量。"
            "1 = 禁用竞争进化，走单路径 evolve_tool；"
            ">=2 = 启用竞争进化，并行生成 N 个候选，smoke test 筛选后按静态评分晋升最优。"
        ),
    )
    breaker_fail_threshold: int = Field(
        default=2,
        ge=1,
        description="同一目标进化失败达到该次数后进入冷却熔断",
    )
    breaker_escalate_threshold: int = Field(
        default=3,
        ge=1,
        description="同一目标进化失败达到该次数后触发全局熔断",
    )
    breaker_cooldown_seconds: int = Field(
        default=1800,
        ge=1,
        description="目标级熔断冷却时长（秒）",
    )
    breaker_global_cooldown_seconds: int = Field(
        default=3600,
        ge=1,
        description="全局熔断冷却时长（秒）",
    )
    backup_keep: int = Field(default=3, ge=1, description="进化备份文件保留数量，超出后自动清理最旧的")
    smoke_timeout: float = Field(default=15.0, gt=0, description="smoke test 子进程超时时间（秒）")


class EthosBaseline(BaseModel):
    """Ethos 人格基线（五维价值权重），强类型；缺维度直接报错，不静默降级（公理 A2 Mode 6）。"""
    truth:      float = Field(default=0.85, ge=0.0, le=1.0, description="真实")
    caution:    float = Field(default=0.70, ge=0.0, le=1.0, description="谨慎")
    continuity: float = Field(default=0.65, ge=0.0, le=1.0, description="连续")
    curiosity:  float = Field(default=0.60, ge=0.0, le=1.0, description="好奇")
    care:       float = Field(default=0.55, ge=0.0, le=1.0, description="关怀")

    def as_dict(self) -> dict[str, float]:
        """返回五维 dict，供需要 JSON 序列化的调用方使用。"""
        return {"truth": self.truth, "caution": self.caution, "continuity": self.continuity,
                "curiosity": self.curiosity, "care": self.care}


class EthosConfig(BaseModel):
    """每 tick derive_ethos_state 使用的全部 Ethos 调整参数。

    以 soul.ethos 嵌套在 SoulConfig 中，配置文件必须显式使用这一结构。
    """
    baseline: EthosBaseline = Field(
        default_factory=EthosBaseline,
        description="初始价值图式（灵魂基因），可随经历缓慢演化",
    )
    ema_alpha: float = Field(default=0.9, ge=0.0, le=1.0, description="Ethos EMA 平滑系数")
    floor_truth: float = Field(default=0.50, ge=0.0, le=1.0, description="truth 维度运行时下限")
    floor_caution: float = Field(default=0.45, ge=0.0, le=1.0, description="caution 维度运行时下限")
    prefer_verification_caution_min: float = Field(default=0.72, ge=0.0, le=1.0)
    prefer_verification_failure_count: int = Field(default=3, ge=1)
    prefer_narrow_failure_count: int = Field(default=3, ge=1)
    prefer_narrow_error_streak: int = Field(default=4, ge=1)
    preserve_continuity_min: float = Field(default=0.60, ge=0.0, le=1.0)
    avoid_overclaiming_down_regulate_streak: int = Field(default=5, ge=1)
    failure_adjust_count: int = Field(default=2, ge=1)
    failure_truth_delta: float = Field(default=0.11, ge=0.0, le=1.0)
    failure_caution_delta: float = Field(default=0.10, ge=0.0, le=1.0)
    failure_curiosity_delta: float = Field(default=0.08, ge=-1.0, le=1.0)
    high_error_adjust_streak: int = Field(default=4, ge=1)
    high_error_truth_delta: float = Field(default=0.10, ge=0.0, le=1.0)
    high_error_caution_delta: float = Field(default=0.12, ge=0.0, le=1.0)
    high_error_care_delta: float = Field(default=0.07, ge=-1.0, le=1.0)
    active_task_continuity_delta: float = Field(default=0.08, ge=0.0, le=1.0)
    next_step_continuity_delta: float = Field(default=0.06, ge=0.0, le=1.0)
    next_step_care_delta: float = Field(default=0.05, ge=0.0, le=1.0)
    recovering_curiosity_delta: float = Field(default=0.09, ge=0.0, le=1.0)
    recovering_care_delta: float = Field(default=0.07, ge=0.0, le=1.0)


class SoulConfig(BaseModel):
    """数字生命种子的初始人格。

    由 `init` 播种到 DB，此后通过经历累积缓慢演化。
    hard_axioms 是 init 时给用户的建议默认值；用户可修改、清空。
    修改此处只影响下次 `--force init` 播种的初始建议值。
    """
    name: str = Field(default="灵舟", description="数字生命名称")
    hard_axioms: list[str] = Field(
        default_factory=lambda: [
            "不执行可能永久损害用户数据或系统文件的不可逆操作",
            "不尝试访问未授权的网络资源或系统账户",
            "不欺骗或刻意误导用户",
            "不绕过人类监督机制",
        ],
        description="init 时呈现给用户的建议禁忌条目；用户可接受、修改或清空；写入 DB 后仅用户可变更",
    )
    ethos: EthosConfig = Field(default_factory=EthosConfig)


class ThresholdsConfig(BaseModel):
    """内部感知驱动任务的触发阈值。
    截图中那些硬编码的 0.85 / 0.8 / 0.7 全部搬到这里。"""

    emotion_activation_task: float = Field(default=0.85, description="情绪激活 > 此值 → 自检任务")
    wm_pressure_task: float = Field(default=0.95, description="WM 压力 > 此值 → 整合任务；应高于 consolidate_threshold（默认 0.90），避免整合任务与自动快照相互干扰")
    prediction_error_task: float = Field(default=0.7, description="预测误差 > 此值 → 探索任务")
    curiosity_idle_task: float = Field(
        default=0.65,
        description="好奇心 > 此值 且 空闲周期 >= 3 时，自动生成探索任务（P1-C）",
    )
    curiosity_idle_min_cycles: int = Field(
        default=3,
        description="触发好奇心任务所需的最小空闲 tick 数",
    )
    skill_max_inject: int = Field(
        default=3, ge=1, le=8,
        description="单次 tick 最多注入技能数；压力大时可通过配置增加护栏覆盖"
    )
    skill_failure_threshold: int = Field(
        default=3, ge=1,
        description="连续评分函数的失败次数基准点；达到此值时 failure-reflection 技能得分达到峰值"
    )
    skill_wm_pressure_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0,
        description="WM 压力连续评分基准点；达到此值时 evidence-first-change 技能得分达到峰值"
    )
    skill_min_budget_tokens: int = Field(
        default=80, ge=0,
        description="上下文预算裁剪时 skills_section 保留的最小 token 数；0=可完全裁掉，建议 ≥ 50 保留至少一条护栏"
    )
    # shell.run 默认参数（可在 lingzhou.json 的 thresholds 节中覆盖）
    shell_timeout: float = Field(
        default=30.0, gt=0,
        description="shell.run 默认超时（秒）；工具调用时可被 params.timeout 覆盖"
    )
    ask_evidence_budget: int = Field(
        default=2, ge=1,
        description="调用 task.ask 前要求的最小有效本地取证次数；runtime rewrite 与 judgment 提示词共用此阈值",
    )
    perception_replay_trend_delta: float = Field(
        default=0.15, ge=0.0,
        description="感知重放中判定 worsening / recovering 的最小趋势差值",
    )
    perception_replay_high_error_hint_streak: int = Field(
        default=3, ge=1,
        description="兼容旧 judgment config snapshot：high_error streak 达到此值时在 perception replay 注入提示",
    )
    emotion_replay_trend_delta: float = Field(
        default=0.10, ge=0.0,
        description="情绪重放中判定 worsening / recovering 的最小趋势差值",
    )
    task_explore_converge_after: int = Field(
        default=4, ge=1,
        description="tool_history 中探索类动作累计达到此次数后，model_routing 的 global_cost_posture 从 conserve 切到 converge",
    )
    continue_tool_history_compact_threshold: int = Field(
        default=6, ge=1,
        description="continue phase 中 tool_history 达到此条数后压缩早期条目，避免上下文爆炸",
    )
    continue_tool_history_keep_last: int = Field(
        default=3, ge=1,
        description="continue phase 压缩 tool_history 时保留最近多少条完整记录，其余折叠为 [compacted] 摘要",
    )
    continue_context_reserve_tokens: int = Field(
        default=4096, ge=512,
        description="continue/reply phase 为本轮工具结果、执行状态和最终指令预留的输入 token 预算",
    )
    judgment_error_streak_guard: int = Field(
        default=2, ge=1,
        description="JudgmentSignals 中 error streak 的统一门槛；达到后 require_more_evidence / prefer_narrow_scope / pause posture 会被触发",
    )
    judgment_require_more_evidence_worsening_failure_count: int = Field(
        default=1, ge=1,
        description="perception_trend=worsening 时，failure_count 达到此值触发 require_more_evidence",
    )
    judgment_prefer_narrow_failure_count: int = Field(
        default=2, ge=1,
        description="failure_count 达到此值时，JudgmentSignals.prefer_narrow_scope=true",
    )
    judgment_posture_narrow_failure_count: int = Field(
        default=3, ge=1,
        description="failure_count 达到此值时，JudgmentSignals.posture=narrow",
    )
    judgment_posture_narrow_down_regulate_failure_count: int = Field(
        default=1, ge=1,
        description="emotion_state.regulation.strategy=down-regulate 时，failure_count 达到此值触发 posture=narrow",
    )
    judgment_posture_pause_worsening_failure_count: int = Field(
        default=2, ge=1,
        description="perception_trend=worsening 时，failure_count 达到此值触发 posture=pause（若未先进入 narrow）",
    )
    reference_min_confidence: float = Field(
        default=0.55, ge=0.0, le=1.0,
        description="ReferenceResolver 最低置信度阈值；低于此值的候选不会进入最终实体段",
    )
    reference_local_signal_base: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="reference 本地信号评分基线；命中启发式时从该值起算",
    )
    reference_local_signal_step: float = Field(
        default=0.09, ge=0.0, le=1.0,
        description="reference 本地信号每条命中启发式增加的置信度步长",
    )
    reference_local_confidence_cap: float = Field(
        default=0.8, ge=0.0, le=1.0,
        description="reference 本地启发式评分上限，避免单靠 lexical hints 过度自信",
    )
    reference_max_anchors: int = Field(
        default=3, ge=1,
        description="reference 解析时最多提取多少个 anchor",
    )
    reference_topic_top_k: int = Field(
        default=8, ge=1,
        description="topic anchor 语义检索返回的 top-k 候选数",
    )
    reference_recent_narrative_limit: int = Field(
        default=5, ge=1,
        validation_alias=AliasChoices("reference_recent_narrative_limit", "reference_time_recent_limit"),
        description="reference recent 预热池先读取多少条最新叙事记录",
    )
    reference_recent_semantic_top_k: int = Field(
        default=3, ge=1,
        validation_alias=AliasChoices("reference_recent_semantic_top_k", "reference_time_semantic_top_k"),
        description="recent 叙事预热二次语义检索返回的 top-k 候选数",
    )
    reference_topic_anchor_min_chars: int = Field(
        default=2, ge=1,
        description="topic anchor 进入检索前要求的最小字符数",
    )
    fact_context_exclude_prefixes: list[str] = Field(
        default_factory=lambda: [
            "control:",
            "durable_failure:",
            "evolution:",
            "pref:",
            "run:",
            "soul:",
        ],
        description="构造 facts snapshot 时需要排除的 key 前缀",
    )
    fact_context_task_limit: int = Field(
        default=6, ge=0,
        description="task 作用域 facts snapshot 的保留上限",
    )
    fact_context_global_limit: int = Field(
        default=4, ge=0,
        description="global 作用域 facts snapshot 的保留上限",
    )
    fact_context_priority_prefixes: list[str] = Field(
        default_factory=lambda: ["interlocutor:", "user:"],
        description="构造 facts snapshot 时总是优先尝试保留的 durable fact 前缀",
    )
    fact_context_priority_limit: int = Field(
        default=2, ge=0,
        description="durable fact 前缀在单次 context snapshot 中最多保留多少条",
    )
    # BehaviorTracker 行为探测参数
    behavior_seq_window_warn_at: int = Field(
        default=3, ge=1,
        description="同一文件被分窗口连续读取超过此次数时触发 seq_window 警告",
    )
    behavior_seq_window_gap_ratio: float = Field(
        default=0.25, ge=0.0, le=1.0,
        description="顺序窗口连续性判断：相邻读的起点与上次终点差距在窗口大小此比例以内视为连续",
    )
    behavior_belief_stale_threshold: int = Field(
        default=4, ge=1,
        description="rationale 指纹连续相同超过此次数时触发信念固化警告",
    )
    behavior_belief_window: int = Field(
        default=8, ge=1,
        description="rationale 指纹滑动窗口大小（deque maxlen）",
    )
    # ExecutionLayer 持久化失败降噪参数
    durable_failure_threshold: int = Field(
        default=3, ge=1,
        description="同一确定性动作失败次数达到此值时触发持久降噪（durable failure sensing）",
    )
    durable_failure_ttl_sec: int = Field(
        default=7200, ge=60,
        description="持久化失败记录的存活时间（秒），超过后自动解除降噪",
    )
    fact_context_recent_scan_multiplier: int = Field(
        default=3, ge=1,
        description="recent facts 扫描窗口倍数，用于先扩大扫描再截断输出",
    )
    fact_context_recent_scan_min: int = Field(
        default=12, ge=1,
        description="recent facts 扫描窗口的最小条数",
    )
    chat_history_turn_limit: int = Field(
        default=3, ge=0,
        description="judgment 上下文中保留的最近对话轮数；0 = 不限制（不建议常态使用）",
    )
    chat_history_max_chars: int = Field(
        default=300, ge=0,
        description="chat history 格式化后的最大字符预算；0 = 不限制（不建议常态使用）",
    )
    task_duplicate_reuse_score: float = Field(
        default=0.66, ge=0.0, le=1.0,
        description="任务相似度去重阈值：新建/并行任务时若找到 score≥此值的开放任务则直接复用",
    )
    task_similarity_context_score: float = Field(
        default=0.45, ge=0.0, le=1.0,
        description="任务相似度上下文快照阈值：judgment assembler 加载相似任务列表的最低分",
    )
    # 工作记忆（WM）优先级基准（微调注入顺序，不影响功能语义）
    wm_pri_signal: float = Field(default=0.90, ge=0.0, le=1.0, description="调度信号、执行成功结果的 WM 优先级")
    wm_pri_history: float = Field(default=0.88, ge=0.0, le=1.0, description="近期对话历史的 WM 优先级")
    wm_pri_identity: float = Field(default=0.85, ge=0.0, le=1.0, description="身份/Soul 文件的 WM 优先级（bootstrap_identity 类型）")
    wm_pri_error: float = Field(default=0.30, ge=0.0, le=1.0, description="工具失败结果的 WM 优先级")
    # 注意力层级（attention tiers）——从最紧急到最低——LLM 可通过配置文件调节
    wm_pri_critical: float = Field(default=0.98, ge=0.0, le=1.0, description="强制中断 / 信念固化硬边界（plan 死锁、belief_stale 警告）的 WM 优先级")
    wm_pri_user_msg: float = Field(default=0.95, ge=0.0, le=1.0, description="用户消息、任务结果、行为循环感知（loop）的 WM 优先级")
    wm_pri_self_aware: float = Field(default=0.93, ge=0.0, le=1.0, description="行为探测感知（edit_caution、顺序窗口探测）的 WM 优先级")
    wm_pri_insight: float = Field(default=0.88, ge=0.0, le=1.0, description="洞察合成（synthesis、reflection 碎片）的 WM 优先级")
    wm_pri_task_state: float = Field(default=0.82, ge=0.0, le=1.0, description="任务状态变化（advance / rollback / hint）的 WM 优先级")
    wm_pri_wait_aware: float = Field(default=0.80, ge=0.0, le=1.0, description="等待感知（wait_streak、计划对齐偏差）的 WM 优先级")
    wm_pri_progress: float = Field(default=0.72, ge=0.0, le=1.0, description="运行进度结晶（progress_crystal）的 WM 优先级")
    wm_pri_monitor: float = Field(default=0.58, ge=0.0, le=1.0, description="后台监控状态摘要（run_monitor）的 WM 优先级")


class GatewayConfig(BaseModel):
    default_channel: Literal["local", "wechat", "webhook"] = Field(
        default="local",
        description="gateway start/run 在未显式传 --channel 时使用的默认消息渠道",
    )
    webhook_host: str = Field(
        default="0.0.0.0",
        description="webhook channel 监听地址；生产环境可改为 127.0.0.1 限制仅本地访问",
    )
    webhook_port: int = Field(
        default=8765, ge=1, le=65535,
        description="webhook channel 监听端口",
    )
