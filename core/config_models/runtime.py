"""core.config_models.runtime — prompts/memory 与运行时记忆辅助函数。"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PromptsConfig(BaseModel):
    """提示词文件路径。支持相对路径（相对于 lingzhou.json）或绝对路径。
    修改 prompt 行为只需换文件，不需要改代码。"""

    system: str = "prompts/system.md"
    judgment: str = "prompts/judgment.md"
    evolution: str = "prompts/evolution.md"


class MemoryConfig(BaseModel):
    """记忆层配置。

    运行时调优请在 lingzhou.json 的 memory 区块覆盖，不要直接修改此处的 default 值。
    自进化机制可以修改 lingzhou.json 配置，但禁止将 Field default 写入源码。
    """
    working_capacity: int = Field(default=40, ge=1, description="工作记忆最大条目数（条目数上限兑底）")
    wm_token_budget_ratio: float = Field(
        default=0.15, ge=0.001, le=0.5,
        description=(
            "工作记忆 token 预算占 judgment 输入预算的比例（自动随模型 context window 伸缩）。"
            "默认 0.15（15%）：GPT-5.4 400K → ~45K；Qwen3.6-plus 1M → ~112K。"
            "pressure = total_wm_tokens / effective_wm_token_budget()。"
        ),
    )
    wm_item_max_tokens: int = Field(
        default=0, ge=0,
        description=(
            "工作记忆单条 content token 上限（粗估）；超出时自动截断并追加省略提示。"
            "0 = 不限制。调优请在 lingzhou.json 的 memory 区块覆盖，不要修改此处 default 值。"
        ),
    )
    episodic_n_recent: int = Field(default=20, ge=1, description="注入 context 的情节记忆块数上限（每块 = 一条完整交互事件，--- 分隔）；越新越靠后（recency bias，Murdock 1962）；Tulving 1983 episode unit")
    semantic_top_k: int = Field(default=5, ge=1, description="语义检索返回条目数")
    failure_limit: int = Field(default=10, ge=1, description="注入 bundle 的失败记录数")
    consolidate_threshold: float = Field(default=0.90, ge=0.0, le=1.0, description="WM 压力超过此値触发整合；提高阈値可减少快照频率，让更多证据在 WM 中存活更长")
    consolidate_low_pressure_skip_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0,
        description="maintenance 阶段的 WM 低压跳过门槛；低于此值时，即使到 consolidate 周期也可跳过整合",
    )
    convergence_bonus: float = Field(default=0.15, ge=0.0, le=1.0, description="多锚点召回的收敛奖励系数：每增加一个独立线索命中，相关度提升此比例")
    max_events: int = Field(default=500, ge=10, description="events.jsonl 最大条目数，超出后裁剪最旧记录")
    semantic_decay_lambda: float = Field(
        default=0.1,
        ge=0.0,
        le=10.0,
        description="语义记忆激活衰减率（Ebbinghaus，λ/天）；0 表示不衰减",
    )
    embedding_model: str | None = Field(
        default=None,
        description="DashScope embedding model ID（如 'text-embedding-v3'）；None=禁用向量混合检索",
    )
    embedding_weight: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="向量得分权重；(1-weight) 为 FTS5/关键词权重（仅 embedding_model 非 None 时生效）",
    )
    semantic_source_weight: float = Field(
        default=0.12, ge=0.0, le=0.5,
        description="语义检索中稳定长期来源/类型的额外权重；值越高，长期沉淀越不容易被短期事件压过",
    )
    semantic_temporal_weight: float = Field(
        default=0.08, ge=0.0, le=0.5,
        description="语义检索中的时间权重；长期记忆获得存活加分，短期事件只获得较轻的新近加分",
    )
    semantic_temporal_window_days: float = Field(
        default=7.0, gt=0.0, le=90.0,
        description="语义检索时间权重的标准时间窗（天）",
    )

    daily_recall_days: int = Field(
        default=2, ge=1, le=14,
        description="daily 补短检索的最近天数窗口；仅在长期记忆命中不足时使用",
    )
    daily_recall_max_chars: int = Field(
        default=0, ge=0,
        description="daily 补短片段的最大字符数预算；0 = 不限制总字符预算，仍只返回检索命中的 evidence 片段",
    )
    daily_recall_semantic_score_threshold: float = Field(
        default=0.55, ge=0.0, le=1.5,
        description="长期语义记忆 top score 达到此值时，认为本轮无需再注入 daily 补短",
    )
    daily_summary_days: int = Field(
        default=7, ge=1, le=30,
        description="weekly daily summary 汇总最近多少天的 daily 轨迹",
    )
    daily_summary_max_chars: int = Field(
        default=0, ge=0,
        description="weekly daily summary 写入长期语义层时的最大字符数预算；0 = 不限制",
    )
    daily_summary_activation: float = Field(
        default=0.78, ge=0.0, le=1.0,
        description="weekly daily summary 进入语义记忆时的 activation",
    )
    daily_summary_importance: float = Field(
        default=0.82, ge=0.0, le=1.0,
        description="weekly daily summary 进入语义记忆时的 importance",
    )
    local_embed_model: str | None = Field(
        default=None,
        description=(
            "本地 SentenceTransformer 模型名（如 'BAAI/bge-m3'）；"
            "设置后优先于 API embedding_model，需安装 sentence-transformers；None=不使用本地模型"
        ),
    )
    local_embed_cache_dir: str | None = Field(
        default=None,
        description=(
            "本地模型 HuggingFace 缓存目录（如 '/root/.cache/huggingface/hub'）；"
            "None=使用系统默认 ~/.cache"
        ),
    )
    chat_crystallize_every: int = Field(
        default=20, ge=1,
        description="chat 结晶间隔：每 N 个同 chat 轮次蒸馏一次 chat_summary 节点写入语义记忆；任务 event 仍独立保留",
    )
    promotion_priority_threshold: float = Field(
        default=0.78, ge=0.0, le=1.0,
        description="WM 整合时提升到语义记忆的最低优先级；低于此值时仅 allowlist kind 会被长期化",
    )
    promotion_max_nodes_per_consolidation: int = Field(
        default=6, ge=0,
        description="单次 consolidate 最多写入多少个长期语义节点，避免短期噪声淹没长期层",
    )
    promotion_min_chars: int = Field(
        default=24, ge=1,
        description="WM 条目正文短于此长度时默认不提升为语义节点，避免长期层充满碎片",
    )
    promotion_body_max_chars: int = Field(
        default=0, ge=0,
        description="提升到语义记忆的单节点正文上限（字符）；0 = 不限制",
    )
    promotion_reinforce_delta: float = Field(
        default=0.05, ge=0.0, le=0.5,
        description="重复命中同一长期节点时额外增加的 activation，表示再巩固",
    )
    promotion_semantic_kinds: list[str] = Field(
        default_factory=lambda: [
            "self_awareness",
            "behavior_sense",
            "task_reflection",
            "meta_reflection",
            "task_replan",
            "routing_guard",
            "task_result",
            "progress_crystal",
            "execute_result",
            "run_monitor",
            "probe_result",
            "subagent_result",
            "skill_activation",
            "skill_evolution",
            "skill_synthesis",
            "self_drive",
            "crash_recovery",
        ],
        description="即使优先级不足，也允许直接提升到语义记忆的 WM kind 白名单",
    )
    promotion_fact_kinds: list[str] = Field(
        default_factory=lambda: ["user_message"],
        description="允许从中抽取 durable facts 的 WM kind 白名单",
    )
    global_md_warn_bytes: int = Field(
        default=80000, ge=1,
        description="global.md 体积告警阈值（字节）；超过后在 maintenance 阶段向 WM 注入记忆压力感知信号",
    )
    global_md_warn_lines: int = Field(
        default=600, ge=1,
        description="global.md 行数告警阈值；超过后在 maintenance 阶段向 WM 注入记忆压力感知信号",
    )
    run_result_success_activation: float = Field(
        default=0.72, ge=0.0, le=1.0,
        description="语义记忆中 run_result 成功节点的 activation；值越高，后续检索越容易回想成功执行证据",
    )
    run_result_failure_activation: float = Field(
        default=0.82, ge=0.0, le=1.0,
        description="语义记忆中 run_result 失败节点的 activation；默认高于成功，用于放大失败证据的可回忆性",
    )
    run_result_success_valence: float = Field(
        default=0.65, ge=0.0, le=1.0,
        description="语义记忆中 run_result 成功节点的 valence；供后续检索与情绪线索参考",
    )
    run_result_failure_valence: float = Field(
        default=0.35, ge=0.0, le=1.0,
        description="语义记忆中 run_result 失败节点的 valence；默认低于成功，用于保留负反馈线索",
    )


def run_result_memory_affect(memory_cfg: Any | None, *, is_failure: bool) -> tuple[float, float]:
    cfg = memory_cfg if memory_cfg is not None else MemoryConfig()
    if is_failure:
        return (
            float(getattr(cfg, "run_result_failure_activation", 0.82)),
            float(getattr(cfg, "run_result_failure_valence", 0.35)),
        )
    return (
        float(getattr(cfg, "run_result_success_activation", 0.72)),
        float(getattr(cfg, "run_result_success_valence", 0.65)),
    )
