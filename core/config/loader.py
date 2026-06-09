"""core.config.loader — 所有可配置项的单一来源（Single Source of Truth）。

规则：代码中不得出现硬编码的阈值、路径或模型名。
      所有行为参数必须通过 Config 读取，来源是 lingzhou.json。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..config_models import (
    EmotionConfig,
    EvolutionConfig,
    GatewayConfig,
    LoggingConfig,
    LoopConfig,
    MemoryConfig,
    PromptsConfig,
    ProviderDefinition,
    SoulConfig,
    ThresholdsConfig,
)
from .budget import adaptive_judgment_input_budget, context_window_input_hard_budget


class Config(BaseModel):
    """所有配置的统一入口。改行为 = 改 lingzhou.json，不改代码。"""

    model_config = ConfigDict(extra="forbid")  # 顶层配置拼写错误必须显式失败

    # ── Provider 层 ─────────────────────────────────────────────────────────
    providers: dict[str, ProviderDefinition]
    model: str = Field(
        description=(
            "模型引用，格式 'provider-name/model-id'，如 'bailian/qwen3.6-plus'。"
            "provider-name 必须在 providers 中定义。"
        )
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout: float | None = Field(
        default=None,
        gt=0,
        description=(
            "LLM 本地请求超时（秒）。默认 None = Lingzhou 不主动限制 LLM，"
            "由供应商/网关自然超时；设为正数时才启用本地 LLM 超时。"
        ),
    )
    thinking: Literal["off", "minimal", "low", "medium", "high"] = Field(
        default="off",
        description=(
            "思考深度等级。off=关闭；其余为由浅到深的思考强度。"
            "对应的 payload 参数由 provider.mode 决定："
            "openai 体系 → enable_thinking + budget_tokens（按比例计算）；"
            "copilot 体系 → reasoning_effort 字符串。"
        ),
    )
    context_window_tokens: int | None = Field(
        default=None, ge=1,
        description=(
            "Escape hatch：仅在使用 provider/models.json 未收录的模型时填写。"
            "已收录模型（如 qwen3.6-plus）省略此项，系统从内置目录自动推断。"
        ),
    )
    max_judgment_input_tokens: int | None = Field(
        default=None, ge=256,
        description=(
            "判断层工作集输入 token 上限覆盖值。默认 None = 根据模型 context window 自适应计算。"
            "设为整数时作为人工上限，低于模型窗口自动推断值时生效，超过则忽略（不会扩大预算）。"
            "建议覆盖范围：8000–64000。"
        ),
    )
    routing: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Judgment 层分阶段路由模型映射。key：'reader'、'reasoner'、'repair'。"
            "value 为 'provider/model-id' 格式。\n"
            "示例: {\"reader\": \"bailian/qwen3.6-plus\", \"reasoner\": \"copilot/gpt-5.4\"}\n"
            "未配置时所有 phase 均使用顶层 model 字段。"
        ),
    )
    model_fallbacks: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "显式模型回退链（按顺序尝试）。key 为 tier（reader/reasoner/repair），"
            "value 为 'provider/model-id' 列表。\n"
            "示例: {\"reader\": [\"bailian/qwen-plus\", \"copilot/gpt-5.4\"]}"
        ),
    )
    model_prices: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description=(
            "按量模型定价（USD / 1M tokens），用于成本追踪。"
            "key 为模型 id（不含 provider 前缀），value 含 input/output 两个字段。\n"
            "订阅制模型（如 copilot/*）无需填写，成本始终为 0。\n"
            "示例: {\"qwen3.6-plus\": {\"input\": 0.50, \"output\": 2.00}}"
        ),
    )
    run_type_routing: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "run_type → 模型档位映射（覆盖 models.json 内置默认值）。\n"
            "key 为 run_type（如 judge/chat_reply/llm/exec…），value 为档位名（reader/reasoner/repair/task_default）。\n"
            "示例: {\"judge\": \"reasoner\", \"chat_reply\": \"reader\"}"
        ),
    )

    # ── 其他配置节 ────────────────────────────────────────────────────────
    loop: LoopConfig = Field(default_factory=LoopConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    emotion: EmotionConfig = Field(default_factory=EmotionConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    soul: SoulConfig = Field(default_factory=SoulConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)

    # 配置文件所在目录，用于解析相对路径（由 load() 填充）
    _base_dir: Path = Path(".")

    @model_validator(mode="after")
    def _apply_provider_auth_profile_defaults(self) -> Config:
        """把 auth CLI 的默认 profile 约定接入 provider 配置。"""
        for name, provider in self.providers.items():
            if provider.auth_profile_id.strip():
                continue
            if provider.mode == "copilot":
                provider.auth_profile_id = "copilot:default"
            elif provider.mode == "codex":
                provider.auth_profile_id = "openai-codex:default"
            else:
                provider.auth_profile_id = f"{name}:default"
        return self

    @classmethod
    def load(cls, path: str | Path = "lingzhou.json", fallback: bool = True) -> Config:
        path = Path(path).expanduser().resolve()
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            data = _strip_config_doc_fields(data)
            cfg = cls.model_validate(data)
            cfg._base_dir = path.parent
            return cfg
        except Exception as e:
            if not fallback:
                raise
            import logging
            backup_path = path.with_suffix(path.suffix + ".lingzhou-backup")
            if backup_path.exists():
                logging.getLogger("lingzhou.config").warning(
                    f"配置文件 {path} 加载失败 ({e})，回退至备份 {backup_path}"
                )
                return cls.load(backup_path, fallback=False)
            raise RuntimeError(f"配置文件 {path} 加载失败且无可用备份: {e}") from e

    def resolve(self, raw: str) -> Path:
        """解析路径：~ 展开；相对路径以 lingzhou.json 所在目录为基准。"""
        p = Path(raw).expanduser()
        if p.is_absolute():
            return p
        return (self._base_dir / p).resolve()

    # ── Provider helpers ───────────────────────────────────────────────────

    @property
    def active_provider_name(self) -> str:
        """从 'provider/model-id' 中解析 provider 名称。"""
        parts = self.model.split("/", 1)
        if len(parts) != 2:
            raise ValueError(
                f"model 格式必须为 'provider-name/model-id'，当前值: {self.model!r}"
            )
        return parts[0]

    @property
    def active_model_id(self) -> str:
        """从 'provider/model-id' 中解析模型 ID。"""
        return self.model.split("/", 1)[1]

    @property
    def active_provider(self) -> ProviderDefinition:
        """返回当前 model ref 对应的 provider 连接配置。"""
        name = self.active_provider_name
        if name not in self.providers:
            raise ValueError(
                f"provider {name!r} 未在 providers 中定义。"
                f"可用 provider: {list(self.providers.keys())}"
            )
        return self.providers[name]

    @property
    def db_path(self) -> Path:
        return self.resolve(self.loop.db_path)

    @property
    def memory_dir(self) -> Path:
        return self.resolve(self.loop.memory_dir)

    @property
    def state_dir(self) -> Path:
        return self.resolve(self.loop.state_dir)

    @property
    def workspace_dir(self) -> Path:
        return self.resolve(self.loop.workspace_dir)

    @property
    def constitution_path(self) -> Path:
        """宪法文件路径（公理 A3：只读挂载，不可被程序改写）。"""
        return self.workspace_dir / "CONSTITUTION.md"

    def judgment_input_token_budget(self) -> int:
        """按模型上下文窗口反推 judgment 输入预算。

        优先级：
          1. lingzhou.json 的 context_window_tokens（escape hatch，用于未收录模型）
          2. provider/models.json 内置目录自动查找（按 active_model_id）
        """
        from provider.catalog import resolve_context_window  # 延迟导入，避免循环

        context_window = resolve_context_window(
            self.active_model_id,
            self.context_window_tokens,
            catalog_path=self.workspace_dir / "models.json",
        )
        if context_window is None:
            raise ValueError(
                f"模型 {self.active_model_id!r} 不在内置目录中。"
                "请在 lingzhou.json 的 context_window_tokens 显式指定上下文窗口大小。"
            )

        hard_budget = context_window_input_hard_budget(context_window)
        if self.max_judgment_input_tokens is not None:
            return min(hard_budget, self.max_judgment_input_tokens)
        return adaptive_judgment_input_budget(context_window)

    def effective_wm_token_budget(self) -> int:
        """WM token 预算 = judgment 输入预算 × wm_token_budget_ratio。

        自动随模型 context window 伸缩，无需手动配置。
        若模型不在内置目录且未配置 context_window_tokens，则回退到 8000。
        """
        try:
            ctx = self.judgment_input_token_budget()
        except ValueError:
            return 8000  # 未知模型 fallback
        return max(256, int(ctx * self.memory.wm_token_budget_ratio))

    def load_prompt(self, key: str) -> str:
        """按 key（对应 PromptsConfig 字段名）加载提示词文件内容。
        搜索顺序：
        1. config.prompts.<key> 指定的路径（相对于 lingzhou.json 所在目录）
        2. 包内置 prompts/<key>.md（lingzhou.py 同级目录）
        """
        raw_path = getattr(self.prompts, key)
        path = self.resolve(raw_path)
        if path.exists():
            return path.read_text(encoding="utf-8")
        # 回退：包内置 prompt（core/ 的上级目录下的 prompts/）
        from core.paths import project_root

        builtin = project_root() / "prompts" / f"{key}.md"
        if builtin.exists():
            return builtin.read_text(encoding="utf-8")
        raise FileNotFoundError(
            f"提示词文件不存在: {path}\n"
            f"也未找到内置回退: {builtin}\n"
            f"（config.prompts.{key} = {raw_path!r}）"
        )


def _strip_config_doc_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_config_doc_fields(item)
            for key, item in value.items()
            if key != "$schema" and not key.startswith("_doc") and not key.startswith("_comment")
        }
    if isinstance(value, list):
        return [_strip_config_doc_fields(item) for item in value]
    return value


def _format_config_doc_default(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def config_reference_defaults() -> dict[str, str]:
    """导出 docs/CONFIG.md 中核心默认值表格的源码真相。"""
    loop = LoopConfig()
    memory = MemoryConfig()
    evolution = EvolutionConfig()
    gateway = GatewayConfig()
    return {
        "loop.max_concurrent_ticks": _format_config_doc_default(loop.max_concurrent_ticks),
        "loop.max_tick_queue": _format_config_doc_default(loop.max_tick_queue),
        "loop.max_idle_gap": _format_config_doc_default(loop.max_idle_gap),
        "loop.active_idle_gap": _format_config_doc_default(loop.active_idle_gap),
        "loop.min_act_gap": _format_config_doc_default(loop.min_act_gap),
        "loop.judge_every": _format_config_doc_default(loop.judge_every),
        "loop.max_consecutive_errors": _format_config_doc_default(loop.max_consecutive_errors),
        "loop.evolve_every": _format_config_doc_default(loop.evolve_every),
        "memory.working_capacity": _format_config_doc_default(memory.working_capacity),
        "memory.max_events": _format_config_doc_default(memory.max_events),
        "memory.semantic_decay_lambda": _format_config_doc_default(memory.semantic_decay_lambda),
        "memory.embedding_weight": _format_config_doc_default(memory.embedding_weight),
        "evolution.enabled": _format_config_doc_default(evolution.enabled),
        "evolution.trigger_min_failures": _format_config_doc_default(evolution.trigger_min_failures),
        "evolution.trigger_window_minutes": _format_config_doc_default(evolution.trigger_window_minutes),
        "evolution.error_streak_evolve": _format_config_doc_default(evolution.error_streak_evolve),
        "evolution.max_attempts": _format_config_doc_default(evolution.max_attempts),
        "evolution.backup": _format_config_doc_default(evolution.backup),
        "gateway.default_channel": _format_config_doc_default(gateway.default_channel),
    }
