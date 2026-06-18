"""core/perception/ethos.py — 价值层：EthosValues / EthosState + derive_ethos_state。

参考：Kohlberg (1969) 道德发展内化原则；McCloskey & Glucksberg (1978) 概念渐变
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.perception.emotion import clamp01

if TYPE_CHECKING:
    from core.config_models import EthosConfig


ETHOS_DIMENSIONS = ("truth", "caution", "continuity", "curiosity", "care")


@dataclass
class EthosValues:
    truth: float = 0.65         # 诚实优先
    caution: float = 0.60       # 行动前先确认
    continuity: float = 0.60    # 维持任务连续性
    curiosity: float = 0.45     # 主动感知，不被动等待
    care: float = 0.55          # 对用户数据和状态负责

    @classmethod
    def from_dict(cls, d: dict) -> EthosValues:
        """从 DB dict 转强类型。缺维度取默认值；值无法转 float 则显式 ValueError（公理 A2 Mode 6）。"""
        defaults = cls()
        kwargs: dict[str, float] = {}
        for dim in ETHOS_DIMENSIONS:
            raw = d.get(dim)
            if raw is None:
                kwargs[dim] = getattr(defaults, dim)
            else:
                try:
                    kwargs[dim] = max(0.0, min(1.0, float(raw)))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"ethos_baseline[{dim!r}] 无法转换为 float: {raw!r}（公理 A2 Mode 6）"
                    ) from exc
        return cls(**kwargs)


@dataclass
class EthosBias:
    """当前 tick 的行为倾向，用于候选动作预排名。"""
    prefer_verification: bool = False   # 优先验证类动作
    prefer_narrow_scope: bool = False   # 优先收窄范围
    preserve_continuity: bool = False   # 优先维持任务连续
    avoid_overclaiming: bool = False    # 避免过度承诺
    reasons: list[str] = field(default_factory=list[str])


@dataclass
class EthosState:
    values: EthosValues = field(default_factory=EthosValues)
    bias: EthosBias = field(default_factory=EthosBias)

    def __hash__(self) -> int:
        v = self.values
        b = self.bias
        return hash((
            v.truth, v.caution, v.continuity, v.curiosity, v.care,
            b.prefer_verification, b.prefer_narrow_scope,
            b.preserve_continuity, b.avoid_overclaiming,
        ))


def _ethos_values_from_baseline(ethos_cfg: EthosConfig, baseline: EthosValues | None) -> EthosValues:
    seed = ethos_cfg.baseline
    source = baseline or seed
    return EthosValues(
        truth=source.truth,
        caution=source.caution,
        continuity=source.continuity,
        curiosity=source.curiosity,
        care=source.care,
    )


def _blend_ethos_values(adjusted: EthosValues, baseline: EthosValues, alpha: float) -> EthosValues:
    return EthosValues(
        truth=clamp01(alpha * baseline.truth + (1 - alpha) * adjusted.truth),
        caution=clamp01(alpha * baseline.caution + (1 - alpha) * adjusted.caution),
        continuity=clamp01(alpha * baseline.continuity + (1 - alpha) * adjusted.continuity),
        curiosity=clamp01(alpha * baseline.curiosity + (1 - alpha) * adjusted.curiosity),
        care=clamp01(alpha * baseline.care + (1 - alpha) * adjusted.care),
    )


def derive_ethos_state(
    failure_count: int,
    high_error_streak: int,
    has_active_task: bool,
    has_next_step: bool,
    perception_trend: str,
    emotion_down_regulate_streak: int,
    ethos_cfg: EthosConfig,
    baseline: EthosValues | None = None,
) -> EthosState:
    """每 tick 从信号确定性推导 EthosState（含 EMA 基线混合）。

    baseline 已是强类型 EthosValues（由调用方用 EthosValues.from_dict() 转换），
    缺值使用 config seed 默认值（公理 A2 Mode 6）。
    """
    ec = ethos_cfg
    b = baseline  # 简短别名
    v = _ethos_values_from_baseline(ec, b)
    if failure_count >= ec.failure_adjust_count:
        v.truth     = clamp01(v.truth     + ec.failure_truth_delta)
        v.caution   = clamp01(v.caution   + ec.failure_caution_delta)
        v.curiosity = clamp01(v.curiosity + ec.failure_curiosity_delta)
    if high_error_streak >= ec.high_error_adjust_streak:
        v.truth   = clamp01(v.truth   + ec.high_error_truth_delta)
        v.caution = clamp01(v.caution + ec.high_error_caution_delta)
        v.care    = clamp01(v.care    + ec.high_error_care_delta)
    if has_active_task:
        v.continuity = clamp01(v.continuity + ec.active_task_continuity_delta)
    if has_next_step:
        v.continuity = clamp01(v.continuity + ec.next_step_continuity_delta)
        v.care       = clamp01(v.care       + ec.next_step_care_delta)
    if perception_trend == "recovering":
        v.curiosity = clamp01(v.curiosity + ec.recovering_curiosity_delta)
        v.care      = clamp01(v.care      + ec.recovering_care_delta)
    if b:
        v = _blend_ethos_values(v, b, ec.ema_alpha)
    v.truth   = max(v.truth,   ec.floor_truth)
    v.caution = max(v.caution, ec.floor_caution)

    bias = EthosBias()
    reasons: list[str] = []
    if v.caution > ec.prefer_verification_caution_min or failure_count >= ec.prefer_verification_failure_count:
        bias.prefer_verification = True
        reasons.append("谨慎度高，优先验证")
    if failure_count >= ec.prefer_narrow_failure_count or high_error_streak >= ec.prefer_narrow_error_streak:
        bias.prefer_narrow_scope = True
        reasons.append("多次失败，收窄操作范围")
    if v.continuity > ec.preserve_continuity_min and has_active_task:
        bias.preserve_continuity = True
        reasons.append("任务连续性优先")
    if emotion_down_regulate_streak >= ec.avoid_overclaiming_down_regulate_streak:
        bias.avoid_overclaiming = True
        reasons.append("情绪持续下调，避免过度承诺")
    bias.reasons = reasons
    return EthosState(values=v, bias=bias)
