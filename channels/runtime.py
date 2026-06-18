from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from .webhook import describe_webhook_channel, start_webhook_channel
from .wechat import describe_wechat_channel, start_wechat_channel

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_HandlerT = TypeVar("_HandlerT")

_CHANNEL_DESCRIBERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "wechat": describe_wechat_channel,
    "webhook": describe_webhook_channel,
}

_CHANNEL_STARTERS: dict[str, Callable[[dict[str, Any], str | Path], object]] = {
    "wechat": start_wechat_channel,
    "webhook": start_webhook_channel,
}


def _channel_handler(channel: str, handlers: dict[str, _HandlerT]) -> _HandlerT:
    handler = handlers.get(channel)
    if handler is None:
        raise ValueError(f"unsupported channel runtime: {channel}")
    return handler


def describe_channel_runtime(channel: str, channel_cfg: dict[str, Any]) -> str:
    describer = _channel_handler(channel, _CHANNEL_DESCRIBERS)
    return describer(channel_cfg)


def start_channel_runtime(channel: str, channel_cfg: dict[str, Any], db_path: str | Path) -> object:
    starter = _channel_handler(channel, _CHANNEL_STARTERS)
    return starter(channel_cfg, db_path)
