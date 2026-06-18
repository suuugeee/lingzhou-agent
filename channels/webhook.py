from __future__ import annotations

import json as _json
import logging
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Any

from store.task.ingress import IngressStore

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

@dataclass
class WebhookConfig:
    host: str = "0.0.0.0"   # 正式来源为 GatewayConfig.webhook_host
    port: int = 8765        # 正式来源为 GatewayConfig.webhook_port
    secret: str = ""


class WebhookChannel:
    def __init__(self, cfg: WebhookConfig, db_path: str | Path) -> None:
        self._cfg = cfg
        self._ingress = IngressStore(db_path)
        self._server: HTTPServer | None = None

    def start(self) -> None:
        cfg = self._cfg
        ingress = self._ingress

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                pass

            def do_POST(self) -> None:
                if self.path != "/message":
                    self.send_response(404)
                    self.end_headers()
                    return
                if cfg.secret and self.headers.get("Authorization", "") != f"Bearer {cfg.secret}":
                    self.send_response(401)
                    self.end_headers()
                    return
                try:
                    length = min(int(self.headers.get("Content-Length", 0)), 65536)
                except (ValueError, TypeError):
                    self.send_response(400)
                    self.end_headers()
                    return
                body = self.rfile.read(length)
                try:
                    payload = _json.loads(body)
                    msg, priority = _normalize_webhook_message(payload)
                except Exception:
                    self.send_response(400)
                    self.end_headers()
                    return
                if not msg:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'{"error":"empty message"}')
                    return

                short = _short_message(msg)
                try:
                    task_id = _enqueue_webhook_task(ingress, msg, priority)
                    resp = _json.dumps({"ok": True, "task_id": task_id}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(resp)
                    log.info("[webhook] 注入任务 id=%d: %s", task_id, short)
                except Exception as exc:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(_json.dumps({"error": str(exc)}).encode())

        self._server = HTTPServer((cfg.host, cfg.port), _Handler)
        threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="webhook-gateway",
        ).start()
        log.info("[webhook] 通道已启动 host=%s port=%d", cfg.host, cfg.port)

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        log.info("[webhook] 通道已停止")


def describe_webhook_channel(wc_cfg: dict[str, Any]) -> str:
    host = wc_cfg.get("host", "0.0.0.0")
    port = int(wc_cfg.get("port", 8765))
    secret = wc_cfg.get("secret")
    return (
        f"Webhook 监听: http://{host}:{port}/message"
        f"{'  (Bearer token)' if secret else '  (无鉴权)'}"
    )


def _normalize_webhook_message(payload: dict[str, Any]) -> tuple[str, str]:
    """从 webhook payload 中抽取 message 与 priority，兼容可选图片/语音字段。"""
    msg = str(payload.get("message", "")).strip()
    for marker, value in (
        ("图片", payload.get("images")),
        ("语音", payload.get("voices")),
        ("语音", payload.get("audio")),
        ("语音", payload.get("audios")),
    ):
        for item in _as_iterable(value):
            item_text = _format_webhook_media(item).strip()
            if item_text:
                label = f"[{marker}消息]"
                msg = f"{msg}\n{label} {item_text}" if msg else f"{label} {item_text}"
    priority = str(payload.get("priority", "high"))
    return msg.strip(), priority


def _as_iterable(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _format_webhook_media(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        try:
            return _json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


def _short_message(message: str) -> str:
    return message.replace("\n", " ")[:28] + ("..." if len(message) > 28 else "")


def _enqueue_webhook_task(ingress: IngressStore, message: str, priority: str) -> int:
    short = _short_message(message)
    return ingress.add_task(
        f"webhook: {short}",
        goal=message,
        priority=priority,
        source="gateway:webhook",
    )


def start_webhook_channel(wc_cfg: dict[str, Any], db_path: str | Path) -> WebhookChannel:
    config = WebhookConfig(
        host=str(wc_cfg.get("host", "0.0.0.0")),
        port=int(wc_cfg.get("port", 8765)),
        secret=str(wc_cfg.get("secret", "") or ""),
    )
    channel = WebhookChannel(config, db_path)
    channel.start()
    return channel
