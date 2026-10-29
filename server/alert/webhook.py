"""
Webhook 告警推送
- HTTP POST JSON 到配置的 URL
- 支持超时和自动重试
"""

import hashlib
import hmac
import json
import time
import urllib.request
import urllib.error

from loguru import logger

_LOG_BODY_LIMIT = 4096


def _business_rejection(body: str) -> str | None:
    """平台以 HTTP 200 + 业务码表示拒绝（如 {"status":404,"msg":"..."}）。

    返回拒绝原因；body 非 JSON 或无 status 字段时视为成功（None）。
    """
    try:
        obj = json.loads(body)
    except ValueError:
        return None
    if not isinstance(obj, dict) or "status" not in obj:
        return None
    code = obj["status"]
    if isinstance(code, int) and not isinstance(code, bool) and 200 <= code < 300:
        return None
    return str(obj.get("msg") or body[:200])


class WebhookAlerter:
    """通过 HTTP POST 推送告警的简单客户端。"""

    def __init__(self, url: str, timeout: float = 10, retries: int = 2,
                 secret: str | None = None):
        """
        Args:
            url: Webhook 接收地址
            timeout: 单次请求超时（秒）
            retries: 失败后重试次数
            secret: 可选 HMAC 密钥；设置后加 X-Signature/X-Timestamp
        """
        self.url = url
        self.timeout = timeout
        self.retries = retries
        self.secret = secret

    def _headers(self, data: bytes) -> dict:
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if self.secret:
            timestamp = str(int(time.time()))
            digest = hmac.new(
                self.secret.encode("utf-8"),
                f"{timestamp}.".encode("utf-8") + data,
                hashlib.sha256,
            ).hexdigest()
            headers["X-Timestamp"] = timestamp
            headers["X-Signature"] = f"sha256={digest}"
        return headers

    def send(self, payload: dict) -> bool:
        """
        发送告警。

        Args:
            payload: 告警数据字典，会被序列化为 JSON

        Returns:
            True 表示发送成功；平台业务拒绝（HTTP 2xx 但 body 的 status 非 2xx）
            时记 ERROR 并返回 False，不重试（同一 payload 再发仍会被拒）
        """
        data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        event_id = payload.get("event_id")
        alert_type = payload.get("alert_type")

        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(
                    self.url,
                    data=data,
                    headers=self._headers(data),
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    status = resp.status
                    body = resp.read(_LOG_BODY_LIMIT).decode("utf-8", errors="replace")
                    if 200 <= status < 300:
                        rejected = _business_rejection(body)
                        if rejected is not None:
                            logger.error(
                                "Webhook 被平台拒绝，不重试 (event_id={}, alert_type={}): {}",
                                event_id, alert_type, rejected,
                            )
                            return False
                        logger.info(
                            "Webhook 推送成功 (event_id={}, alert_type={}, status={}): {}",
                            event_id, alert_type, status, body[:200],
                        )
                        return True
                    else:
                        logger.warning(
                            "Webhook 返回非 2xx (attempt={}/{}, status={})",
                            attempt + 1, self.retries + 1, status,
                        )
            except urllib.error.HTTPError as e:
                logger.warning(
                    "Webhook HTTP 错误 (attempt={}/{}, status={}): {}",
                    attempt + 1, self.retries + 1, e.code, e.reason,
                )
            except urllib.error.URLError as e:
                logger.warning(
                    "Webhook 连接错误 (attempt={}/{}): {}",
                    attempt + 1, self.retries + 1, e.reason,
                )
            except Exception as e:
                logger.error("Webhook 未知错误 (attempt={}/{}): {}", attempt + 1, self.retries + 1, e)

        logger.error(
            "Webhook 发送失败（已达最大重试次数）: {} (event_id={}, alert_type={})",
            self.url, event_id, alert_type,
        )
        return False
