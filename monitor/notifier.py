"""Notification helpers for console, file, and webhook delivery."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from loguru import logger

from core.i18n import get_runtime_language, text_for
from core.storage import Storage


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Notifier:
    """Notification manager."""

    def __init__(self, storage: Storage):
        self.storage = storage
        self.channels: list[NotificationChannel] = []

    def add_channel(self, channel: "NotificationChannel"):
        self.channels.append(channel)
        logger.info(f"Notification channel added: {channel.name}")

    def notify(
        self,
        event_type: str,
        title: str,
        body: str,
        level: str = "info",
    ):
        for channel in self.channels:
            if not channel.accepts(level):
                continue
            try:
                channel.send(event_type, title, body, level)
            except Exception as exc:
                logger.error(f"Notification failed ({channel.name}): {exc}")

    def notify_trade_open(
        self,
        symbol: str,
        direction: str,
        price: float,
        confidence: float,
        rationale: str,
    ):
        lang = get_runtime_language(self.storage)
        self.notify(
            "trade_open",
            text_for(lang, f"开仓: {symbol} {direction}", f"Open: {symbol} {direction}"),
            text_for(
                lang,
                f"价格: ${price:,.2f} | 置信度: {confidence:.0%}\n理由: {rationale[:120]}",
                f"Price: ${price:,.2f} | Confidence: {confidence:.0%}\nReason: {rationale[:120]}",
            ),
            level="info",
        )

    def notify_trade_close(
        self,
        symbol: str,
        price: float,
        pnl: float,
        pnl_pct: float,
        reason: str,
    ):
        lang = get_runtime_language(self.storage)
        prefix_zh = "盈利" if pnl >= 0 else "亏损"
        prefix_en = "Profit" if pnl >= 0 else "Loss"
        self.notify(
            "trade_close",
            text_for(lang, f"{prefix_zh}: {symbol}", f"{prefix_en}: {symbol}"),
            text_for(
                lang,
                (
                    f"价格: ${price:,.2f}\n"
                    f"盈亏: ${pnl:+,.2f} ({pnl_pct:+.2f}%)\n"
                    f"原因: {reason}"
                ),
                (
                    f"Price: ${price:,.2f}\n"
                    f"PnL: ${pnl:+,.2f} ({pnl_pct:+.2f}%)\n"
                    f"Reason: {reason}"
                ),
            ),
            level="success" if pnl >= 0 else "warning",
        )

    def notify_stop_loss(self, symbol: str, price: float, loss_pct: float):
        lang = get_runtime_language(self.storage)
        self.notify(
            "stop_loss",
            text_for(lang, f"止损触发: {symbol}", f"Stop Loss Triggered: {symbol}"),
            text_for(
                lang,
                f"当前价格: ${price:,.2f}\n亏损: {loss_pct:.2f}%",
                f"Current Price: ${price:,.2f}\nLoss: {loss_pct:.2f}%",
            ),
            level="error",
        )

    def notify_daily_report(
        self,
        positions: list,
        trades: list,
        balance: float,
    ):
        lang = get_runtime_language(self.storage)
        total_pnl, total_trades, win_trades = self._daily_realized_summary()
        win_rate = win_trades / total_trades if total_trades else 0
        now = utc_now()
        body = text_for(
            lang,
            (
                f"CryptoAI 每日交易报告 ({now.strftime('%Y-%m-%d')})\n\n"
                f"权益: ${balance:,.2f}\n"
                f"今日盈亏: ${total_pnl:+,.2f}\n"
                f"今日交易: {total_trades} | 胜率: {win_rate:.0%} ({win_trades}/{total_trades})\n"
                f"当前持仓: {len(positions)}\n"
            ),
            (
                f"CryptoAI Daily Trading Report ({now.strftime('%Y-%m-%d')})\n\n"
                f"Equity: ${balance:,.2f}\n"
                f"Today's PnL: ${total_pnl:+,.2f}\n"
                f"Today's Trades: {total_trades} | Win Rate: {win_rate:.0%} ({win_trades}/{total_trades})\n"
                f"Open Positions: {len(positions)}\n"
            ),
        )
        for position in positions:
            body += (
                f"- {position['symbol']} {position['direction']} "
                f"@ ${position['entry_price']:,.2f}\n"
            )
        self.notify(
            "daily_report",
            text_for(lang, "每日交易报告", "Daily Trading Report"),
            body,
            level="info",
        )

    def _daily_realized_summary(self) -> tuple[float, int, int]:
        today_start = utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
        with self.storage._conn() as conn:
            rows = conn.execute(
                """
                SELECT payload_json FROM execution_events
                WHERE event_type IN ('close', 'live_close') AND created_at >= ?
                ORDER BY created_at DESC
                """,
                (today_start.isoformat(),),
            ).fetchall()
        total_pnl = 0.0
        total_trades = 0
        win_trades = 0
        for row in rows:
            try:
                payload = json.loads(row["payload_json"] or "{}")
            except Exception:
                continue
            incremental = payload.get("incremental_pnl")
            if incremental is None:
                incremental = payload.get("pnl", 0.0)
            try:
                pnl = float(incremental or 0.0)
            except (TypeError, ValueError):
                continue
            total_pnl += pnl
            total_trades += 1
            win_trades += int(pnl > 0)
        return total_pnl, total_trades, win_trades

    def notify_analysis_result(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        rationale: str,
    ):
        lang = get_runtime_language(self.storage)
        action_zh = "建议开仓" if direction == "LONG" and confidence >= 0.65 else "继续观望"
        action_en = "Open position suggested" if direction == "LONG" and confidence >= 0.65 else "Keep observing"
        self.notify(
            "analysis",
            text_for(lang, f"分析结果: {symbol}", f"Analysis Result: {symbol}"),
            text_for(
                lang,
                f"方向: {direction} | 置信度: {confidence:.0%} | {action_zh}\n{rationale[:160]}",
                f"Direction: {direction} | Confidence: {confidence:.0%} | {action_en}\n{rationale[:160]}",
            ),
            level="info",
        )


class NotificationChannel:
    """Base notification channel."""

    name = "base"

    def accepts(self, level: str) -> bool:
        return True

    def send(
        self,
        event_type: str,
        title: str,
        body: str,
        level: str = "info",
    ):
        raise NotImplementedError


class ConsoleChannel(NotificationChannel):
    """Console notification channel."""

    name = "console"

    def send(
        self,
        event_type: str,
        title: str,
        body: str,
        level: str = "info",
    ):
        colors = {
            "info": "\033[36m",
            "success": "\033[32m",
            "warning": "\033[33m",
            "error": "\033[31m",
            "critical": "\033[35m",
        }
        color = colors.get(level, "\033[0m")
        reset = "\033[0m"
        print(f"{color}[{event_type}] {title}{reset}\n{body}")


class FileChannel(NotificationChannel):
    """File notification channel."""

    name = "file"

    def __init__(self, log_dir: str = "data/notifications"):
        from pathlib import Path

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def send(
        self,
        event_type: str,
        title: str,
        body: str,
        level: str = "info",
    ):
        now = utc_now()
        log_file = self.log_dir / f"notify_{now.strftime('%Y-%m-%d')}.log"
        with open(log_file, "a", encoding="utf-8") as handle:
            handle.write(
                f"[{now.isoformat()}] [{event_type}] [{level}] {title}\n"
                f"{body}\n\n"
            )


class WebhookChannel(NotificationChannel):
    """Generic webhook channel."""

    name = "webhook"

    def __init__(self, url: str, secret: str = ""):
        self.url = url
        self.secret = secret

    def send(
        self,
        event_type: str,
        title: str,
        body: str,
        level: str = "info",
    ):
        import base64
        import hashlib
        import hmac as hmac_mod
        import time

        import requests

        payload = {
            "event_type": event_type,
            "title": title,
            "body": body,
            "level": level,
            "timestamp": utc_now().isoformat(),
        }
        headers = {"Content-Type": "application/json"}
        if self.secret:
            ts = str(int(time.time()))
            string_to_sign = f"{ts}\n{self.secret}"
            hmac_code = hmac_mod.new(
                string_to_sign.encode("utf-8"),
                digestmod=hashlib.sha256,
            ).digest()
            headers["X-Timestamp"] = ts
            headers["X-Signature"] = base64.b64encode(hmac_code).decode("utf-8")

        response = requests.post(
            self.url,
            json=payload,
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()


class CriticalWebhookChannel(WebhookChannel):
    """Webhook channel for high-severity alerts only."""

    name = "critical_webhook"

    def accepts(self, level: str) -> bool:
        return level in {"critical", "error"}


class FeishuWebhookChannel(NotificationChannel):
    """Feishu custom bot webhook channel."""

    name = "feishu_webhook"

    def __init__(self, url: str, secret: str = ""):
        self.url = url
        self.secret = secret

    def send(
        self,
        event_type: str,
        title: str,
        body: str,
        level: str = "info",
    ):
        import base64
        import hashlib
        import hmac as hmac_mod
        import time

        import requests

        timestamp = str(int(time.time()))
        payload = {
            "msg_type": "text",
            "content": {
                "text": f"[{level.upper()}] {title}\n{body}",
            },
        }
        if self.secret:
            string_to_sign = f"{timestamp}\n{self.secret}"
            signature = base64.b64encode(
                hmac_mod.new(
                    string_to_sign.encode("utf-8"),
                    digestmod=hashlib.sha256,
                ).digest()
            ).decode("utf-8")
            payload["timestamp"] = timestamp
            payload["sign"] = signature

        response = requests.post(
            self.url,
            json=payload,
            timeout=10,
        )
        response.raise_for_status()


class CriticalFeishuWebhookChannel(FeishuWebhookChannel):
    """Feishu webhook channel for high-severity alerts only."""

    name = "critical_feishu_webhook"

    def accepts(self, level: str) -> bool:
        return level in {"critical", "error"}
