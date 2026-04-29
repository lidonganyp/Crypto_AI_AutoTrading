"""LLM analysis helpers for DeepSeek and Qwen."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from loguru import logger
from openai import OpenAI

from core.models import RiskLevel, SignalDirection, TradeSignal
from core.openai_client_factory import build_openai_client
from core.storage import Storage


SYSTEM_PROMPT = """你是一位专业的加密货币交易分析师。
请基于给定的行情、技术指标、情绪和持仓信息输出严格 JSON。"""

ANALYSIS_PROMPT_TEMPLATE = """请分析以下加密货币市场数据，并给出交易信号。
## 交易对
{symbol}

## K线数据 ({timeframe})
{ohlcv_summary}

## 技术指标
{technical_summary}

## 市场情绪
{sentiment_summary}

## 当前持仓
{position_summary}

请输出严格 JSON，字段包括：
direction, confidence, rationale, key_factors, risk_level,
suggested_stop_pct, suggested_target_pct
"""


class LLMAnalyzer:
    """Run configured LLMs and parse their outputs."""

    RUNTIME_FAILURE_BACKOFF_SECONDS = 300

    def __init__(
        self,
        storage: Storage,
        settings: dict,
        model_clients: dict[str, OpenAI] | None = None,
    ):
        self.storage = storage
        self.settings = dict(settings or {})
        self.models: dict[str, OpenAI] = model_clients.copy() if model_clients else {}
        self._disabled_models: set[str] = set()
        self._runtime_backoff_until: dict[str, datetime] = {}

        if not self._provider_names():
            logger.warning("No LLM API keys configured")

    def analyze(
        self,
        symbol: str,
        timeframe: str = "1h",
        ohlcv_data: list[dict] | None = None,
        technical_data: dict | None = None,
        sentiment_data: dict | None = None,
        position_data: list[dict] | None = None,
    ) -> list[TradeSignal]:
        """Analyze a symbol with all configured models."""
        signals: list[TradeSignal] = []
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            symbol=symbol,
            timeframe=timeframe,
            ohlcv_summary=self._format_ohlcv(ohlcv_data),
            technical_summary=self._format_technical(technical_data),
            sentiment_summary=self._format_sentiment(sentiment_data),
            position_summary=self._format_positions(position_data),
        )

        for model_name in self._provider_names():
            if not self._model_available(model_name):
                continue
            client = self._client_for(model_name)
            if client is None:
                continue
            model_id = self._get_model_id(model_name)
            try:
                signal = self._call_model(
                    client,
                    model_id,
                    prompt,
                    model_name,
                    symbol,
                )
                if signal:
                    signals.append(signal)
                    self.storage.insert_signal(
                        {
                            "symbol": symbol,
                            **signal.model_dump(mode="json"),
                        }
                    )
            except Exception as exc:
                self._runtime_backoff_until[model_name] = self._now() + timedelta(
                    seconds=self.RUNTIME_FAILURE_BACKOFF_SECONDS
                )
                logger.error(f"LLM analysis failed ({model_name}): {exc}")

        return signals

    def _provider_names(self) -> list[str]:
        providers: list[str] = []
        for model_name in ("deepseek", "qwen"):
            if model_name in self.models or self.settings.get(f"{model_name}_api_key"):
                providers.append(model_name)
        for model_name in self.models:
            if model_name not in providers:
                providers.append(model_name)
        return providers

    def _client_for(self, model_name: str) -> OpenAI | None:
        client = self.models.get(model_name)
        if client is not None:
            return client
        if model_name in self._disabled_models:
            return None
        api_key = self.settings.get(f"{model_name}_api_key")
        if not api_key:
            return None
        try:
            client = build_openai_client(
                api_key=api_key,
                base_url=self.settings.get(
                    f"{model_name}_api_base",
                    "https://api.deepseek.com/v1"
                    if model_name == "deepseek"
                    else "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
            )
        except Exception as exc:
            logger.error(f"LLM client init failed ({model_name}): {exc}")
            self._disabled_models.add(model_name)
            return None
        self.models[model_name] = client
        logger.info(f"{model_name} client initialized")
        return client

    def _model_available(self, model_name: str) -> bool:
        backoff_until = self._runtime_backoff_until.get(model_name)
        if backoff_until is None:
            return True
        if self._now() >= backoff_until:
            self._runtime_backoff_until.pop(model_name, None)
            return True
        return False

    def _call_model(
        self,
        client: OpenAI,
        model_id: str,
        prompt: str,
        source: str,
        symbol: str,
    ) -> TradeSignal | None:
        """Call one model and parse its output."""
        logger.info(f"Calling {source} ({model_id}) for {symbol}...")
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )

        content = response.choices[0].message.content or ""
        signal = self._parse_signal_response(content, source)
        if signal:
            logger.info(
                f"{source} signal: {signal.direction.value} "
                f"confidence={signal.confidence:.2f}"
            )
        return signal

    def _get_model_id(self, model_name: str) -> str:
        model_ids = {
            "deepseek": self.settings.get("deepseek_model", "deepseek-chat"),
            "qwen": self.settings.get("qwen_model", "qwen-max"),
        }
        return model_ids.get(model_name, model_name)

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _parse_signal_response(content: str, source: str) -> TradeSignal | None:
        """Parse plain or fenced JSON into a TradeSignal."""
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]
        content = content.strip()

        try:
            payload = json.loads(content)
            return TradeSignal(
                direction=SignalDirection(payload.get("direction", "FLAT")),
                confidence=float(payload.get("confidence", 0)),
                rationale=payload.get("rationale", ""),
                key_factors=payload.get("key_factors", []),
                risk_level=RiskLevel(payload.get("risk_level", "MEDIUM")),
                suggested_stop_pct=float(
                    payload.get("suggested_stop_pct", 0.05)
                ),
                suggested_target_pct=float(
                    payload.get("suggested_target_pct", 0.10)
                ),
                source=source,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error(f"Failed to parse {source} response: {exc}")
            logger.debug(f"Raw response: {content}")
            return None

    @staticmethod
    def _format_ohlcv(data: list[dict] | None) -> str:
        if not data:
            return "无数据"
        recent = data[:24]
        lines = []
        for candle in reversed(recent):
            dt = datetime.fromtimestamp(candle["timestamp"] / 1000, tz=timezone.utc)
            lines.append(
                f"  {dt.strftime('%m-%d %H:%M')} "
                f"O={candle['open']:.1f} H={candle['high']:.1f} "
                f"L={candle['low']:.1f} C={candle['close']:.1f} "
                f"V={candle['volume']:.0f}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_technical(data: dict | None) -> str:
        if not data:
            return "无技术指标数据"
        lines = []
        for key, value in data.items():
            if isinstance(value, list):
                lines.append(
                    f"  {key}: {', '.join(f'{item:.4f}' for item in value[-5:])}"
                )
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    @staticmethod
    def _format_sentiment(data: dict | None) -> str:
        if not data:
            return "无情绪数据"
        return (
            f"  来源: {data.get('source')} | 值: {data.get('value')} | "
            f"{data.get('summary', '')}"
        )

    @staticmethod
    def _format_positions(data: list[dict] | None) -> str:
        if not data:
            return "当前无持仓"
        lines = []
        for position in data:
            lines.append(
                f"  {position['symbol']} {position['direction']} "
                f"@ {position['entry_price']} qty={position['quantity']}"
            )
        return "\n".join(lines)
