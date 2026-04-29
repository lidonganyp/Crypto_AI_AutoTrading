"""Trade reflection and lightweight learning helpers."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone

from loguru import logger
from openai import OpenAI

from config import get_settings
from core.i18n import get_runtime_language, text_for
from core.models import MarketRegime, TradeReflection
from core.storage import Storage
from learning.experience_store import ExperienceStore


REFLECTION_SYSTEM_PROMPT = """你是一位专业的加密货币交易复盘分析师。
请基于给定交易信息，总结正确信号、错误信号、经验教训和市场状态。
输出必须是严格 JSON。"""


class TradeReflector:
    """Generate rule-based or LLM-based reflections for trades."""

    def __init__(
        self,
        storage: Storage,
        llm_client: OpenAI | None = None,
        model_id: str = "deepseek-chat",
    ):
        self.storage = storage
        self.llm_client = llm_client
        self.model_id = model_id

    def reflect_trade(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        confidence: float,
        rationale: str,
        entry_time: str,
        entry_price: float,
        exit_price: float,
        exit_time: str | None = None,
        pnl: float | None = None,
        pnl_pct: float | None = None,
        source: str = "trade",
        experience_weight: float | None = None,
    ) -> TradeReflection | None:
        """Reflect on a completed or in-flight trade."""
        logger.info(f"Reflecting on trade {trade_id} ({symbol})...")

        context = self._build_reflection_context(
            trade_id,
            symbol,
            direction,
            confidence,
            rationale,
            entry_time,
            entry_price,
            exit_price,
            exit_time,
            pnl,
            pnl_pct,
        )

        if not self.llm_client:
            return self._rule_based_reflection(
                trade_id,
                symbol,
                direction,
                confidence,
                rationale,
                pnl,
                pnl_pct,
                source=source,
                experience_weight=experience_weight,
            )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
                temperature=0.3,
                max_tokens=800,
            )

            content = (response.choices[0].message.content or "").strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
            payload = json.loads(content.strip())

            reflection = TradeReflection(
                trade_id=trade_id,
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                rationale=rationale,
                source=source,
                experience_weight=ExperienceStore.reflection_weight(
                    trade_id,
                    source,
                    experience_weight,
                ),
                realized_return_pct=pnl_pct,
                outcome_24h=pnl_pct,
                correct_signals=payload.get("correct_signals", []),
                wrong_signals=payload.get("wrong_signals", []),
                lesson=payload.get("lesson", ""),
                market_regime=self._normalize_market_regime(
                    payload.get("market_regime")
                ),
            )
            self.storage._insert_reflection(reflection)
            logger.info(f"Reflection saved: {reflection.lesson[:80]}...")
            return reflection
        except Exception as exc:
            logger.error(f"LLM reflection failed: {exc}")
            return self._rule_based_reflection(
                trade_id,
                symbol,
                direction,
                confidence,
                rationale,
                pnl,
                pnl_pct,
                source=source,
                experience_weight=experience_weight,
            )

    def _build_reflection_context(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        confidence: float,
        rationale: str,
        entry_time: str,
        entry_price: float,
        exit_price: float,
        exit_time: str | None,
        pnl: float | None,
        pnl_pct: float | None,
    ) -> str:
        """Build an LLM-friendly reflection prompt."""
        lang = get_runtime_language(self.storage)
        _ = int(datetime.fromisoformat(entry_time).timestamp() * 1000)
        past_lessons = self.storage.get_similar_lessons(symbol, direction, limit=3)
        pnl_display = f"{pnl_pct:+.2f}%" if pnl_pct is not None else "N/A"
        lessons_block = (
            "\n".join(f"- {lesson}" for lesson in past_lessons)
            if past_lessons
            else text_for(lang, "- （无历史经验）", "- (No historical lessons)")
        )

        return text_for(
            lang,
            f"""## 交易回顾

**交易ID**: {trade_id}
**交易对**: {symbol}
**方向**: {direction}
**入场时间**: {entry_time}
**入场价格**: ${entry_price:,.2f}
**置信度**: {confidence:.0%}
**决策理由**: {rationale}

**退出时间**: {exit_time or '尚未退出'}
**退出价格**: {'$' + f'{exit_price:,.2f}' if exit_price else 'N/A'}
**盈亏**: {'$' + f'{pnl:+,.2f}' if pnl is not None else 'N/A'} ({pnl_display})

## 历史类似经验
{lessons_block}

## 请分析
1. 哪些信号是正确的
2. 哪些信号是错误的
3. 最重要的一条经验教训
4. 当前市场状态
5. 置信度是否偏高或偏低
""",
            f"""## Trade Review

**Trade ID**: {trade_id}
**Symbol**: {symbol}
**Direction**: {direction}
**Entry Time**: {entry_time}
**Entry Price**: ${entry_price:,.2f}
**Confidence**: {confidence:.0%}
**Decision Rationale**: {rationale}

**Exit Time**: {exit_time or 'Not exited yet'}
**Exit Price**: {'$' + f'{exit_price:,.2f}' if exit_price else 'N/A'}
**PnL**: {'$' + f'{pnl:+,.2f}' if pnl is not None else 'N/A'} ({pnl_display})

## Historical Similar Lessons
{lessons_block}

## Please Analyze
1. Which signals were correct
2. Which signals were wrong
3. The most important lesson
4. Current market regime
5. Whether confidence was too high or too low
""",
        )

    def _rule_based_reflection(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        confidence: float,
        rationale: str,
        pnl: float | None,
        pnl_pct: float | None,
        source: str = "trade",
        experience_weight: float | None = None,
    ) -> TradeReflection:
        """Fallback reflection when no LLM is configured."""
        lang = get_runtime_language(self.storage)
        if pnl_pct is None:
            reflection = TradeReflection(
                trade_id=trade_id,
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                rationale=rationale,
                source=source,
                experience_weight=ExperienceStore.reflection_weight(
                    trade_id,
                    source,
                    experience_weight,
                ),
                realized_return_pct=pnl_pct,
                lesson=text_for(lang, "交易尚未完成，暂时无法复盘。", "Trade not finished yet, reflection unavailable."),
                market_regime=MarketRegime.UNKNOWN,
            )
            self.storage._insert_reflection(reflection)
            return reflection

        correct: list[str] = []
        wrong: list[str] = []
        if pnl_pct > 0:
            correct.append(text_for(lang, f"方向判断正确（{direction}）", f"Direction was correct ({direction})"))
            if confidence > 0.5:
                correct.append(text_for(lang, "置信度评估基本合理", "Confidence estimate was broadly reasonable"))
            else:
                wrong.append(text_for(lang, "置信度偏低", "Confidence was too low"))
            lesson = text_for(lang, f"盈利交易，收益 {pnl_pct:+.2f}%", f"Profitable trade, return {pnl_pct:+.2f}%")
        else:
            wrong.append(text_for(lang, f"方向判断错误（{direction}）", f"Direction was wrong ({direction})"))
            if confidence > 0.7:
                wrong.append(text_for(lang, "高置信度但判断错误", "Confidence was high but the decision was wrong"))
            lesson = text_for(lang, f"亏损交易，收益 {pnl_pct:+.2f}%", f"Losing trade, return {pnl_pct:+.2f}%")

        setup_profile = ExperienceStore.parse_setup_profile(rationale)
        if setup_profile.get("regime"):
            regime = self._normalize_market_regime(setup_profile.get("regime"))
        else:
            regime = self._market_regime_from_pnl(pnl_pct)
        reflection = TradeReflection(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            rationale=rationale,
            source=source,
            experience_weight=ExperienceStore.reflection_weight(
                trade_id,
                source,
                experience_weight,
            ),
            realized_return_pct=pnl_pct,
            outcome_24h=pnl_pct,
            correct_signals=correct,
            wrong_signals=wrong,
            lesson=lesson,
            market_regime=regime,
        )
        self.storage._insert_reflection(reflection)
        return reflection

    def reflect_all_open_trades(self, hours_since_entry: int = 24) -> list:
        """Reflect on open trades that have been held for long enough."""
        trades = self.storage.get_open_trades()
        positions_by_symbol = {
            position["symbol"]: position
            for position in self.storage.get_positions()
        }
        results = []
        exchange = self._build_okx_exchange()

        for trade in trades:
            entry_dt = datetime.fromisoformat(trade["entry_time"])
            elapsed = (
                datetime.now(timezone.utc) - entry_dt
            ).total_seconds() / 3600
            if elapsed < hours_since_entry:
                continue

            symbol = trade["symbol"]
            okx_sym = {
                "BTC/USDT": "BTC/USDT:USDT",
                "ETH/USDT": "ETH/USDT:USDT",
            }.get(symbol, symbol + ":USDT")

            try:
                current_price = exchange.fetch_ticker(okx_sym)["last"]
                pnl_pct = (
                    current_price / trade["entry_price"] - 1
                ) * 100
                active_position = positions_by_symbol.get(symbol)
                active_quantity = (
                    active_position["quantity"]
                    if active_position is not None
                    else trade["quantity"]
                )
                reflection = self.reflect_trade(
                    trade_id=trade["id"],
                    symbol=symbol,
                    direction=trade["direction"],
                    confidence=trade["confidence"],
                    rationale=trade.get("rationale", ""),
                    entry_time=trade["entry_time"],
                    entry_price=trade["entry_price"],
                    exit_price=current_price,
                    exit_time=datetime.now(timezone.utc).isoformat(),
                    pnl=(
                        current_price - trade["entry_price"]
                    ) * active_quantity,
                    pnl_pct=pnl_pct,
                )
                if reflection:
                    results.append(reflection)
            except Exception as exc:
                logger.error(f"Failed to reflect on {trade['id']}: {exc}")

        return results

    def generate_weekly_review(self) -> str:
        """Generate a human-readable weekly review."""
        lang = get_runtime_language(self.storage)
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)

        with self.storage._conn() as conn:
            closed_rows = conn.execute(
                """SELECT * FROM trades
                   WHERE exit_time >= ? AND status = 'closed'
                   ORDER BY exit_time DESC""",
                (week_ago.isoformat(),),
            ).fetchall()
            closed_trades = [dict(row) for row in closed_rows]

            open_rows = conn.execute(
                """SELECT * FROM trades WHERE status = 'open'
                   ORDER BY entry_time DESC""",
            ).fetchall()
            open_trades = [dict(row) for row in open_rows]

            reflection_rows = conn.execute(
                """SELECT * FROM reflections
                   WHERE created_at >= ?
                   ORDER BY created_at DESC""",
                (week_ago.isoformat(),),
            ).fetchall()
            reflections = [dict(row) for row in reflection_rows]

        consecutive_losses = self.storage.get_state("consecutive_losses") or "0"
        consecutive_wins = self.storage.get_state("consecutive_wins") or "0"
        daily_pnl = self.storage.get_state("daily_pnl_pct") or "0"
        trade_count = self.storage.get_state("daily_trade_count") or "0"

        lines = [
            text_for(lang, "# 周度复盘报告", "# Weekly Review Report"),
            text_for(lang, f"**时间范围**: {week_ago.strftime('%Y-%m-%d')} ~ {now.strftime('%Y-%m-%d')}", f"**Period**: {week_ago.strftime('%Y-%m-%d')} ~ {now.strftime('%Y-%m-%d')}"),
            text_for(lang, f"**生成时间**: {now.strftime('%Y-%m-%d %H:%M UTC')}", f"**Generated At**: {now.strftime('%Y-%m-%d %H:%M UTC')}"),
            "",
        ]

        if closed_trades:
            total = len(closed_trades)
            wins = sum(
                1 for trade in closed_trades if (trade.get("pnl_pct") or 0) > 0
            )
            losses = total - wins
            win_rate = wins / total * 100 if total else 0
            total_pnl = sum(trade.get("pnl") or 0 for trade in closed_trades)
            total_pnl_pct = sum(
                trade.get("pnl_pct") or 0 for trade in closed_trades
            )
            lines += [
                text_for(lang, "## 交易统计", "## Trade Statistics"),
                text_for(lang, f"- 总交易: {total} 笔（赢 {wins} / 亏 {losses}）", f"- Total Trades: {total} (Win {wins} / Loss {losses})"),
                text_for(lang, f"- 胜率: {win_rate:.1f}%", f"- Win Rate: {win_rate:.1f}%"),
                text_for(lang, f"- 总盈亏: ${total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)", f"- Total PnL: ${total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)"),
                "",
            ]
        else:
            lines += [text_for(lang, "## 交易统计", "## Trade Statistics"), text_for(lang, "- 本周无已关闭交易", "- No closed trades this week"), ""]

        if open_trades:
            lines.append(text_for(lang, "## 当前持仓", "## Open Positions"))
            for trade in open_trades:
                lines.append(
                    text_for(
                        lang,
                        f"- {trade['symbol']} | {trade['direction']} | 入场 ${trade['entry_price']:,.2f}",
                        f"- {trade['symbol']} | {trade['direction']} | Entry ${trade['entry_price']:,.2f}",
                    )
                )
            lines.append("")

        lines += [
            text_for(lang, "## 心理状态", "## Psychology State"),
            text_for(lang, f"- 连亏: {consecutive_losses}", f"- Consecutive Losses: {consecutive_losses}"),
            text_for(lang, f"- 连赢: {consecutive_wins}", f"- Consecutive Wins: {consecutive_wins}"),
            text_for(lang, f"- 今日盈亏: {daily_pnl}%", f"- Today's PnL: {daily_pnl}%"),
            text_for(lang, f"- 今日交易次数: {trade_count}", f"- Today's Trade Count: {trade_count}"),
            "",
        ]

        if reflections:
            lines.append(text_for(lang, "## 本周经验教训", "## Lessons This Week"))
            lessons = [item["lesson"] for item in reflections if item.get("lesson")]
            for lesson in dict.fromkeys(lessons):
                lines.append(f"- {lesson}")
            lines.append("")

            wrong_signals: list[str] = []
            for item in reflections:
                wrong = item.get("wrong_signals") or "[]"
                try:
                    wrong_signals.extend(json.loads(wrong))
                except Exception:
                    continue
            if wrong_signals:
                lines.append(text_for(lang, "## 常见错误信号", "## Common Wrong Signals"))
                for signal, count in Counter(wrong_signals).most_common(5):
                    lines.append(text_for(lang, f"- {signal} ({count} 次)", f"- {signal} ({count} times)"))
                lines.append("")

        lines.append(
            text_for(
                lang,
                f"*报告由 CryptoAI 自动生成 | {now.strftime('%Y-%m-%d %H:%M UTC')}*",
                f"*Report generated automatically by CryptoAI | {now.strftime('%Y-%m-%d %H:%M UTC')}*",
            )
        )
        return "\n".join(lines)

    def generate_daily_summary(self) -> str:
        """Generate a short daily summary."""
        lang = get_runtime_language(self.storage)
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")

        with self.storage._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM trades
                   WHERE date(entry_time) = date(?)
                   ORDER BY entry_time DESC""",
                (today,),
            ).fetchall()
            today_trades = [dict(row) for row in rows]

            prev_row = conn.execute(
                """SELECT COUNT(*) as cnt,
                          COALESCE(SUM(pnl_pct), 0) as total_pnl
                   FROM trades
                   WHERE date(exit_time) = date(?)
                   AND status = 'closed'""",
                (yesterday,),
            ).fetchone()
            yesterday_stats = dict(prev_row) if prev_row else {}

        consecutive_losses = self.storage.get_state("consecutive_losses") or "0"
        consecutive_wins = self.storage.get_state("consecutive_wins") or "0"
        daily_pnl = self.storage.get_state("daily_pnl_pct") or "0"

        lines = [text_for(lang, f"CryptoAI 每日简报 ({today})", f"CryptoAI Daily Brief ({today})"), ""]
        if today_trades:
            closed = [trade for trade in today_trades if trade["status"] == "closed"]
            open_trades = [trade for trade in today_trades if trade["status"] == "open"]
            lines.append(
                text_for(
                    lang,
                    f"今日交易: {len(today_trades)} 笔（已平 {len(closed)} / 持仓 {len(open_trades)}）",
                    f"Today's Trades: {len(today_trades)} (Closed {len(closed)} / Open {len(open_trades)})",
                )
            )
            if closed:
                total_pnl = sum(trade.get("pnl_pct") or 0 for trade in closed)
                wins = sum(
                    1 for trade in closed if (trade.get("pnl_pct") or 0) > 0
                )
                lines.append(
                    text_for(
                        lang,
                        f"今日盈亏: {total_pnl:+.2f}% (赢 {wins}/{len(closed)})",
                        f"Today's PnL: {total_pnl:+.2f}% (Win {wins}/{len(closed)})",
                    )
                )
        else:
            lines.append(text_for(lang, "今日无新交易", "No new trades today"))

        lines.append(
            text_for(
                lang,
                f"心理状态: 连亏{consecutive_losses} | 连赢{consecutive_wins} | 今日盈亏{daily_pnl}%",
                f"Psychology: losses {consecutive_losses} | wins {consecutive_wins} | today's pnl {daily_pnl}%",
            )
        )
        if yesterday_stats:
            lines.append(
                text_for(
                    lang,
                    f"昨日已平交易: {yesterday_stats.get('cnt', 0)} | 昨日收益: {yesterday_stats.get('total_pnl', 0):+.2f}%",
                    f"Yesterday's Closed Trades: {yesterday_stats.get('cnt', 0)} | Yesterday's Return: {yesterday_stats.get('total_pnl', 0):+.2f}%",
                )
            )
        return "\n".join(lines)

    def _update_stats_after_reflection(
        self,
        trade_id: str,
        pnl_pct: float | None,
    ):
        """Update aggregate reflection stats."""
        _ = trade_id
        if pnl_pct is None:
            return

        total = float(self.storage.get_state("total_pnl_pct") or "0")
        total += pnl_pct
        self.storage.set_state("total_pnl_pct", str(round(total, 4)))

        count = int(self.storage.get_state("total_trades") or "0") + 1
        self.storage.set_state("total_trades", str(count))

        wins = int(self.storage.get_state("total_wins") or "0")
        if pnl_pct > 0:
            wins += 1
        self.storage.set_state("total_wins", str(wins))

        win_rate = wins / count * 100 if count > 0 else 0
        self.storage.set_state("win_rate", str(round(win_rate, 2)))
        logger.debug(
            f"Stats updated: total_trades={count}, "
            f"win_rate={win_rate:.1f}%, total_pnl_pct={total:+.2f}%"
        )

    @staticmethod
    def _normalize_market_regime(value: str | None) -> MarketRegime:
        raw = (value or "").strip().upper()
        mapping = {
            "UPTREND": MarketRegime.UPTREND,
            "BULL_NORMAL": MarketRegime.UPTREND,
            "BULL_TREND": MarketRegime.UPTREND,
            "RANGE": MarketRegime.RANGE,
            "SIDEWAYS": MarketRegime.RANGE,
            "DOWNTREND": MarketRegime.DOWNTREND,
            "BEAR_CRASH": MarketRegime.DOWNTREND,
            "BEAR_TREND": MarketRegime.DOWNTREND,
            "EXTREME_FEAR": MarketRegime.EXTREME_FEAR,
            "EXTREME_GREED": MarketRegime.EXTREME_GREED,
            "UNKNOWN": MarketRegime.UNKNOWN,
        }
        return mapping.get(raw, MarketRegime.UNKNOWN)

    @staticmethod
    def _market_regime_from_pnl(pnl_pct: float) -> MarketRegime:
        if pnl_pct < -5:
            return MarketRegime.DOWNTREND
        if pnl_pct > 5:
            return MarketRegime.UPTREND
        return MarketRegime.RANGE

    def _build_okx_exchange(self):
        import ccxt

        params = {"enableRateLimit": True}
        proxy_url = get_settings().exchange.proxy_url or ""
        if proxy_url:
            params["proxies"] = {
                "http": proxy_url,
                "https": proxy_url,
            }
        return ccxt.okx(params)
