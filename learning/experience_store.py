"""Experience Store — 经验库存储与检索"""
from __future__ import annotations

import json
import re

from core.storage import Storage


class ExperienceStore:
    """交易经验库 — 存储和检索历史经验"""

    SHADOW_REFLECTION_WEIGHT = 0.35
    PAPER_CANARY_REFLECTION_WEIGHT = 0.60

    def __init__(self, storage: Storage):
        self.storage = storage

    @classmethod
    def reflection_weight(
        cls,
        trade_id: str | None,
        source: str | None = None,
        experience_weight: float | None = None,
    ) -> float:
        if experience_weight is not None:
            try:
                weight = float(experience_weight)
            except (TypeError, ValueError):
                weight = 0.0
            if weight > 0:
                return weight
        source_text = str(source or "").strip().lower()
        if source_text == "shadow_observation":
            return cls.SHADOW_REFLECTION_WEIGHT
        if source_text == "paper_canary":
            return cls.PAPER_CANARY_REFLECTION_WEIGHT
        trade_id_text = str(trade_id or "")
        if trade_id_text.startswith("shadow:"):
            return cls.SHADOW_REFLECTION_WEIGHT
        return 1.0

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"\b\w+\b", (text or "").lower())
            if token
        }

    @staticmethod
    def _similarity_score(
        query_tokens: set[str],
        document_tokens: set[str],
        symbol_match: bool,
        regime_match: bool,
    ) -> float:
        union = query_tokens | document_tokens
        lexical_score = (
            len(query_tokens & document_tokens) / len(union)
            if union
            else 0.0
        )
        score = lexical_score
        if symbol_match:
            score += 0.25
        if regime_match:
            score += 0.15
        return min(1.0, score)

    @staticmethod
    def normalize_validation_reason(reason: str | None) -> str:
        text = (reason or "ok").strip().lower()
        if not text or text == "ok":
            return "ok"
        primary = text.split(",", 1)[0].strip()
        normalized = re.sub(r"[^a-z0-9_]+", "_", primary).strip("_")
        return normalized or "unknown"

    @staticmethod
    def liquidity_bucket(
        liquidity_ratio: float | None,
        min_liquidity_ratio: float,
    ) -> str:
        ratio = float(liquidity_ratio or 0.0)
        if ratio < min_liquidity_ratio:
            return "weak"
        if ratio < min_liquidity_ratio + 0.25:
            return "mid"
        return "strong"

    @staticmethod
    def news_bucket(
        news_sources: list[str] | None,
        coverage_score: float | None,
        service_health_score: float | None,
    ) -> str:
        source_count = len(news_sources or [])
        coverage = float(coverage_score or 0.0)
        health = float(service_health_score or 0.0)
        if health < 0.35:
            return "down"
        if source_count == 0 and coverage < 0.15:
            return "thin"
        if source_count >= 2 and coverage >= 0.6:
            return "strong"
        return "mid"

    @classmethod
    def build_setup_profile(
        cls,
        symbol: str,
        market_regime: str,
        validation_reason: str | None,
        liquidity_ratio: float | None,
        min_liquidity_ratio: float,
        news_sources: list[str] | None,
        news_coverage_score: float | None,
        news_service_health_score: float | None,
    ) -> dict[str, str]:
        return {
            "symbol": str(symbol),
            "regime": str(market_regime or "UNKNOWN").upper(),
            "validation": cls.normalize_validation_reason(validation_reason),
            "liquidity_bucket": cls.liquidity_bucket(
                liquidity_ratio,
                min_liquidity_ratio,
            ),
            "news_bucket": cls.news_bucket(
                news_sources,
                news_coverage_score,
                news_service_health_score,
            ),
        }

    @staticmethod
    def encode_setup_profile(profile: dict[str, str]) -> str:
        keys = ("symbol", "regime", "validation", "liquidity_bucket", "news_bucket")
        parts = [
            f"{key}:{str(profile.get(key, '')).strip()}"
            for key in keys
            if str(profile.get(key, "")).strip()
        ]
        return "setup_profile=" + "|".join(parts) if parts else ""

    @staticmethod
    def parse_setup_profile(rationale: str | None) -> dict[str, str]:
        text = rationale or ""
        match = re.search(r"setup_profile=([^\n]+)", text)
        if not match:
            return {}
        encoded = match.group(1).strip()
        if ";" in encoded:
            encoded = encoded.split(";", 1)[0].strip()
        profile: dict[str, str] = {}
        for item in encoded.split("|"):
            if ":" not in item:
                continue
            key, value = item.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                profile[key] = value
        return profile

    def get_similar_lessons(
        self, symbol: str, direction: str, limit: int = 5
    ) -> list[str]:
        """检索类似交易的经验教训"""
        with self.storage._conn() as conn:
            rows = conn.execute(
                """SELECT lesson FROM reflections
                   WHERE symbol = ? AND direction = ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (symbol, direction, limit),
            ).fetchall()
            return [r["lesson"] for r in rows if r["lesson"]]

    def get_all_lessons(self, limit: int = 20) -> list[dict]:
        """获取所有经验"""
        with self.storage._conn() as conn:
            rows = conn.execute(
                """SELECT trade_id, symbol, direction, confidence,
                          COALESCE(realized_return_pct, outcome_24h) AS realized_return_pct,
                          outcome_24h, lesson, market_regime, created_at
                   FROM reflections
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_regime_distribution(self) -> dict:
        """统计不同市场状态下的交易表现"""
        with self.storage._conn() as conn:
            rows = conn.execute("""
                SELECT market_regime,
                       COUNT(*) as total,
                       AVG(COALESCE(realized_return_pct, outcome_24h)) as avg_pnl,
                       SUM(CASE WHEN COALESCE(realized_return_pct, outcome_24h) > 0 THEN 1 ELSE 0 END) as wins
                FROM reflections
                GROUP BY market_regime
            """).fetchall()
            return [dict(r) for r in rows]

    def find_similar_setups(
        self,
        symbol: str,
        direction: str,
        market_regime: str,
        context: str,
        limit: int = 5,
    ) -> list[dict]:
        query_tokens = self._tokenize(
            f"{symbol} {direction} {market_regime} {context}"
        )
        with self.storage._conn() as conn:
            rows = conn.execute(
                """SELECT trade_id, symbol, direction, confidence, rationale,
                          source, experience_weight,
                          COALESCE(realized_return_pct, outcome_24h) AS realized_return_pct,
                          outcome_24h, lesson, market_regime, correct_signals,
                          wrong_signals, created_at
                   FROM reflections
                   WHERE direction = ?
                   ORDER BY created_at DESC
                   LIMIT 200""",
                (direction,),
            ).fetchall()

        matches: list[dict] = []
        for row in rows:
            correct_signals = json.loads(row["correct_signals"] or "[]")
            wrong_signals = json.loads(row["wrong_signals"] or "[]")
            document = " ".join(
                [
                    str(row["symbol"] or ""),
                    str(row["direction"] or ""),
                    str(row["market_regime"] or ""),
                    str(row["rationale"] or ""),
                    str(row["lesson"] or ""),
                    " ".join(str(item) for item in correct_signals),
                    " ".join(str(item) for item in wrong_signals),
                ]
            )
            score = self._similarity_score(
                query_tokens=query_tokens,
                document_tokens=self._tokenize(document),
                symbol_match=str(row["symbol"]) == symbol,
                regime_match=str(row["market_regime"] or "") == market_regime,
            )
            weight = self.reflection_weight(
                row["trade_id"],
                row["source"],
                row["experience_weight"],
            )
            score *= weight
            if score <= 0.0:
                continue
            matches.append(
                {
                    "trade_id": row["trade_id"],
                    "symbol": row["symbol"],
                    "market_regime": row["market_regime"],
                    "outcome_24h": row["realized_return_pct"],
                    "lesson": row["lesson"],
                    "experience_weight": round(weight, 4),
                    "similarity_score": round(score, 4),
                    "created_at": row["created_at"],
                }
            )

        matches.sort(
            key=lambda item: (
                -float(item["similarity_score"]),
                item["created_at"],
            ),
        )
        return matches[:limit]

    def aggregate_setup_performance(
        self,
        direction: str,
        setup_profile: dict[str, str],
        limit: int = 200,
    ) -> dict[str, float | int | str]:
        with self.storage._conn() as conn:
            rows = conn.execute(
                """SELECT trade_id, symbol, rationale, source, experience_weight,
                          COALESCE(realized_return_pct, outcome_24h) AS realized_return_pct,
                          outcome_24h, created_at
                   FROM reflections
                   WHERE direction = ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (direction, limit),
            ).fetchall()

        matched_outcomes: list[float] = []
        matched_weights: list[float] = []
        matched_symbols: set[str] = set()
        for row in rows:
            parsed = self.parse_setup_profile(row["rationale"])
            if not parsed:
                continue
            if parsed.get("symbol") != setup_profile.get("symbol"):
                continue
            if parsed.get("regime") != setup_profile.get("regime"):
                continue
            if parsed.get("validation") != setup_profile.get("validation"):
                continue
            if parsed.get("liquidity_bucket") != setup_profile.get("liquidity_bucket"):
                continue
            if parsed.get("news_bucket") != setup_profile.get("news_bucket"):
                continue
            outcome = row["realized_return_pct"]
            if outcome is None:
                continue
            matched_outcomes.append(float(outcome))
            matched_weights.append(
                self.reflection_weight(
                    row["trade_id"],
                    row["source"],
                    row["experience_weight"],
                )
            )
            matched_symbols.add(str(row["symbol"]))

        count = len(matched_outcomes)
        weighted_count = sum(matched_weights) if matched_weights else 0.0
        avg_outcome = (
            sum(outcome * weight for outcome, weight in zip(matched_outcomes, matched_weights))
            / weighted_count
            if weighted_count
            else 0.0
        )
        win_rate = (
            sum(weight for outcome, weight in zip(matched_outcomes, matched_weights) if outcome > 0)
            / weighted_count
            if weighted_count
            else 0.0
        )
        negative_ratio = (
            sum(weight for outcome, weight in zip(matched_outcomes, matched_weights) if outcome < 0)
            / weighted_count
            if weighted_count
            else 0.0
        )
        return {
            "count": count,
            "weighted_count": round(weighted_count, 4),
            "avg_outcome_24h": round(avg_outcome, 4),
            "win_rate": round(win_rate, 4),
            "negative_ratio": round(negative_ratio, 4),
            "matched_symbols": len(matched_symbols),
        }

    def generate_weekly_report(self) -> str:
        """生成周报"""
        lessons = self.get_all_lessons(limit=50)
        if not lessons:
            return "本周暂无交易经验记录。"

        total = len(lessons)
        wins = sum(1 for l in lessons if (l.get("outcome_24h") or 0) > 0)
        avg_pnl = sum(l.get("outcome_24h") or 0 for l in lessons) / total

        report = f"""## 📊 CryptoAI 周报

### 交易统计
- 总交易次数: {total}
- 盈利次数: {wins} ({wins/total:.0%})
- 平均收益: {avg_pnl:+.2f}%

### 经验教训精选
"""
        for i, l in enumerate(lessons[:10], 1):
            regime = l.get("market_regime", "N/A")
            pnl = l.get("outcome_24h")
            pnl_str = f"{pnl:+.2f}%" if pnl is not None else "N/A"
            report += f"{i}. [{regime}] {l['lesson']} (PnL: {pnl_str})\n"

        return report
