"""DCA engine."""
from __future__ import annotations

from datetime import datetime, timezone
from loguru import logger

from core.storage import Storage


class DCAEngine:
    """Simple DCA allocator with regime-based scaling."""

    def __init__(
        self,
        storage: Storage,
        total_balance: float = 10000.0,
        dca_ratio: float = 0.80,
        dca_interval_hours: int = 24,
        default_symbols: list[str] | None = None,
    ):
        self.storage = storage
        self.total_balance = total_balance
        self.dca_ratio = dca_ratio
        self.dca_interval_hours = dca_interval_hours
        self.default_symbols = default_symbols or ["BTC/USDT", "ETH/USDT"]
        self.symbol_weights = {"BTC/USDT": 0.6, "ETH/USDT": 0.4}

    def update_symbols(
        self,
        symbols: list[str],
        weights: dict[str, float] | None = None,
    ):
        """Update DCA symbols and normalize weights to sum to 1."""
        if not symbols:
            return

        self.default_symbols = symbols
        if weights:
            raw_weights = {
                symbol: max(0.0, float(weights.get(symbol, 0.0)))
                for symbol in symbols
            }
            total_weight = sum(raw_weights.values())
            if total_weight > 0:
                normalized_weights: dict[str, float] = {}
                running_total = 0.0
                for index, symbol in enumerate(symbols):
                    if index == len(symbols) - 1:
                        normalized_weights[symbol] = round(
                            max(0.0, 1.0 - running_total), 6
                        )
                    else:
                        normalized = round(
                            raw_weights[symbol] / total_weight, 6
                        )
                        normalized_weights[symbol] = normalized
                        running_total += normalized
                self.symbol_weights = normalized_weights
            else:
                equal_weight = round(1.0 / len(symbols), 6)
                self.symbol_weights = {symbol: equal_weight for symbol in symbols}
        else:
            equal_weights: dict[str, float] = {}
            running_total = 0.0
            for index, symbol in enumerate(symbols):
                if index == len(symbols) - 1:
                    equal_weights[symbol] = round(
                        max(0.0, 1.0 - running_total), 6
                    )
                else:
                    equal_weight = round(1.0 / len(symbols), 6)
                    equal_weights[symbol] = equal_weight
                    running_total += equal_weight
            self.symbol_weights = equal_weights

        logger.info(
            f"DCA symbols updated: {symbols} | weights: {self.symbol_weights}"
        )

    def get_dca_config(self, regime_state: str = "UNKNOWN") -> dict:
        """Return DCA multiplier for the current regime."""
        base_config = {
            "enabled": True,
            "multiplier": 1.0,
            "reason": "normal dca",
            "action": "buy",
        }

        regime_map = {
            "BULL_TREND": {
                "enabled": True,
                "multiplier": 1.0,
                "reason": "bull trend, keep steady dca",
                "action": "buy",
            },
            "BULL_CONSOL": {
                "enabled": True,
                "multiplier": 1.0,
                "reason": "bull consolidation, keep steady dca",
                "action": "buy",
            },
            "BEAR_TREND": {
                "enabled": True,
                "multiplier": 1.5,
                "reason": "bear trend, increase dca",
                "action": "buy",
            },
            "BEAR_RALLY": {
                "enabled": True,
                "multiplier": 1.0,
                "reason": "bear rally, normal dca",
                "action": "buy",
            },
            "EXTREME_FEAR": {
                "enabled": True,
                "multiplier": 2.5,
                "reason": "extreme fear, aggressive dca",
                "action": "buy_aggressive",
            },
            "EXTREME_GREED": {
                "enabled": True,
                "multiplier": 0.3,
                "reason": "extreme greed, reduce dca",
                "action": "buy_minimal",
            },
            "UNKNOWN": {
                "enabled": True,
                "multiplier": 1.0,
                "reason": "unknown regime, normal dca",
                "action": "buy",
            },
        }
        return regime_map.get(regime_state, base_config)

    def calculate_dca_amount(
        self,
        regime_state: str = "UNKNOWN",
        fear_greed: float | None = None,
    ) -> dict:
        """Calculate DCA allocations for the current cycle."""
        _ = fear_greed
        config = self.get_dca_config(regime_state)
        if not config["enabled"]:
            logger.info("DCA disabled for current regime")
            return {
                "total_usdt": 0,
                "allocations": {},
                "regime_state": regime_state,
                "multiplier": 0,
                "reason": "DCA paused",
            }

        monthly_investments = 30
        base_amount = self.total_balance * self.dca_ratio / monthly_investments
        total_usdt = base_amount * config["multiplier"]
        max_single = self.total_balance * 0.05
        total_usdt = min(total_usdt, max_single)

        allocations: dict[str, float] = {}
        allocated_total = 0.0
        symbols = list(self.symbol_weights.keys())
        for index, symbol in enumerate(symbols):
            weight = self.symbol_weights[symbol]
            if index == len(symbols) - 1:
                amount = round(total_usdt - allocated_total, 2)
            else:
                amount = round(total_usdt * weight, 2)
                allocated_total += amount
            allocations[symbol] = max(0.0, amount)

        logger.info(
            f"DCA: ${total_usdt:.2f} | multiplier={config['multiplier']}x | "
            f"regime={regime_state} | {config['reason']}"
        )

        return {
            "total_usdt": round(total_usdt, 2),
            "allocations": allocations,
            "regime_state": regime_state,
            "multiplier": config["multiplier"],
            "reason": config["reason"],
        }

    def get_dca_summary(self) -> dict:
        """Summarize DCA history from trade records."""
        with self.storage._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM trades WHERE rationale LIKE '%DCA%'"
            ).fetchone()
            total_dca = row["cnt"] if row else 0

            row = conn.execute(
                "SELECT COALESCE(SUM(entry_price * quantity), 0) as total "
                "FROM trades WHERE rationale LIKE '%DCA%'"
            ).fetchone()
            total_invested = row["total"] if row else 0

        return {
            "total_dca_trades": total_dca,
            "total_invested": round(total_invested, 2),
            "dca_ratio": self.dca_ratio,
            "strategy": "80% DCA + 20% active trading",
        }

    def should_execute_dca(self) -> bool:
        """Check whether the DCA cooldown interval has elapsed."""
        last_dca = self.storage.get_state("last_dca_time")
        if not last_dca:
            return True

        try:
            last_dt = datetime.fromisoformat(last_dca)
            elapsed_hours = (
                datetime.now(timezone.utc) - last_dt
            ).total_seconds() / 3600
            return elapsed_hours >= self.dca_interval_hours
        except Exception:
            return True

    def record_dca_execution(
        self,
        trade_id: str,
        symbol: str,
        amount: float,
        price: float,
    ):
        """Record the last DCA execution time."""
        _ = trade_id
        self.storage.set_state(
            "last_dca_time",
            datetime.now(timezone.utc).isoformat(),
        )
        logger.info(f"DCA executed: {symbol} ${amount:.2f} @ ${price:.2f}")
