"""Risk management rules for CryptoAI v3."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from statistics import mean

from config import RiskSettings, StrategySettings
from core.models import AccountState, Position, RiskCheckResult
from strategy.correlation_risk import CorrelationRiskManager


class RiskManager:
    """Apply portfolio, loss, and circuit-breaker constraints."""

    def __init__(self, risk: RiskSettings, strategy: StrategySettings):
        self.risk = risk
        self.strategy = strategy
        self.correlation_manager = CorrelationRiskManager(
            high_threshold=self.risk.correlation_high_threshold,
            medium_threshold=self.risk.correlation_medium_threshold,
            max_group_exposure=self.risk.correlation_hard_exposure_pct,
        )

    def build_account_state(
        self,
        equity: float,
        positions: list[dict],
        realized_pnl_today: float = 0.0,
        realized_pnl_week: float = 0.0,
        unrealized_pnl: float = 0.0,
        unrealized_pnl_today: float | None = None,
        unrealized_pnl_week: float | None = None,
        peak_equity: float | None = None,
        cooldown_until: datetime | None = None,
        circuit_breaker_active: bool = False,
    ) -> AccountState:
        exposure = sum(
            float(position.get("current_price") or position["entry_price"])
            * float(position["quantity"])
            for position in positions
        )
        peak = peak_equity or equity
        drawdown = 0.0 if peak <= 0 else max(0.0, (peak - equity) / peak)
        daily_pnl = realized_pnl_today + (
            unrealized_pnl if unrealized_pnl_today is None else unrealized_pnl_today
        )
        weekly_pnl = realized_pnl_week + (
            unrealized_pnl if unrealized_pnl_week is None else unrealized_pnl_week
        )
        return AccountState(
            equity=equity,
            realized_pnl=realized_pnl_today,
            unrealized_pnl=unrealized_pnl,
            daily_loss_pct=abs(min(daily_pnl / equity, 0.0)) if equity else 0.0,
            weekly_loss_pct=abs(min(weekly_pnl / equity, 0.0)) if equity else 0.0,
            drawdown_pct=drawdown,
            total_exposure_pct=exposure / equity if equity else 0.0,
            open_positions=len(positions),
            cooldown_until=cooldown_until,
            circuit_breaker_active=circuit_breaker_active,
        )

    def can_open_position(
        self,
        account: AccountState,
        positions: list[dict],
        symbol: str,
        atr: float,
        entry_price: float,
        liquidity_ratio: float,
        liquidity_floor_override: float | None = None,
        consecutive_wins: int = 0,
        consecutive_losses: int = 0,
        performance_snapshot=None,
        correlation_price_data: dict[str, list[float]] | None = None,
    ) -> RiskCheckResult:
        now = datetime.now(timezone.utc)
        if account.circuit_breaker_active:
            return RiskCheckResult(allowed=False, reason="circuit breaker active")
        if account.cooldown_until and now < account.cooldown_until:
            return RiskCheckResult(allowed=False, reason="cooldown active")
        if account.daily_loss_pct >= self.risk.daily_loss_limit_pct:
            return RiskCheckResult(allowed=False, reason="daily loss limit hit")
        if account.drawdown_pct >= self.risk.max_drawdown_pct:
            return RiskCheckResult(allowed=False, reason="max drawdown hit")
        liquidity_floor = self._effective_liquidity_floor(liquidity_floor_override)
        if liquidity_ratio < liquidity_floor:
            return RiskCheckResult(
                allowed=False,
                reason="insufficient liquidity",
                liquidity_floor_used=liquidity_floor,
                observed_liquidity_ratio=liquidity_ratio,
            )

        portfolio_heat = self._portfolio_heat_adjustment(
            account=account,
            performance_snapshot=performance_snapshot,
        )
        effective_max_positions = int(portfolio_heat["effective_max_positions"])
        if account.open_positions >= effective_max_positions:
            reason = "max positions reached"
            if (
                float(portfolio_heat["factor"]) < 0.999
                and effective_max_positions < self.risk.max_positions
            ):
                reason = (
                    "portfolio heat max positions reached: "
                    f"limit={effective_max_positions}/{self.risk.max_positions}; "
                    f"{portfolio_heat['reason']}"
                )
            return RiskCheckResult(
                allowed=False,
                reason=reason,
                liquidity_floor_used=liquidity_floor,
                observed_liquidity_ratio=liquidity_ratio,
                portfolio_heat_factor=float(portfolio_heat["factor"]),
                effective_max_total_exposure_pct=float(
                    portfolio_heat["effective_max_total_exposure_pct"]
                ),
                effective_max_positions=effective_max_positions,
            )

        current_symbol_exposure = sum(
            float(p.get("current_price") or p["entry_price"]) * float(p["quantity"])
            for p in positions
            if p["symbol"] == symbol
        )
        if account.equity > 0 and (
            current_symbol_exposure / account.equity
        ) >= self.risk.max_symbol_exposure_pct:
            return RiskCheckResult(allowed=False, reason="symbol exposure exceeded")

        risk_amount = account.equity * self.risk.single_trade_risk_pct
        stop_distance = max(atr * 2.0, entry_price * self.strategy.fixed_stop_loss_pct)
        quantity = risk_amount / stop_distance if stop_distance > 0 else 0.0
        position_value = quantity * entry_price

        symbol_cap = account.equity * self.risk.max_symbol_exposure_pct
        total_cap = max(
            0.0,
            account.equity * float(portfolio_heat["effective_max_total_exposure_pct"])
            - (account.total_exposure_pct * account.equity),
        )
        dynamic_factor = self._dynamic_position_factor(
            account,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            performance_snapshot=performance_snapshot,
        )
        base_allowed_position_value = min(position_value, symbol_cap, total_cap) * dynamic_factor
        if base_allowed_position_value <= 0:
            return RiskCheckResult(allowed=False, reason="no exposure budget")
        correlation_adjustment = self._correlation_position_adjustment(
            account=account,
            positions=positions,
            symbol=symbol,
            base_position_value=base_allowed_position_value,
            correlation_price_data=correlation_price_data,
        )
        if correlation_adjustment["blocked"]:
            return RiskCheckResult(
                allowed=False,
                reason=correlation_adjustment["reason"],
                liquidity_floor_used=liquidity_floor,
                observed_liquidity_ratio=liquidity_ratio,
                allowed_risk_amount=risk_amount,
                allowed_position_value=0.0,
                dynamic_position_factor=dynamic_factor,
                portfolio_heat_factor=float(portfolio_heat["factor"]),
                effective_max_total_exposure_pct=float(
                    portfolio_heat["effective_max_total_exposure_pct"]
                ),
                effective_max_positions=effective_max_positions,
                correlation_position_factor=0.0,
                correlation_effective_exposure_pct=correlation_adjustment[
                    "projected_effective_exposure_pct"
                ],
                correlation_crowded_symbols=correlation_adjustment["crowded_symbols"],
                stop_loss_pct=self.strategy.fixed_stop_loss_pct,
                take_profit_levels=self.strategy.take_profit_levels,
                trailing_stop_drawdown_pct=self.strategy.trailing_stop_drawdown_pct,
            )
        allowed_position_value = (
            base_allowed_position_value * correlation_adjustment["factor"]
        )
        if allowed_position_value <= 0:
            return RiskCheckResult(
                allowed=False,
                reason=(
                    correlation_adjustment["reason"]
                    or str(portfolio_heat["reason"])
                    or "no exposure budget"
                ),
                liquidity_floor_used=liquidity_floor,
                observed_liquidity_ratio=liquidity_ratio,
                allowed_risk_amount=risk_amount,
                allowed_position_value=0.0,
                dynamic_position_factor=dynamic_factor,
                portfolio_heat_factor=float(portfolio_heat["factor"]),
                effective_max_total_exposure_pct=float(
                    portfolio_heat["effective_max_total_exposure_pct"]
                ),
                effective_max_positions=effective_max_positions,
                correlation_position_factor=correlation_adjustment["factor"],
                correlation_effective_exposure_pct=correlation_adjustment[
                    "projected_effective_exposure_pct"
                ],
                correlation_crowded_symbols=correlation_adjustment["crowded_symbols"],
                stop_loss_pct=self.strategy.fixed_stop_loss_pct,
                take_profit_levels=self.strategy.take_profit_levels,
                trailing_stop_drawdown_pct=self.strategy.trailing_stop_drawdown_pct,
            )

        return RiskCheckResult(
            allowed=True,
            reason=str(correlation_adjustment["reason"] or portfolio_heat["reason"]),
            liquidity_floor_used=liquidity_floor,
            observed_liquidity_ratio=liquidity_ratio,
            allowed_risk_amount=risk_amount,
            allowed_position_value=allowed_position_value,
            dynamic_position_factor=dynamic_factor,
            portfolio_heat_factor=float(portfolio_heat["factor"]),
            effective_max_total_exposure_pct=float(
                portfolio_heat["effective_max_total_exposure_pct"]
            ),
            effective_max_positions=effective_max_positions,
            correlation_position_factor=correlation_adjustment["factor"],
            correlation_effective_exposure_pct=correlation_adjustment[
                "projected_effective_exposure_pct"
            ],
            correlation_crowded_symbols=correlation_adjustment["crowded_symbols"],
            stop_loss_pct=self.strategy.fixed_stop_loss_pct,
            take_profit_levels=self.strategy.take_profit_levels,
            trailing_stop_drawdown_pct=self.strategy.trailing_stop_drawdown_pct,
        )

    def _effective_liquidity_floor(
        self,
        liquidity_floor_override: float | None = None,
    ) -> float:
        base_floor = float(self.strategy.min_liquidity_ratio)
        try:
            override = float(liquidity_floor_override) if liquidity_floor_override is not None else base_floor
        except (TypeError, ValueError):
            override = base_floor
        override = max(0.0, override)
        return min(base_floor, override)

    def update_cooldown_after_losses(
        self,
        consecutive_losses: int,
    ) -> datetime | None:
        if consecutive_losses >= self.risk.consecutive_loss_limit:
            return datetime.now(timezone.utc) + timedelta(
                hours=self.risk.consecutive_loss_cooldown_hours
            )
        return None

    def apply_account_cooldown(
        self,
        account: AccountState,
        current_cooldown_until: datetime | None = None,
        now: datetime | None = None,
    ) -> datetime | None:
        now = now or datetime.now(timezone.utc)
        cooldown_until = current_cooldown_until

        if account.weekly_loss_pct >= self.risk.weekly_loss_limit_pct:
            weekly_until = now + timedelta(days=self.risk.weekly_loss_cooldown_days)
            cooldown_until = self._max_datetime(cooldown_until, weekly_until)

        if account.drawdown_pct >= self.risk.drawdown_pause_pct:
            drawdown_until = now + timedelta(days=self.risk.drawdown_cooldown_days)
            cooldown_until = self._max_datetime(cooldown_until, drawdown_until)

        return cooldown_until

    def check_circuit_breaker(self, account: AccountState) -> str:
        if account.daily_loss_pct >= self.risk.daily_loss_limit_pct:
            return "daily_loss_limit"
        if account.drawdown_pct >= self.risk.max_drawdown_pct:
            return "max_drawdown_limit"
        return ""

    def _dynamic_position_factor(
        self,
        account: AccountState,
        consecutive_wins: int,
        consecutive_losses: int,
        performance_snapshot=None,
    ) -> float:
        factor = 1.0
        if consecutive_wins >= self.risk.consecutive_win_threshold:
            factor *= self.risk.consecutive_win_position_boost
        if consecutive_losses >= self.risk.consecutive_loss_cut_threshold:
            factor = min(factor, self.risk.consecutive_loss_position_cut)
        if account.daily_loss_pct >= self.risk.daily_loss_position_cut_threshold_pct:
            factor = min(factor, self.risk.daily_loss_position_cut_factor)
        if account.drawdown_pct >= self.risk.drawdown_position_cut_threshold_pct:
            factor = min(factor, self.risk.drawdown_position_cut_factor)
        performance_factor = self._performance_position_factor(performance_snapshot)
        factor = min(factor, performance_factor)
        return max(0.0, factor)

    def _performance_position_factor(self, performance_snapshot) -> float:
        if performance_snapshot is None:
            return 1.0
        recent_closed_trades = int(
            getattr(performance_snapshot, "recent_closed_trades", 0) or 0
        )
        if recent_closed_trades < 3:
            return 1.0
        paper_grace = self._paper_portfolio_heat_grace(performance_snapshot)
        factor = 1.0
        recent_expectancy_pct = float(
            getattr(performance_snapshot, "recent_expectancy_pct", 0.0) or 0.0
        )
        recent_profit_factor = float(
            getattr(performance_snapshot, "recent_profit_factor", 0.0) or 0.0
        )
        recent_max_drawdown_pct = float(
            getattr(performance_snapshot, "recent_max_drawdown_pct", 0.0) or 0.0
        )
        recent_sortino_like = float(
            getattr(performance_snapshot, "recent_sortino_like", 0.0) or 0.0
        )
        equity_return_pct = float(
            getattr(performance_snapshot, "equity_return_pct", 0.0) or 0.0
        )

        if recent_expectancy_pct < 0:
            factor = min(
                factor,
                0.90 if paper_grace else 0.75,
            )
        if recent_profit_factor < 1.0:
            factor = min(
                factor,
                0.75 if paper_grace else 0.70,
            )
        if recent_sortino_like < 0:
            factor = min(
                factor,
                0.75 if paper_grace else 0.70,
            )
        if recent_max_drawdown_pct > self.risk.drawdown_position_cut_threshold_pct * 100:
            factor = min(factor, self.risk.drawdown_position_cut_factor)
        if equity_return_pct < 0 and recent_profit_factor < 1.0:
            factor = min(
                factor,
                0.75 if paper_grace else 0.50,
            )
        return max(0.0, factor)

    def _portfolio_heat_adjustment(
        self,
        *,
        account: AccountState,
        performance_snapshot,
    ) -> dict:
        default_payload = {
            "factor": 1.0,
            "effective_max_total_exposure_pct": self.risk.max_total_exposure_pct,
            "effective_max_positions": self.risk.max_positions,
            "reason": "",
        }
        if performance_snapshot is None:
            return default_payload

        recent_closed_trades = int(
            getattr(performance_snapshot, "recent_closed_trades", 0) or 0
        )
        if recent_closed_trades < self.risk.portfolio_heat_min_recent_trades:
            return default_payload

        paper_grace = self._paper_portfolio_heat_grace(performance_snapshot)
        floor = min(
            1.0,
            max(0.10, float(self.risk.portfolio_heat_exposure_floor_pct or 1.0)),
        )
        if paper_grace:
            floor = max(floor, 0.70)
        factor = 1.0
        reasons: list[str] = []

        recent_expectancy_pct = float(
            getattr(performance_snapshot, "recent_expectancy_pct", 0.0) or 0.0
        )
        recent_profit_factor = float(
            getattr(performance_snapshot, "recent_profit_factor", 0.0) or 0.0
        )
        recent_sortino_like = float(
            getattr(performance_snapshot, "recent_sortino_like", 0.0) or 0.0
        )
        equity_return_pct = float(
            getattr(performance_snapshot, "equity_return_pct", 0.0) or 0.0
        )
        recent_max_drawdown_pct = float(
            getattr(performance_snapshot, "recent_max_drawdown_pct", 0.0) or 0.0
        )
        recent_drawdown_velocity_pct = float(
            getattr(
                performance_snapshot,
                "recent_drawdown_velocity_pct",
                recent_max_drawdown_pct / max(recent_closed_trades, 1),
            )
            or 0.0
        )
        recent_return_volatility_pct = float(
            getattr(performance_snapshot, "recent_return_volatility_pct", 0.0) or 0.0
        )
        recent_loss_cluster_ratio_pct = float(
            getattr(performance_snapshot, "recent_loss_cluster_ratio_pct", 0.0) or 0.0
        )

        if recent_expectancy_pct < 0:
            if paper_grace:
                expectancy_factor = 0.90
                if recent_expectancy_pct <= -0.70:
                    expectancy_factor = floor
                elif recent_expectancy_pct <= -0.30:
                    expectancy_factor = 0.80
            else:
                expectancy_factor = 0.85
                if recent_expectancy_pct <= -0.70:
                    expectancy_factor = floor
                elif recent_expectancy_pct <= -0.30:
                    expectancy_factor = 0.70
            factor = min(factor, expectancy_factor)
            reasons.append(f"expectancy={recent_expectancy_pct:.2f}%")

        if recent_profit_factor < 1.0:
            if paper_grace:
                profit_factor_cap = max(
                    floor,
                    min(0.90, 0.72 + recent_profit_factor * 0.22),
                )
            else:
                profit_factor_cap = max(
                    floor,
                    min(0.85, 0.55 + recent_profit_factor * 0.30),
                )
            factor = min(factor, profit_factor_cap)
            reasons.append(f"pf={recent_profit_factor:.2f}")

        if recent_sortino_like < 0:
            if paper_grace:
                sortino_factor = 0.82 if recent_sortino_like >= -0.30 else 0.75
            else:
                sortino_factor = 0.75 if recent_sortino_like >= -0.30 else 0.65
            factor = min(factor, max(floor, sortino_factor))
            reasons.append(f"sortino={recent_sortino_like:.2f}")

        if equity_return_pct < 0:
            if paper_grace:
                equity_factor = 0.90 if equity_return_pct >= -1.0 else 0.80
            else:
                equity_factor = 0.85 if equity_return_pct >= -1.0 else 0.75
            factor = min(factor, max(floor, equity_factor))
            reasons.append(f"equity={equity_return_pct:.2f}%")

        drawdown_velocity_factor = self._high_value_heat_factor(
            value=recent_drawdown_velocity_pct,
            soft=self.risk.portfolio_heat_soft_drawdown_velocity_pct,
            hard=self.risk.portfolio_heat_hard_drawdown_velocity_pct,
            floor=floor,
        )
        if drawdown_velocity_factor < 0.999:
            factor = min(factor, drawdown_velocity_factor)
            reasons.append(f"ddv={recent_drawdown_velocity_pct:.2f}%/trade")

        volatility_factor = self._high_value_heat_factor(
            value=recent_return_volatility_pct,
            soft=self.risk.portfolio_heat_soft_return_volatility_pct,
            hard=self.risk.portfolio_heat_hard_return_volatility_pct,
            floor=floor,
        )
        if volatility_factor < 0.999:
            factor = min(factor, volatility_factor)
            reasons.append(f"vol={recent_return_volatility_pct:.2f}%")

        loss_cluster_factor = self._high_value_heat_factor(
            value=recent_loss_cluster_ratio_pct,
            soft=self.risk.portfolio_heat_soft_loss_cluster_pct,
            hard=self.risk.portfolio_heat_hard_loss_cluster_pct,
            floor=floor,
        )
        if loss_cluster_factor < 0.999:
            factor = min(factor, loss_cluster_factor)
            reasons.append(f"loss_cluster={recent_loss_cluster_ratio_pct:.1f}%")

        factor = min(1.0, max(floor, factor))
        effective_max_total_exposure_pct = (
            self.risk.max_total_exposure_pct * factor
        )
        effective_max_positions = max(
            1,
            min(
                self.risk.max_positions,
                int(round(self.risk.max_positions * factor)),
            ),
        )
        reason = ""
        if factor < 0.999:
            prefix = "paper portfolio heat" if paper_grace else "portfolio heat"
            reason = f"{prefix} {factor:.2f}: {', '.join(reasons[:4])}"
        return {
            "factor": factor,
            "effective_max_total_exposure_pct": effective_max_total_exposure_pct,
            "effective_max_positions": effective_max_positions,
            "reason": reason,
        }

    def _paper_portfolio_heat_grace(self, performance_snapshot) -> bool:
        if performance_snapshot is None:
            return False
        if str(getattr(performance_snapshot, "runtime_mode", "") or "").lower() != "paper":
            return False
        paper_canary_open_count = int(
            getattr(performance_snapshot, "paper_canary_open_count", 0) or 0
        )
        if paper_canary_open_count <= 0:
            return False
        recent_closed_trades = int(
            getattr(performance_snapshot, "recent_closed_trades", 0) or 0
        )
        grace_trade_limit = max(
            int(self.risk.portfolio_heat_min_recent_trades) * 2,
            8,
        )
        return recent_closed_trades < grace_trade_limit

    @staticmethod
    def _high_value_heat_factor(
        *,
        value: float,
        soft: float,
        hard: float,
        floor: float,
    ) -> float:
        if value <= soft:
            return 1.0
        if hard <= soft:
            return floor
        if value >= hard:
            return floor
        normalized = (value - soft) / (hard - soft)
        return max(floor, 1.0 - normalized * (1.0 - floor))

    def _correlation_position_adjustment(
        self,
        *,
        account: AccountState,
        positions: list[dict],
        symbol: str,
        base_position_value: float,
        correlation_price_data: dict[str, list[float]] | None,
    ) -> dict:
        default_payload = {
            "factor": 1.0,
            "blocked": False,
            "reason": "",
            "projected_effective_exposure_pct": 0.0,
            "crowded_symbols": [],
        }
        if (
            account.equity <= 0
            or base_position_value <= 0
            or not positions
            or not correlation_price_data
        ):
            return default_payload

        candidate_prices = self._correlation_series(symbol, correlation_price_data)
        if len(candidate_prices) < 10:
            return default_payload

        correlated_positions = []
        for position in positions:
            held_symbol = str(position.get("symbol") or "")
            held_direction = str(position.get("direction") or "LONG").upper()
            if not held_symbol or held_symbol == symbol or held_direction == "SHORT":
                continue

            held_prices = self._correlation_series(held_symbol, correlation_price_data)
            if len(held_prices) < 10:
                continue

            avg_abs_corr = self._average_abs_correlation(candidate_prices, held_prices)
            if avg_abs_corr < self.risk.correlation_medium_threshold:
                continue

            exposure_value = float(
                (position.get("current_price") or position.get("entry_price") or 0.0)
            ) * float(position.get("quantity") or 0.0)
            exposure_pct = exposure_value / account.equity if account.equity else 0.0
            effective_exposure_pct = exposure_pct * avg_abs_corr
            correlated_positions.append(
                {
                    "symbol": held_symbol,
                    "avg_abs_corr": avg_abs_corr,
                    "effective_exposure_pct": effective_exposure_pct,
                }
            )

        if not correlated_positions:
            return default_payload

        correlated_positions.sort(
            key=lambda item: item["effective_exposure_pct"],
            reverse=True,
        )
        crowded_symbols = [item["symbol"] for item in correlated_positions[:3]]
        strongest_corr = max(item["avg_abs_corr"] for item in correlated_positions)
        current_effective_exposure_pct = sum(
            item["effective_exposure_pct"] for item in correlated_positions
        )
        candidate_exposure_pct = base_position_value / account.equity
        candidate_effective_exposure_pct = candidate_exposure_pct * strongest_corr
        projected_effective_exposure_pct = (
            current_effective_exposure_pct + candidate_effective_exposure_pct
        )

        if (
            strongest_corr >= self.risk.correlation_high_threshold
            and projected_effective_exposure_pct >= self.risk.correlation_hard_exposure_pct
        ):
            return {
                "factor": 0.0,
                "blocked": True,
                "reason": (
                    "correlation crowding block: "
                    f"eff={projected_effective_exposure_pct:.1%}, "
                    f"peers={','.join(crowded_symbols)}"
                ),
                "projected_effective_exposure_pct": projected_effective_exposure_pct,
                "crowded_symbols": crowded_symbols,
            }

        if projected_effective_exposure_pct <= self.risk.correlation_soft_exposure_pct:
            return {
                **default_payload,
                "projected_effective_exposure_pct": projected_effective_exposure_pct,
                "crowded_symbols": crowded_symbols,
            }

        exposure_band = max(
            1e-9,
            self.risk.correlation_hard_exposure_pct
            - self.risk.correlation_soft_exposure_pct,
        )
        normalized_overflow = min(
            1.0,
            (
                projected_effective_exposure_pct
                - self.risk.correlation_soft_exposure_pct
            )
            / exposure_band,
        )
        factor = max(
            self.risk.correlation_position_floor,
            1.0 - normalized_overflow,
        )
        return {
            "factor": factor,
            "blocked": False,
            "reason": (
                f"correlation haircut {factor:.2f}: "
                f"eff={projected_effective_exposure_pct:.1%}, "
                f"peers={','.join(crowded_symbols)}"
            ),
            "projected_effective_exposure_pct": projected_effective_exposure_pct,
            "crowded_symbols": crowded_symbols,
        }

    @staticmethod
    def _correlation_series(
        symbol: str,
        correlation_price_data: dict[str, list[float]],
    ) -> list[float]:
        for candidate in (symbol, symbol.replace(":USDT", ""), f"{symbol}:USDT"):
            series = correlation_price_data.get(candidate)
            if series:
                return list(series)
        return []

    def _average_abs_correlation(
        self,
        series_a: list[float],
        series_b: list[float],
    ) -> float:
        corrs = []
        for window in (30, 60, 90):
            corr = self.correlation_manager.calculate_correlation(
                series_a[-window:] if len(series_a) >= window else series_a,
                series_b[-window:] if len(series_b) >= window else series_b,
            )
            corrs.append(abs(float(corr or 0.0)))
        usable_corrs = [corr for corr in corrs if corr > 0]
        if not usable_corrs:
            return 0.0
        return float(mean(usable_corrs))

    @staticmethod
    def _max_datetime(left: datetime | None, right: datetime | None) -> datetime | None:
        if left is None:
            return right
        if right is None:
            return left
        return max(left, right)
