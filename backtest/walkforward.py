"""Walk-forward evaluation for CryptoAI v3."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

from config import Settings
from core.i18n import get_default_language, normalize_language, text_for
from core.storage import Storage
from strategy.model_trainer import ModelTrainer, load_xgboost


@dataclass
class WalkForwardSplit:
    train_rows: int
    test_rows: int
    win_rate: float
    avg_trade_return_pct: float
    total_return_pct: float


class WalkForwardBacktester:
    """Run rolling train/test evaluation on 4h samples."""

    def __init__(self, storage: Storage, settings: Settings):
        self.storage = storage
        self.settings = settings
        self.trainer = ModelTrainer(storage, settings)

    def run(self, symbol: str) -> dict:
        dataset = self.trainer.build_dataset(symbol)
        rows = dataset["rows"]
        labels = dataset["labels"]
        feature_names = dataset["feature_names"]
        current_closes = dataset["current_closes"]
        next_closes = dataset["next_closes"]
        trade_return_pcts = dataset.get("trade_return_pcts", [])
        total_rows = len(labels)

        if total_rows < self.settings.training.minimum_training_rows:
            return {
                "symbol": symbol,
                "splits": [],
                "summary": {
                    "total_splits": 0,
                    "avg_win_rate": 0.0,
                    "avg_trade_return_pct": 0.0,
                    "total_return_pct": 0.0,
                    "profit_factor": 0.0,
                    "sharpe_like": 0.0,
                },
            }

        train_min = self.settings.training.minimum_training_rows
        test_window = max(1, self.settings.training.walkforward_window_days * 6)
        threshold = self.settings.model.xgboost_probability_threshold
        splits: list[WalkForwardSplit] = []
        all_trade_returns: list[float] = []

        xgb = load_xgboost()

        for train_end in range(train_min, total_rows - test_window + 1, test_window):
            x_train = rows[:train_end]
            y_train = labels[:train_end]
            x_test = rows[train_end : train_end + test_window]
            y_test = labels[train_end : train_end + test_window]
            test_current = current_closes[train_end : train_end + test_window]
            test_next = next_closes[train_end : train_end + test_window]
            test_trade_returns = trade_return_pcts[train_end : train_end + test_window]

            probabilities = self._predict_probabilities(
                x_train,
                y_train,
                x_test,
                feature_names,
                xgb,
            )
            trade_returns = []
            wins = 0
            outcome_stream = (
                test_trade_returns
                if test_trade_returns
                else [
                    ((next_close / current_close) - 1.0) * 100.0 - 0.15
                    for current_close, next_close in zip(test_current, test_next)
                ]
            )
            for probability, net_return in zip(
                probabilities,
                outcome_stream,
            ):
                if probability < threshold:
                    continue
                trade_return = float(net_return or 0.0)
                trade_returns.append(trade_return)
                if trade_return > 0:
                    wins += 1

            win_rate = (wins / len(trade_returns) * 100) if trade_returns else 0.0
            avg_trade_return_pct = (
                sum(trade_returns) / len(trade_returns)
                if trade_returns else 0.0
            )
            total_return_pct = sum(trade_returns)
            splits.append(
                WalkForwardSplit(
                    train_rows=len(x_train),
                    test_rows=len(x_test),
                    win_rate=win_rate,
                    avg_trade_return_pct=avg_trade_return_pct,
                    total_return_pct=total_return_pct,
                )
            )
            all_trade_returns.extend(trade_returns)

        summary = self._summary(all_trade_returns, splits)
        return {
            "symbol": symbol,
            "splits": [split.__dict__ for split in splits],
            "summary": summary,
        }

    @staticmethod
    def render_report(result: dict, lang: str | None = None) -> str:
        lang = normalize_language(lang or get_default_language())
        summary = result["summary"]
        lines = [
            text_for(lang, f"# Walk-Forward 报告: {result['symbol']}", f"# Walk-Forward Report: {result['symbol']}"),
            text_for(lang, f"- 总切分数: {summary['total_splits']}", f"- Total Splits: {summary['total_splits']}"),
            text_for(lang, f"- 平均胜率: {summary['avg_win_rate']:.2f}%", f"- Avg Win Rate: {summary['avg_win_rate']:.2f}%"),
            text_for(lang, f"- 平均单笔收益: {summary['avg_trade_return_pct']:.4f}%", f"- Avg Trade Return: {summary['avg_trade_return_pct']:.4f}%"),
            text_for(lang, f"- 总收益: {summary['total_return_pct']:.4f}%", f"- Total Return: {summary['total_return_pct']:.4f}%"),
            text_for(lang, f"- 盈亏因子: {summary['profit_factor']:.4f}", f"- Profit Factor: {summary['profit_factor']:.4f}"),
            text_for(lang, f"- 类夏普值: {summary['sharpe_like']:.4f}", f"- Sharpe-like: {summary['sharpe_like']:.4f}"),
            "",
            text_for(lang, "## 分段结果", "## Splits"),
        ]
        for index, split in enumerate(result["splits"], start=1):
            lines.append(text_for(
                lang,
                f"{index}. 训练={split['train_rows']} 测试={split['test_rows']} 胜率={split['win_rate']:.2f}% 平均收益={split['avg_trade_return_pct']:.4f}% 总收益={split['total_return_pct']:.4f}%",
                f"{index}. train={split['train_rows']} test={split['test_rows']} win_rate={split['win_rate']:.2f}% avg_trade_return={split['avg_trade_return_pct']:.4f}% total_return={split['total_return_pct']:.4f}%",
            ))
        return "\n".join(lines)

    def _predict_probabilities(
        self,
        x_train: list[list[float]],
        y_train: list[int],
        x_test: list[list[float]],
        feature_names: list[str],
        xgb,
    ) -> list[float]:
        if xgb is not None:
            train_matrix = xgb.DMatrix(
                x_train,
                label=y_train,
                feature_names=feature_names,
            )
            test_matrix = xgb.DMatrix(x_test, feature_names=feature_names)
            booster = xgb.train(
                {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "max_depth": 6,
                    "learning_rate": 0.01,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "min_child_weight": 3,
                    "gamma": 0.1,
                    "reg_alpha": 0.1,
                    "reg_lambda": 1.0,
                    "seed": 42,
                },
                train_matrix,
                num_boost_round=120,
            )
            return [float(value) for value in booster.predict(test_matrix)]

        baseline = sum(y_train) / len(y_train) if y_train else 0.5
        return [baseline for _ in x_test]

    @staticmethod
    def _summary(
        trade_returns: list[float],
        splits: list[WalkForwardSplit],
    ) -> dict:
        if not splits:
            return {
                "total_splits": 0,
                "avg_win_rate": 0.0,
                "avg_trade_return_pct": 0.0,
                "total_return_pct": 0.0,
                "profit_factor": 0.0,
                "sharpe_like": 0.0,
            }

        wins = [ret for ret in trade_returns if ret > 0]
        losses = [ret for ret in trade_returns if ret <= 0]
        profit_factor = (
            sum(wins) / abs(sum(losses))
            if losses and abs(sum(losses)) > 1e-12
            else 0.0
        )
        avg_return = sum(trade_returns) / len(trade_returns) if trade_returns else 0.0
        std = np.std(trade_returns) if len(trade_returns) > 1 else 0.0
        sharpe_like = (avg_return / std * sqrt(len(trade_returns))) if std > 1e-12 else 0.0
        return {
            "total_splits": len(splits),
            "avg_win_rate": sum(split.win_rate for split in splits) / len(splits),
            "avg_trade_return_pct": (
                sum(split.avg_trade_return_pct for split in splits) / len(splits)
            ),
            "total_return_pct": sum(trade_returns),
            "profit_factor": profit_factor,
            "sharpe_like": sharpe_like,
        }
