"""Helpers for parsing report metrics across localized markdown reports."""
from __future__ import annotations


METRIC_KEY_ALIASES = {
    "已平仓交易数": "Closed Trades",
    "胜率": "Win Rate",
    "累计已实现盈亏": "Total Realized PnL",
    "平均单笔盈亏": "Average Trade PnL",
    "累计交易成本": "Total Trade Cost",
    "累计滑点拖累": "Total Slippage Drag",
    "权益累计收益": "Equity Return",
    "最近闭环交易数": "Recent Closed Trades",
    "最近净期望收益": "Recent Net Expectancy",
    "最近净盈亏比": "Recent Net Profit Factor",
    "最近净回撤": "Recent Net Max Drawdown",
    "最近净 Sharpe": "Recent Net Sharpe",
    "最近净 Sortino": "Recent Net Sortino",
    "平均持有时长": "Average Holding Hours",
    "最新权益": "Latest Equity",
    "最新回撤": "Latest Drawdown",
    "最近 Holdout 准确率": "Latest Holdout Accuracy",
    "最近 Walk-Forward 收益": "Latest Walk-Forward Return",
    "已评估预测数": "Evaluated Predictions",
    "最近预测窗口样本": "Recent Prediction Window",
    "扩展预测窗口样本": "Expanded Prediction Window",
    "当前执行宇宙预测窗口样本": "Current Execution Universe Window",
    "当前执行宇宙 XGBoost 方向准确率": "Current Execution Universe XGBoost Direction Accuracy",
    "当前执行宇宙 LLM 动作准确率": "Current Execution Universe LLM Action Accuracy",
    "当前执行宇宙 融合信号准确率": "Current Execution Universe Fusion Signal Accuracy",
    "当前执行宇宙 执行闭环准确率": "Current Execution Universe Execution Accuracy",
    "XGBoost 方向准确率": "XGBoost Direction Accuracy",
    "LLM 动作准确率": "LLM Action Accuracy",
    "融合信号准确率": "Fusion Signal Accuracy",
    "扩展窗口 XGBoost 方向准确率": "Expanded XGBoost Direction Accuracy",
    "扩展窗口 LLM 动作准确率": "Expanded LLM Action Accuracy",
    "扩展窗口 融合信号准确率": "Expanded Fusion Signal Accuracy",
    "执行闭环准确率": "Execution Accuracy",
    "Shadow 观察准确率": "Shadow Accuracy",
    "衰减状态": "Degradation Status",
    "衰减原因": "Degradation Reason",
    "建议 XGB 阈值": "Recommended XGB Threshold",
    "建议最终阈值": "Recommended Final Threshold",
    "净收益": "Daily Net PnL",
    "盈亏因子": "Daily Profit Factor",
    "最大回撤": "Daily Max Drawdown",
    "每笔平均净收益": "Average Net PnL Per Trade",
    "风控避免亏损": "Risk-Blocked Loss Avoided",
    "Fast Alpha 净收益": "Fast Alpha Net PnL",
    "Fast Alpha 平仓数": "Fast Alpha Closed Trades",
    "Fast Alpha 开仓数": "Fast Alpha Opens",
    "Short-horizon 放行开仓数": "Short-horizon Softened Opens",
    "Short-horizon 放行净收益": "Short-horizon Softened Net PnL",
    "Short-horizon 放行平仓数": "Short-horizon Softened Closed Trades",
    "Short-horizon 负期望暂停次数": "Short-horizon Negative-Edge Pauses",
    "Short-horizon 当前状态": "Short-Horizon Status",
    "Short-horizon 最近样本": "Short-Horizon Recent Samples",
    "Short-horizon 最近净期望": "Short-Horizon Recent Expectancy",
    "Short-horizon 最近净盈亏比": "Short-Horizon Recent Profit Factor",
    "Short-horizon 放行后状态": "Short-Horizon Softened Status",
    "Short-horizon 放行后净期望": "Short-Horizon Softened Expectancy",
    "Short-horizon 放行后净盈亏比": "Short-Horizon Softened Profit Factor",
    "Fast Alpha 总开仓数": "Fast Alpha Total Opens",
    "Fast Alpha 最近24h开仓数": "Fast Alpha Last 24h Opens",
    "Fast Alpha 胜率": "Fast Alpha Win Rate",
    "Fast Alpha 平均净收益": "Fast Alpha Avg Net Return",
    "Fast Alpha 执行准确率": "Fast Alpha Execution Accuracy",
    "可疑标的数": "Suspicious Symbols",
    "最近实盘净期望": "Recent Live Expectancy",
    "最近实盘盈亏因子": "Recent Live Profit Factor",
    "最近实盘最大回撤": "Recent Live Max Drawdown",
}


def canonical_metric_key(key: str) -> str:
    return METRIC_KEY_ALIASES.get((key or "").strip(), (key or "").strip())


def parse_markdown_metrics(content: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for line in (content or "").splitlines():
        if not line.startswith("- "):
            continue
        body = line[2:]
        if ": " not in body:
            continue
        key, value = body.split(": ", 1)
        metrics[canonical_metric_key(key)] = value.strip()
    return metrics
