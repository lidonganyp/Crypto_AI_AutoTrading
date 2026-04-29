"""Model training utilities for CryptoAI v3."""
from __future__ import annotations

import json
import math
import os
import shutil
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from config import Settings, resolve_project_path
from core.i18n import get_default_language, normalize_language, text_for
from core.feature_pipeline import FeatureInput, FeaturePipeline
from core.scoring import objective_score_from_metrics, objective_score_quality
from core.storage import Storage
from core.trade_simulation import (
    effective_trade_horizon_hours,
    horizon_bars,
    simulate_long_trade,
)
from monitor.backtest_live_consistency_report import BacktestLiveConsistencyReporter


def symbol_model_suffix(symbol: str) -> str:
    return (
        str(symbol)
        .strip()
        .upper()
        .replace("/", "_")
        .replace(":", "_")
        .replace("-", "_")
    )


def model_path_for_symbol(base_path: str | Path, symbol: str) -> Path:
    path = Path(base_path)
    suffix = path.suffix or ".json"
    return path.with_name(f"{path.stem}_{symbol_model_suffix(symbol)}{suffix}")


@dataclass
class TrainingSummary:
    symbol: str
    rows: int
    feature_count: int
    positives: int
    negatives: int
    model_path: str
    trained_with_xgboost: bool
    model_id: str = ""
    active_model_path: str = ""
    active_model_id: str = ""
    challenger_model_path: str = ""
    challenger_model_id: str = ""
    incumbent_model_id: str = ""
    promoted_to_active: bool = False
    promotion_status: str = ""
    promotion_reason: str = ""
    holdout_rows: int = 0
    holdout_accuracy: float = 0.0
    holdout_logloss: float = 0.0
    candidate_holdout_accuracy: float = 0.0
    candidate_holdout_logloss: float = 0.0
    incumbent_holdout_accuracy: float | None = None
    incumbent_holdout_logloss: float | None = None
    candidate_walkforward_summary: dict[str, float | int | str] = field(default_factory=dict)
    candidate_walkforward_splits: list[dict[str, float | int]] = field(default_factory=list)
    incumbent_walkforward_summary: dict[str, float | int | str] = field(default_factory=dict)
    recent_walkforward_baseline_summary: dict[str, float | int | str] = field(default_factory=dict)
    previous_active_backup_path: str = ""
    previous_active_backup_meta_path: str = ""
    top_features: list[str] = field(default_factory=list)
    dataset_start_timestamp_ms: int = 0
    dataset_end_timestamp_ms: int = 0
    dataset_start_at: str = ""
    dataset_end_at: str = ""
    reason: str = ""


def load_xgboost():
    try:
        import xgboost as xgb
    except Exception:  # pragma: no cover
        return None
    return xgb


class ModelTrainer:
    """Build datasets and train XGBoost when available."""

    def __init__(
        self,
        storage: Storage,
        settings: Settings,
        pipeline: FeaturePipeline | None = None,
    ):
        self.storage = storage
        self.settings = settings
        self.pipeline = pipeline or FeaturePipeline()
        self.base_model_path = resolve_project_path(
            self.settings.model.xgboost_model_path,
            self.settings,
        )
        self.base_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.challenger_base_model_path = resolve_project_path(
            self.settings.ab_testing.challenger_model_path,
            self.settings,
        )
        self.challenger_base_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_model_dir = self.base_model_path.parent / "backups"
        self.backup_model_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return (
            str(symbol)
            .strip()
            .upper()
            .replace(" ", "")
            .replace("-", "/")
        )

    def _fast_alpha_training_enabled_for_symbol(self, symbol: str) -> bool:
        if not bool(self.settings.training.fast_alpha_training_enabled):
            return False
        fast_symbols = {
            self._normalize_symbol(item)
            for item in (self.settings.strategy.fast_alpha_symbols or [])
            if str(item).strip()
        }
        return self._normalize_symbol(symbol) in fast_symbols

    def _training_profile(self, symbol: str) -> dict[str, float | int | str]:
        if self._fast_alpha_training_enabled_for_symbol(symbol):
            return {
                "mode": "fast_alpha",
                "sample_timeframe": str(self.settings.strategy.lower_timeframe or "1h"),
                "horizon_hours": max(
                    1,
                    int(
                        self.settings.training.fast_alpha_training_horizon_hours
                        or self.settings.strategy.fast_alpha_max_hold_hours
                        or 6
                    ),
                ),
                "label_min_abs_net_return_pct": max(
                    0.0,
                    float(
                        self.settings.training.fast_alpha_label_min_abs_net_return_pct
                        or 0.0
                    ),
                ),
                "start_index": 120,
            }
        return {
            "mode": "default",
            "sample_timeframe": str(self.settings.strategy.primary_timeframe or "4h"),
            "horizon_hours": int(effective_trade_horizon_hours(self.settings)),
            "label_min_abs_net_return_pct": 0.0,
            "start_index": 55,
        }

    @staticmethod
    def _label_from_net_return_pct(
        net_return_pct: float,
        min_abs_net_return_pct: float,
    ) -> int | None:
        threshold = max(0.0, float(min_abs_net_return_pct or 0.0))
        if net_return_pct >= threshold:
            return 1
        if net_return_pct <= -threshold:
            return 0
        return None

    def train_symbol(self, symbol: str) -> TrainingSummary:
        dataset = self.build_dataset(symbol)
        active_model_path = model_path_for_symbol(self.base_model_path, symbol)
        challenger_model_path = model_path_for_symbol(
            self.challenger_base_model_path,
            symbol,
        )
        incumbent_metadata = self._read_model_metadata(active_model_path)
        incumbent_model_id = self._resolve_existing_model_id(
            active_model_path,
            incumbent_metadata,
        )
        active_model_path.parent.mkdir(parents=True, exist_ok=True)
        challenger_model_path.parent.mkdir(parents=True, exist_ok=True)
        rows = len(dataset["labels"])
        feature_count = len(dataset["feature_names"])
        positives = sum(dataset["labels"])
        negatives = rows - positives
        timestamps = [int(ts) for ts in dataset.get("timestamps", []) if ts is not None]
        dataset_start_timestamp_ms = timestamps[0] if timestamps else 0
        dataset_end_timestamp_ms = timestamps[-1] if timestamps else 0
        dataset_start_at = self._format_timestamp_ms(dataset_start_timestamp_ms)
        dataset_end_at = self._format_timestamp_ms(dataset_end_timestamp_ms)
        candidate_model_id = self._build_model_id(symbol, dataset_end_timestamp_ms)

        if rows < self.settings.training.minimum_training_rows:
            summary = TrainingSummary(
                symbol=symbol,
                rows=rows,
                feature_count=feature_count,
                positives=positives,
                negatives=negatives,
                model_path=str(active_model_path),
                active_model_path=str(active_model_path),
                challenger_model_path=str(challenger_model_path),
                trained_with_xgboost=False,
                incumbent_model_id=incumbent_model_id,
                dataset_start_timestamp_ms=dataset_start_timestamp_ms,
                dataset_end_timestamp_ms=dataset_end_timestamp_ms,
                dataset_start_at=dataset_start_at,
                dataset_end_at=dataset_end_at,
                reason="insufficient_rows",
            )
            self._write_metadata(summary)
            return summary

        xgb = load_xgboost()
        if xgb is None:
            logger.warning("xgboost not installed, writing fallback metadata only")
            summary = TrainingSummary(
                symbol=symbol,
                rows=rows,
                feature_count=feature_count,
                positives=positives,
                negatives=negatives,
                model_path=str(active_model_path),
                active_model_path=str(active_model_path),
                challenger_model_path=str(challenger_model_path),
                trained_with_xgboost=False,
                incumbent_model_id=incumbent_model_id,
                dataset_start_timestamp_ms=dataset_start_timestamp_ms,
                dataset_end_timestamp_ms=dataset_end_timestamp_ms,
                dataset_start_at=dataset_start_at,
                dataset_end_at=dataset_end_at,
                reason="xgboost_missing",
            )
            self._write_metadata(summary)
            return summary

        split_index = max(self.settings.training.minimum_training_rows, int(rows * 0.8))
        split_index = min(split_index, rows - 1)
        x_train = dataset["rows"][:split_index]
        y_train = dataset["labels"][:split_index]
        x_test = dataset["rows"][split_index:]
        y_test = dataset["labels"][split_index:]

        booster = self._train_booster(
            x_train=x_train,
            y_train=y_train,
            feature_names=dataset["feature_names"],
            xgb=xgb,
        )
        candidate_holdout_accuracy, candidate_holdout_logloss = self._evaluate_booster(
            booster=booster,
            x_test=x_test,
            y_test=y_test,
            feature_names=dataset["feature_names"],
            xgb=xgb,
        )
        incumbent_metrics = self._evaluate_model_path(
            model_path=active_model_path,
            x_test=x_test,
            y_test=y_test,
            feature_names=dataset["feature_names"],
            xgb=xgb,
        )
        candidate_walkforward = self._candidate_walkforward_analysis(
            symbol=symbol,
            dataset=dataset,
            xgb=xgb,
        )
        incumbent_walkforward_summary = self._latest_walkforward_summary(symbol)
        recent_walkforward_baseline_summary = self._recent_walkforward_baseline(symbol)
        candidate_qualified, promotion_status, promotion_reason = (
            self._promotion_decision(
                candidate_accuracy=candidate_holdout_accuracy,
                candidate_logloss=candidate_holdout_logloss,
                incumbent_metrics=incumbent_metrics,
                candidate_walkforward_summary=candidate_walkforward["summary"],
                incumbent_walkforward_summary=incumbent_walkforward_summary,
                recent_walkforward_baseline_summary=recent_walkforward_baseline_summary,
            )
        )
        if candidate_qualified:
            consistency_reason = self._walkforward_live_consistency_veto(
                symbol=symbol,
                candidate_walkforward_summary=candidate_walkforward["summary"],
            )
            if consistency_reason:
                candidate_qualified = False
                promotion_status = "rejected"
                promotion_reason = consistency_reason
        promoted_to_active = candidate_qualified and incumbent_metrics is None
        if candidate_qualified and incumbent_metrics is not None:
            promotion_status = "canary_pending"
        backup_paths = {"model_path": "", "meta_path": ""}
        if promoted_to_active:
            backup_paths = self._backup_active_model_artifacts(
                symbol=symbol,
                active_model_path=active_model_path,
                dataset_end_timestamp_ms=dataset_end_timestamp_ms,
            )
            self._save_model_atomic(booster, active_model_path)
            self._remove_model_artifacts(challenger_model_path)
            model_path = active_model_path
            effective_holdout_accuracy = candidate_holdout_accuracy
            effective_holdout_logloss = candidate_holdout_logloss
        else:
            self._save_model_atomic(booster, challenger_model_path)
            model_path = challenger_model_path
            effective_holdout_accuracy = float(
                incumbent_metrics["accuracy"]
                if incumbent_metrics is not None
                else candidate_holdout_accuracy
            )
            effective_holdout_logloss = float(
                incumbent_metrics["logloss"]
                if incumbent_metrics is not None
                else candidate_holdout_logloss
            )

        score_map = booster.get_score(importance_type="gain")
        top_features = [
            name for name, _ in sorted(
                score_map.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:5]
        ]
        summary = TrainingSummary(
            symbol=symbol,
            rows=rows,
            feature_count=feature_count,
            positives=positives,
            negatives=negatives,
            model_path=str(model_path),
            active_model_path=str(active_model_path),
            active_model_id=(
                candidate_model_id if promoted_to_active else incumbent_model_id
            ),
            challenger_model_path=(
                str(challenger_model_path)
                if not promoted_to_active
                else ""
            ),
            challenger_model_id=(
                candidate_model_id if not promoted_to_active else ""
            ),
            incumbent_model_id=incumbent_model_id,
            trained_with_xgboost=True,
            model_id=candidate_model_id,
            promoted_to_active=promoted_to_active,
            promotion_status=promotion_status,
            promotion_reason=promotion_reason,
            holdout_rows=len(y_test),
            holdout_accuracy=effective_holdout_accuracy,
            holdout_logloss=effective_holdout_logloss,
            candidate_holdout_accuracy=candidate_holdout_accuracy,
            candidate_holdout_logloss=candidate_holdout_logloss,
            incumbent_holdout_accuracy=(
                None
                if incumbent_metrics is None
                else float(incumbent_metrics["accuracy"])
            ),
            incumbent_holdout_logloss=(
                None
                if incumbent_metrics is None
                else float(incumbent_metrics["logloss"])
            ),
            candidate_walkforward_summary=dict(candidate_walkforward["summary"]),
            candidate_walkforward_splits=list(candidate_walkforward["splits"]),
            incumbent_walkforward_summary=dict(incumbent_walkforward_summary),
            recent_walkforward_baseline_summary=dict(recent_walkforward_baseline_summary),
            previous_active_backup_path=str(backup_paths.get("model_path") or ""),
            previous_active_backup_meta_path=str(backup_paths.get("meta_path") or ""),
            top_features=top_features,
            dataset_start_timestamp_ms=dataset_start_timestamp_ms,
            dataset_end_timestamp_ms=dataset_end_timestamp_ms,
            dataset_start_at=dataset_start_at,
            dataset_end_at=dataset_end_at,
            reason="trained",
        )
        self._write_metadata(summary)
        logger.info(
            f"Trained XGBoost model for {symbol} with {rows} rows "
            f"(candidate_holdout_acc={candidate_holdout_accuracy:.3f}, "
            f"promotion_status={promotion_status}, "
            f"promotion_reason={promotion_reason})"
        )
        return summary

    def build_dataset(self, symbol: str) -> dict:
        market_symbol = f"{symbol}:USDT" if ":USDT" not in symbol else symbol
        candles_1h = self.storage.get_ohlcv(
            market_symbol,
            "1h",
            limit=self.settings.training.dataset_limit_1h,
        )
        candles_4h = self.storage.get_ohlcv(
            market_symbol,
            "4h",
            limit=self.settings.training.dataset_limit_4h,
        )
        candles_1d = self.storage.get_ohlcv(
            market_symbol,
            "1d",
            limit=self.settings.training.dataset_limit_1d,
        )
        candles_1h.sort(key=lambda item: item["timestamp"])
        candles_4h.sort(key=lambda item: item["timestamp"])
        candles_1d.sort(key=lambda item: item["timestamp"])

        profile = self._training_profile(symbol)
        sample_timeframe = str(profile["sample_timeframe"])
        sample_candles = candles_1h if sample_timeframe == "1h" else candles_4h
        if len(candles_4h) < 60 or len(sample_candles) < int(profile["start_index"]) + 2:
            return {"rows": [], "labels": [], "feature_names": []}

        rows: list[list[float]] = []
        labels: list[int] = []
        feature_names: list[str] = []
        timestamps: list[int] = []
        current_closes: list[float] = []
        next_closes: list[float] = []
        trade_return_pcts: list[float] = []
        horizon_hours = int(profile["horizon_hours"])
        future_bars = horizon_bars(horizon_hours, sample_timeframe)
        min_label_abs_net_return_pct = float(profile["label_min_abs_net_return_pct"])
        start_index = int(profile["start_index"])

        for index in range(start_index, len(sample_candles) - future_bars):
            current_timestamp = int(sample_candles[index]["timestamp"])
            current_close = float(sample_candles[index]["close"])

            one_h_subset = [
                candle for candle in candles_1h if candle["timestamp"] <= current_timestamp
            ][-240:]
            four_h_subset = [
                candle for candle in candles_4h if candle["timestamp"] <= current_timestamp
            ][-240:]
            one_d_subset = [
                candle for candle in candles_1d if candle["timestamp"] <= current_timestamp
            ][-240:]
            if len(one_h_subset) < 120 or len(four_h_subset) < 60 or len(one_d_subset) < 30:
                continue

            snapshot = self.pipeline.build(
                FeatureInput(
                    symbol=symbol,
                    candles_1h=one_h_subset,
                    candles_4h=four_h_subset,
                    candles_1d=one_d_subset,
                )
            )
            if not snapshot.valid or not snapshot.values:
                continue

            outcome = simulate_long_trade(
                future_candles=sample_candles[index + 1 : index + 1 + future_bars],
                entry_price=float(current_close),
                timeframe=sample_timeframe,
                max_hold_hours=horizon_hours,
                stop_loss_pct=self.settings.strategy.fixed_stop_loss_pct,
                take_profit_levels=self.settings.strategy.take_profit_levels,
            )
            if outcome is None:
                continue

            net_return_pct = float(outcome["net_return_pct"])
            label = self._label_from_net_return_pct(
                net_return_pct,
                min_label_abs_net_return_pct,
            )
            if label is None:
                continue
            if not feature_names:
                feature_names = list(snapshot.values.keys())
            rows.append([snapshot.values[name] for name in feature_names])
            labels.append(label)
            timestamps.append(current_timestamp)
            current_closes.append(float(current_close))
            next_closes.append(float(outcome["exit_price"]))
            trade_return_pcts.append(net_return_pct)

        return {
            "rows": rows,
            "labels": labels,
            "feature_names": feature_names,
            "timestamps": timestamps,
            "current_closes": current_closes,
            "next_closes": next_closes,
            "trade_return_pcts": trade_return_pcts,
        }

    def count_training_rows(self, symbol: str) -> int:
        signature = self.training_data_signature(symbol)
        return int(signature["rows"])

    def training_data_signature(self, symbol: str) -> dict[str, int]:
        dataset = self.build_dataset(symbol)
        timestamps = [
            int(timestamp)
            for timestamp in (dataset.get("timestamps") or [])
            if timestamp is not None
        ]
        return {
            "rows": len(dataset.get("labels") or []),
            "start_timestamp": timestamps[0] if timestamps else 0,
            "end_timestamp": timestamps[-1] if timestamps else 0,
        }

    @staticmethod
    def _format_timestamp_ms(timestamp_ms: int) -> str:
        if timestamp_ms <= 0:
            return ""
        return datetime.fromtimestamp(
            timestamp_ms / 1000,
            tz=timezone.utc,
        ).isoformat()

    @staticmethod
    def _read_model_metadata(model_path: Path) -> dict:
        meta_path = model_path.with_suffix(".meta.json")
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _resolve_existing_model_id(model_path: Path, metadata: dict | None = None) -> str:
        payload = metadata if isinstance(metadata, dict) else {}
        model_id = str(payload.get("model_id") or "").strip()
        if model_id:
            return model_id
        try:
            stat = model_path.stat()
        except OSError:
            return ""
        return f"{model_path.name}@{int(stat.st_mtime_ns)}"

    @staticmethod
    def _build_model_id(symbol: str, dataset_end_timestamp_ms: int) -> str:
        suffix = (
            str(dataset_end_timestamp_ms)
            if dataset_end_timestamp_ms > 0
            else datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        )
        return f"mdl_{symbol_model_suffix(symbol)}_{suffix}"

    @staticmethod
    def render_report(summary: TrainingSummary, lang: str | None = None) -> str:
        lang = normalize_language(lang or get_default_language())
        lines = [
            text_for(lang, f"# 训练报告: {summary.symbol}", f"# Training Report: {summary.symbol}"),
            text_for(lang, f"- 样本数: {summary.rows}", f"- Rows: {summary.rows}"),
            text_for(lang, f"- 特征数: {summary.feature_count}", f"- Feature Count: {summary.feature_count}"),
            text_for(lang, f"- 正样本数: {summary.positives}", f"- Positive Labels: {summary.positives}"),
            text_for(lang, f"- 负样本数: {summary.negatives}", f"- Negative Labels: {summary.negatives}"),
            text_for(lang, f"- 是否使用 XGBoost: {summary.trained_with_xgboost}", f"- XGBoost Used: {summary.trained_with_xgboost}"),
            text_for(lang, f"- 训练结果: {summary.reason}", f"- Reason: {summary.reason}"),
            text_for(lang, f"- 模型晋级状态: {summary.promotion_status or 'n/a'}", f"- Promotion Status: {summary.promotion_status or 'n/a'}"),
            text_for(lang, f"- 晋级原因: {summary.promotion_reason or 'n/a'}", f"- Promotion Reason: {summary.promotion_reason or 'n/a'}"),
            text_for(lang, f"- Holdout 样本数: {summary.holdout_rows}", f"- Holdout Rows: {summary.holdout_rows}"),
            text_for(lang, f"- 已部署模型 Holdout 准确率: {summary.holdout_accuracy:.4f}", f"- Active Holdout Accuracy: {summary.holdout_accuracy:.4f}"),
            f"- Active Holdout Logloss: {summary.holdout_logloss:.6f}",
            text_for(lang, f"- 候选模型 Holdout 准确率: {summary.candidate_holdout_accuracy:.4f}", f"- Candidate Holdout Accuracy: {summary.candidate_holdout_accuracy:.4f}"),
            f"- Candidate Holdout Logloss: {summary.candidate_holdout_logloss:.6f}",
        ]
        consistency_note = ModelTrainer._promotion_reason_display_note(
            summary.promotion_reason,
            lang=lang,
        )
        if consistency_note:
            lines.append(consistency_note)
        if summary.incumbent_holdout_accuracy is not None:
            lines.append(
                text_for(
                    lang,
                    f"- 旧模型 Holdout 准确率: {summary.incumbent_holdout_accuracy:.4f}",
                    f"- Incumbent Holdout Accuracy: {summary.incumbent_holdout_accuracy:.4f}",
                )
            )
        if summary.incumbent_holdout_logloss is not None:
            lines.append(
                f"- Incumbent Holdout Logloss: {summary.incumbent_holdout_logloss:.6f}"
            )
        candidate_wf = summary.candidate_walkforward_summary or {}
        if candidate_wf:
            lines.append(
                text_for(
                    lang,
                    f"- 候选模型 Walk-Forward 收益: {float(candidate_wf.get('total_return_pct', 0.0)):.4f}%",
                    f"- Candidate Walk-Forward Return: {float(candidate_wf.get('total_return_pct', 0.0)):.4f}%",
                )
            )
            lines.append(
                text_for(
                    lang,
                    f"- 候选模型 Walk-Forward 胜率: {float(candidate_wf.get('avg_win_rate', 0.0)):.2f}%",
                    f"- Candidate Walk-Forward Win Rate: {float(candidate_wf.get('avg_win_rate', 0.0)):.2f}%",
                )
            )
            lines.append(
                text_for(
                    lang,
                    f"- 候选模型 Walk-Forward 盈亏因子: {float(candidate_wf.get('profit_factor', 0.0)):.4f}",
                    f"- Candidate Walk-Forward Profit Factor: {float(candidate_wf.get('profit_factor', 0.0)):.4f}",
                )
            )
            lines.append(
                text_for(
                    lang,
                    f"- 候选模型 Walk-Forward 最大回撤: {float(candidate_wf.get('max_drawdown_pct', 0.0)):.4f}%",
                    f"- Candidate Walk-Forward Max Drawdown: {float(candidate_wf.get('max_drawdown_pct', 0.0)):.4f}%",
                )
            )
        incumbent_wf = summary.incumbent_walkforward_summary or {}
        if incumbent_wf:
            lines.append(
                text_for(
                    lang,
                    f"- 旧模型 Walk-Forward 收益: {float(incumbent_wf.get('total_return_pct', 0.0)):.4f}%",
                    f"- Incumbent Walk-Forward Return: {float(incumbent_wf.get('total_return_pct', 0.0)):.4f}%",
                )
            )
        recent_wf = summary.recent_walkforward_baseline_summary or {}
        if recent_wf:
            lines.append(
                text_for(
                    lang,
                    f"- 最近 Walk-Forward 基线收益: {float(recent_wf.get('avg_total_return_pct', 0.0)):.4f}%",
                    f"- Recent Walk-Forward Baseline Return: {float(recent_wf.get('avg_total_return_pct', 0.0)):.4f}%",
                )
            )
            lines.append(
                text_for(
                    lang,
                    f"- 最近 Walk-Forward 基线胜率: {float(recent_wf.get('avg_win_rate', 0.0)):.2f}%",
                    f"- Recent Walk-Forward Baseline Win Rate: {float(recent_wf.get('avg_win_rate', 0.0)):.2f}%",
                )
            )
        if summary.top_features:
            lines.append(
                text_for(
                    lang,
                    f"- 重要特征: {', '.join(summary.top_features)}",
                    f"- Top Features: {', '.join(summary.top_features)}",
                )
            )
        return "\n".join(lines)

    @staticmethod
    def _promotion_reason_display_note(reason: str, *, lang: str) -> str:
        reason_text = str(reason or "").strip()
        if reason_text == "candidate_walkforward_live_pnl_divergence":
            return text_for(
                lang,
                "- 一致性校验: 已拒绝，原因=Walk-Forward 收益与最近实盘净盈亏明显背离",
                "- Consistency Gate: rejected because walk-forward return diverges from recent live net PnL",
            )
        if reason_text == "candidate_walkforward_live_profit_factor_divergence":
            return text_for(
                lang,
                "- 一致性校验: 已拒绝，原因=Walk-Forward 盈亏因子与最近实盘盈亏因子明显背离",
                "- Consistency Gate: rejected because walk-forward profit factor diverges from recent live profit factor",
            )
        return ""

    def _write_metadata(self, summary: TrainingSummary):
        metadata_path = Path(summary.model_path).with_suffix(".meta.json")
        self._write_text_atomic(
            metadata_path,
            json.dumps(summary.__dict__, ensure_ascii=False, indent=2),
        )

    @staticmethod
    def _logloss(probabilities: list[float], labels: list[int]) -> float:
        eps = 1e-12
        total = 0.0
        for probability, label in zip(probabilities, labels):
            probability = min(max(probability, eps), 1.0 - eps)
            total += -(
                label * math.log(probability)
                + (1 - label) * math.log(1 - probability)
            )
        return total / len(labels) if labels else 0.0

    def _train_booster(
        self,
        *,
        x_train: list[list[float]],
        y_train: list[int],
        feature_names: list[str],
        xgb,
        num_boost_round: int | None = None,
    ):
        train_matrix = xgb.DMatrix(
            x_train,
            label=y_train,
            feature_names=feature_names,
        )
        params = {
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
        }
        if self.settings.model.xgboost_nthread > 0:
            params["nthread"] = self.settings.model.xgboost_nthread
        return xgb.train(
            params,
            train_matrix,
            num_boost_round=(
                num_boost_round
                if num_boost_round is not None
                else self.settings.model.xgboost_num_boost_round
            ),
        )

    def _evaluate_booster(
        self,
        *,
        booster,
        x_test: list[list[float]],
        y_test: list[int],
        feature_names: list[str],
        xgb,
    ) -> tuple[float, float]:
        if not x_test:
            return 0.0, 0.0
        test_matrix = xgb.DMatrix(
            x_test,
            label=y_test,
            feature_names=feature_names,
        )
        probabilities = [float(value) for value in booster.predict(test_matrix)]
        predictions = [1 if value >= 0.5 else 0 for value in probabilities]
        accuracy = (
            sum(int(pred == actual) for pred, actual in zip(predictions, y_test))
            / len(y_test)
        )
        return accuracy, self._logloss(probabilities, y_test)

    def _evaluate_model_path(
        self,
        *,
        model_path: Path,
        x_test: list[list[float]],
        y_test: list[int],
        feature_names: list[str],
        xgb,
    ) -> dict[str, float] | None:
        if not x_test or not model_path.exists():
            return None
        try:
            if model_path.stat().st_size <= 0:
                return None
        except OSError:
            return None
        try:
            booster = xgb.Booster()
            booster.load_model(str(model_path))
        except Exception:
            return None
        try:
            accuracy, logloss = self._evaluate_booster(
                booster=booster,
                x_test=x_test,
                y_test=y_test,
                feature_names=feature_names,
                xgb=xgb,
            )
        except Exception:
            return None
        return {"accuracy": accuracy, "logloss": logloss}

    def _candidate_walkforward_analysis(
        self,
        *,
        symbol: str,
        dataset: dict,
        xgb,
    ) -> dict[str, object]:
        rows = dataset.get("rows", [])
        labels = dataset.get("labels", [])
        feature_names = dataset.get("feature_names", [])
        current_closes = dataset.get("current_closes", [])
        next_closes = dataset.get("next_closes", [])
        trade_return_pcts = dataset.get("trade_return_pcts", [])
        total_rows = len(labels)
        if total_rows < self.settings.training.minimum_training_rows:
            return {
                "summary": self._walkforward_summary([], [], symbol=symbol),
                "splits": [],
            }

        train_min = self.settings.training.minimum_training_rows
        test_window = max(1, self.settings.training.walkforward_window_days * 6)
        threshold = self.settings.model.xgboost_probability_threshold
        splits: list[dict[str, float | int]] = []
        trade_returns: list[float] = []

        for train_end in range(train_min, total_rows - test_window + 1, test_window):
            x_train = rows[:train_end]
            y_train = labels[:train_end]
            x_test = rows[train_end : train_end + test_window]
            y_test = labels[train_end : train_end + test_window]
            test_current = current_closes[train_end : train_end + test_window]
            test_next = next_closes[train_end : train_end + test_window]
            test_trade_returns = trade_return_pcts[train_end : train_end + test_window]
            booster = self._train_booster(
                x_train=x_train,
                y_train=y_train,
                feature_names=feature_names,
                xgb=xgb,
                num_boost_round=min(self.settings.model.xgboost_num_boost_round, 120),
            )
            probabilities = self._predict_probabilities(
                booster=booster,
                x_test=x_test,
                feature_names=feature_names,
                xgb=xgb,
            )
            split_returns: list[float] = []
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
                split_returns.append(trade_return)
                if trade_return > 0:
                    wins += 1
            splits.append(
                {
                    "train_rows": len(x_train),
                    "test_rows": len(x_test),
                    "win_rate": (wins / len(split_returns) * 100) if split_returns else 0.0,
                    "avg_trade_return_pct": (
                        sum(split_returns) / len(split_returns)
                        if split_returns
                        else 0.0
                    ),
                    "total_return_pct": sum(split_returns),
                }
            )
            trade_returns.extend(split_returns)

        return {
            "summary": self._walkforward_summary(
                trade_returns,
                splits,
                symbol=symbol,
            ),
            "splits": splits,
        }

    def _latest_walkforward_summary(self, symbol: str) -> dict[str, float | int | str]:
        with self.storage._conn() as conn:
            row = conn.execute(
                "SELECT summary_json FROM walkforward_runs WHERE symbol = ? "
                "ORDER BY created_at DESC, id DESC LIMIT 1",
                (symbol,),
            ).fetchone()
        if row is None:
            return {}
        try:
            summary = json.loads(row["summary_json"] or "{}")
        except Exception:
            return {}
        return summary if isinstance(summary, dict) else {}

    def _recent_walkforward_baseline(
        self,
        symbol: str,
        limit: int = 3,
    ) -> dict[str, float | int | str]:
        with self.storage._conn() as conn:
            rows = conn.execute(
                "SELECT summary_json FROM walkforward_runs WHERE symbol = ? "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (symbol, limit),
            ).fetchall()
        summaries: list[dict[str, float | int | str]] = []
        for row in rows:
            try:
                summary = json.loads(row["summary_json"] or "{}")
            except Exception:
                continue
            if isinstance(summary, dict):
                summaries.append(summary)
        if len(summaries) < 2:
            return {}

        def mean(keys: str | tuple[str, ...]) -> float:
            key_names = (keys,) if isinstance(keys, str) else keys
            values: list[float] = []
            for summary in summaries:
                value = 0.0
                for key in key_names:
                    if key in summary and summary.get(key) is not None:
                        value = float(summary.get(key, 0.0) or 0.0)
                        break
                values.append(value)
            return sum(values) / len(values) if values else 0.0

        return {
            "symbol": symbol,
            "history_count": len(summaries),
            "avg_total_return_pct": mean("total_return_pct"),
            "avg_trade_return_pct": mean("avg_trade_return_pct"),
            "avg_expectancy_pct": mean(("expectancy_pct", "avg_trade_return_pct")),
            "avg_profit_factor": mean("profit_factor"),
            "avg_win_rate": mean("avg_win_rate"),
            "avg_max_drawdown_pct": mean("max_drawdown_pct"),
            "min_total_return_pct": min(
                float(summary.get("total_return_pct", 0.0) or 0.0)
                for summary in summaries
            ),
        }

    def _backup_active_model_artifacts(
        self,
        *,
        symbol: str,
        active_model_path: Path,
        dataset_end_timestamp_ms: int,
    ) -> dict[str, Path | str]:
        if not active_model_path.exists():
            return {"model_path": "", "meta_path": ""}
        suffix = (
            str(dataset_end_timestamp_ms)
            if dataset_end_timestamp_ms > 0
            else datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        )
        backup_model_path = self.backup_model_dir / (
            f"{active_model_path.stem}_{symbol_model_suffix(symbol)}_{suffix}{active_model_path.suffix}"
        )
        shutil.copy2(active_model_path, backup_model_path)
        active_meta_path = active_model_path.with_suffix(".meta.json")
        backup_meta_path = backup_model_path.with_suffix(".meta.json")
        if active_meta_path.exists():
            shutil.copy2(active_meta_path, backup_meta_path)
        else:
            backup_meta_path = Path("")
        return {
            "model_path": backup_model_path,
            "meta_path": backup_meta_path if str(backup_meta_path) else "",
        }

    @staticmethod
    def _walkforward_objective_snapshot(
        summary: dict[str, float | int | str] | None,
    ) -> dict[str, float | int]:
        payload = summary or {}
        total_splits = int(payload.get("total_splits", 0) or 0)
        trade_count = int(payload.get("trade_count", total_splits) or 0)
        executed_count = max(trade_count, total_splits)
        win_rate = float(payload.get("avg_win_rate", 0.0) or 0.0)
        expectancy_pct = float(
            payload.get("expectancy_pct", payload.get("avg_trade_return_pct", 0.0))
            or 0.0
        )
        avg_trade_return_pct = float(
            payload.get("avg_trade_return_pct", expectancy_pct) or 0.0
        )
        metrics = {
            "sample_count": executed_count,
            "executed_count": executed_count,
            "accuracy": win_rate / 100.0,
            "expectancy_pct": expectancy_pct,
            "avg_trade_return_pct": avg_trade_return_pct,
            "profit_factor": float(payload.get("profit_factor", 0.0) or 0.0),
            "max_drawdown_pct": float(payload.get("max_drawdown_pct", 0.0) or 0.0),
            "trade_win_rate": win_rate / 100.0,
            "avg_cost_pct": 0.0,
        }
        objective_score = objective_score_from_metrics(metrics)
        return {
            "total_splits": total_splits,
            "trade_count": trade_count,
            "total_return_pct": float(payload.get("total_return_pct", 0.0) or 0.0),
            "expectancy_pct": expectancy_pct,
            "avg_trade_return_pct": avg_trade_return_pct,
            "profit_factor": float(payload.get("profit_factor", 0.0) or 0.0),
            "win_rate": win_rate,
            "max_drawdown_pct": float(payload.get("max_drawdown_pct", 0.0) or 0.0),
            "objective_score": objective_score,
            "objective_quality": objective_score_quality(
                {**metrics, "objective_score": objective_score}
            ),
        }

    @staticmethod
    def _promotion_decision(
        *,
        candidate_accuracy: float,
        candidate_logloss: float,
        incumbent_metrics: dict[str, float] | None,
        candidate_walkforward_summary: dict[str, float | int | str] | None,
        incumbent_walkforward_summary: dict[str, float | int | str] | None,
        recent_walkforward_baseline_summary: dict[str, float | int | str] | None,
    ) -> tuple[bool, str, str]:
        eps = 1e-12
        if incumbent_metrics is None:
            return True, "promoted", "no_incumbent_model"
        incumbent_accuracy = float(incumbent_metrics["accuracy"])
        incumbent_logloss = float(incumbent_metrics["logloss"])
        if candidate_accuracy < max(0.45, incumbent_accuracy - 0.18):
            return (
                False,
                "rejected",
                "candidate_holdout_accuracy_below_safety_floor",
            )
        if candidate_logloss > max(0.95, incumbent_logloss + 0.35):
            return (
                False,
                "rejected",
                "candidate_holdout_logloss_above_safety_floor",
            )

        improvement_reason = ""
        candidate_wf = ModelTrainer._walkforward_objective_snapshot(
            candidate_walkforward_summary
        )
        incumbent_wf = ModelTrainer._walkforward_objective_snapshot(
            incumbent_walkforward_summary
        )
        candidate_splits = int(candidate_wf["total_splits"])
        incumbent_splits = int(incumbent_wf["total_splits"])
        candidate_return = float(candidate_wf["total_return_pct"])
        candidate_expectancy = float(candidate_wf["expectancy_pct"])
        candidate_avg_trade_return = float(candidate_wf["avg_trade_return_pct"])
        candidate_profit_factor = float(candidate_wf["profit_factor"])
        candidate_win_rate = float(candidate_wf["win_rate"])
        candidate_max_drawdown = float(candidate_wf["max_drawdown_pct"])
        candidate_objective_score = float(candidate_wf["objective_score"])
        candidate_objective_quality = float(candidate_wf["objective_quality"])

        if candidate_splits > 0:
            if candidate_objective_score < -0.1:
                return False, "rejected", "candidate_negative_walkforward_quality"
            if candidate_expectancy < -0.05:
                return False, "rejected", "candidate_negative_walkforward_expectancy"
            if (
                int(candidate_wf["trade_count"]) >= 3
                and candidate_profit_factor < 0.95
                and candidate_expectancy < 0.05
            ):
                return (
                    False,
                    "rejected",
                    "candidate_subscale_walkforward_profit_factor",
                )

            recent_wf = recent_walkforward_baseline_summary or {}
            recent_count = int(recent_wf.get("history_count", 0) or 0)
            if recent_count >= 2:
                recent_avg_return = float(
                    recent_wf.get("avg_total_return_pct", 0.0) or 0.0
                )
                recent_avg_expectancy = float(
                    recent_wf.get(
                        "avg_expectancy_pct",
                        recent_wf.get("avg_trade_return_pct", 0.0),
                    )
                    or 0.0
                )
                recent_avg_profit_factor = float(
                    recent_wf.get("avg_profit_factor", 0.0) or 0.0
                )
                recent_avg_win_rate = float(
                    recent_wf.get("avg_win_rate", 0.0) or 0.0
                )
                recent_avg_drawdown = float(
                    recent_wf.get("avg_max_drawdown_pct", 0.0) or 0.0
                )
                return_tolerance = max(0.25, abs(recent_avg_return) * 0.4)
                if (
                    recent_avg_return > 0
                    and candidate_return < recent_avg_return - return_tolerance
                ):
                    return (
                        False,
                        "rejected",
                        "candidate_below_recent_walkforward_baseline",
                    )
                if (
                    recent_avg_expectancy > 0
                    and candidate_expectancy
                    < recent_avg_expectancy
                    - max(0.10, abs(recent_avg_expectancy) * 0.35)
                ):
                    return (
                        False,
                        "rejected",
                        "candidate_below_recent_walkforward_expectancy",
                    )
                if (
                    recent_avg_profit_factor > 1.0
                    and candidate_profit_factor < recent_avg_profit_factor - 0.15
                ):
                    return (
                        False,
                        "rejected",
                        "candidate_below_recent_walkforward_profit_factor",
                    )
                if (
                    recent_avg_win_rate > 50.0
                    and candidate_win_rate < recent_avg_win_rate - 8.0
                ):
                    return (
                        False,
                        "rejected",
                        "candidate_below_recent_walkforward_win_rate",
                    )
                if (
                    recent_avg_drawdown > 0
                    and candidate_max_drawdown
                    > max(recent_avg_drawdown + 1.0, recent_avg_drawdown * 1.35)
                ):
                    return (
                        False,
                        "rejected",
                        "candidate_above_recent_walkforward_drawdown",
                    )

            if incumbent_splits > 0:
                incumbent_expectancy = float(incumbent_wf["expectancy_pct"])
                incumbent_avg_trade_return = float(incumbent_wf["avg_trade_return_pct"])
                incumbent_profit_factor = float(incumbent_wf["profit_factor"])
                incumbent_win_rate = float(incumbent_wf["win_rate"])
                incumbent_max_drawdown = float(incumbent_wf["max_drawdown_pct"])
                incumbent_objective_score = float(incumbent_wf["objective_score"])
                incumbent_objective_quality = float(incumbent_wf["objective_quality"])
                if (
                    candidate_objective_quality < incumbent_objective_quality - 0.15
                    and candidate_objective_score < incumbent_objective_score + 0.25
                ):
                    return False, "rejected", "candidate_lower_walkforward_quality"
                if (
                    candidate_expectancy < incumbent_expectancy - 0.10
                    and candidate_profit_factor < incumbent_profit_factor - 0.05
                ):
                    return (
                        False,
                        "rejected",
                        "candidate_lower_walkforward_expectancy",
                    )
                if (
                    candidate_profit_factor < incumbent_profit_factor - 0.15
                    and candidate_expectancy < incumbent_expectancy + 0.05
                ):
                    return (
                        False,
                        "rejected",
                        "candidate_lower_walkforward_profit_factor",
                    )
                if (
                    incumbent_max_drawdown > 0
                    and candidate_max_drawdown
                    > incumbent_max_drawdown
                    + max(1.0, incumbent_max_drawdown * 0.25)
                    and candidate_expectancy < incumbent_expectancy + 0.10
                ):
                    return (
                        False,
                        "rejected",
                        "candidate_higher_walkforward_drawdown",
                    )
                if candidate_expectancy > incumbent_expectancy + 0.10:
                    improvement_reason = "candidate_higher_walkforward_expectancy"
                elif candidate_profit_factor > incumbent_profit_factor + 0.10:
                    improvement_reason = "candidate_higher_walkforward_profit_factor"
                elif (
                    incumbent_max_drawdown > 0
                    and candidate_max_drawdown
                    < incumbent_max_drawdown
                    - max(1.0, incumbent_max_drawdown * 0.15)
                    and candidate_expectancy >= incumbent_expectancy - 0.05
                ):
                    improvement_reason = "candidate_lower_walkforward_drawdown"
                elif candidate_avg_trade_return > incumbent_avg_trade_return + 0.10:
                    improvement_reason = (
                        "candidate_higher_walkforward_avg_trade_return"
                    )
                elif candidate_objective_score > incumbent_objective_score + 0.25:
                    improvement_reason = "candidate_higher_walkforward_quality"
                elif (
                    candidate_win_rate > incumbent_win_rate + 3.0
                    and candidate_expectancy >= incumbent_expectancy - 0.02
                ):
                    improvement_reason = "candidate_higher_walkforward_win_rate"
            else:
                if (
                    candidate_profit_factor < 1.0 - eps
                    and candidate_expectancy <= 0.0
                ):
                    return (
                        False,
                        "rejected",
                        "candidate_negative_walkforward_expectancy",
                    )
                if candidate_expectancy > 0.10:
                    improvement_reason = "candidate_positive_walkforward_expectancy"
                elif candidate_objective_score > 0.25:
                    improvement_reason = "candidate_positive_walkforward_quality"

        if not improvement_reason:
            if (
                candidate_accuracy > incumbent_accuracy + 0.05
                and candidate_logloss <= incumbent_logloss + 0.02
            ):
                improvement_reason = "candidate_higher_holdout_accuracy"
            elif (
                candidate_logloss < incumbent_logloss - 0.08
                and candidate_accuracy >= incumbent_accuracy - 0.02
            ):
                improvement_reason = "candidate_lower_holdout_logloss"

        if not improvement_reason:
            return False, "rejected", "candidate_not_better_than_incumbent"
        return True, "promoted", improvement_reason

    def _walkforward_live_consistency_veto(
        self,
        *,
        symbol: str,
        candidate_walkforward_summary: dict[str, float | int | str] | None,
    ) -> str:
        reporter = BacktestLiveConsistencyReporter(self.storage, self.settings)
        flags = set(
            reporter.consistency_flags(
                symbol,
                walkforward_override=candidate_walkforward_summary,
            )
        )
        if "walkforward_live_pnl_divergence" in flags:
            return "candidate_walkforward_live_pnl_divergence"
        if "profit_factor_divergence" in flags:
            return "candidate_walkforward_live_profit_factor_divergence"
        return ""

    @staticmethod
    def _predict_probabilities(
        *,
        booster,
        x_test: list[list[float]],
        feature_names: list[str],
        xgb,
    ) -> list[float]:
        if not x_test:
            return []
        test_matrix = xgb.DMatrix(x_test, feature_names=feature_names)
        return [float(value) for value in booster.predict(test_matrix)]

    @staticmethod
    def _walkforward_summary(
        trade_returns: list[float],
        splits: list[dict[str, float | int]],
        *,
        symbol: str = "",
    ) -> dict[str, float | int | str]:
        if not splits:
            return {
                "symbol": symbol,
                "total_splits": 0,
                "avg_win_rate": 0.0,
                "avg_trade_return_pct": 0.0,
                "expectancy_pct": 0.0,
                "total_return_pct": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_pct": 0.0,
                "trade_count": 0,
                "positive_trade_count": 0,
                "sharpe_like": 0.0,
            }
        wins = [ret for ret in trade_returns if ret > 0]
        losses = [ret for ret in trade_returns if ret <= 0]
        profit_factor = (
            sum(wins) / abs(sum(losses))
            if losses and abs(sum(losses)) > 1e-12
            else (5.0 if wins else 0.0)
        )
        avg_return = sum(trade_returns) / len(trade_returns) if trade_returns else 0.0
        if len(trade_returns) > 1:
            variance = sum((ret - avg_return) ** 2 for ret in trade_returns) / len(
                trade_returns
            )
            std = math.sqrt(variance)
        else:
            std = 0.0
        sharpe_like = (
            avg_return / std * math.sqrt(len(trade_returns))
            if std > 1e-12
            else 0.0
        )
        max_drawdown_pct = ModelTrainer._returns_max_drawdown_pct(trade_returns)
        return {
            "symbol": symbol,
            "total_splits": len(splits),
            "avg_win_rate": sum(float(split["win_rate"]) for split in splits) / len(splits),
            "avg_trade_return_pct": (
                sum(float(split["avg_trade_return_pct"]) for split in splits)
                / len(splits)
            ),
            "expectancy_pct": avg_return,
            "total_return_pct": sum(trade_returns),
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_drawdown_pct,
            "trade_count": len(trade_returns),
            "positive_trade_count": len(wins),
            "sharpe_like": sharpe_like,
        }

    @staticmethod
    def _returns_max_drawdown_pct(trade_returns: list[float]) -> float:
        equity = 1.0
        peak = 1.0
        max_drawdown = 0.0
        for raw_return in trade_returns:
            equity *= 1.0 + float(raw_return) / 100.0
            peak = max(peak, equity)
            if peak > 0:
                max_drawdown = max(max_drawdown, (peak - equity) / peak)
        return max_drawdown * 100

    @staticmethod
    def _save_model_atomic(booster, model_path: Path) -> None:
        temp_path = model_path.with_name(
            f".{model_path.stem}.tmp{model_path.suffix}"
        )
        booster.save_model(str(temp_path))
        os.replace(temp_path, model_path)

    @staticmethod
    def _write_text_atomic(path: Path, content: str) -> None:
        temp_path = path.with_name(f".{path.name}.tmp")
        temp_path.write_text(content, encoding="utf-8")
        os.replace(temp_path, path)

    @staticmethod
    def _remove_model_artifacts(model_path: Path) -> None:
        model_path.unlink(missing_ok=True)
        model_path.with_suffix(".meta.json").unlink(missing_ok=True)
