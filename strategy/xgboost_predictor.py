"""XGBoost prediction wrapper with a deterministic fallback."""
from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from core.models import FeatureSnapshot, PredictionResult

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - optional dependency
    xgb = None


class XGBoostPredictor:
    """Predict 4h up-probability from engineered features."""

    def __init__(
        self,
        model_path: str,
        enable_fallback: bool = True,
    ):
        self.model_path = Path(model_path)
        self.enable_fallback = enable_fallback
        self.model = None
        self.model_loaded = False
        self.load_failure_kind = ""
        self.load_error = ""
        self.model_id = ""
        try:
            stat = self.model_path.stat()
            self.model_signature = (
                str(self.model_path),
                int(stat.st_mtime_ns),
                int(stat.st_size),
            )
        except OSError:
            self.model_signature = (str(self.model_path), None, None)
        self.model_id = self._resolve_model_id()

        if xgb is None:
            self.load_failure_kind = "xgboost_runtime_unavailable"
            return
        if not self.model_path.exists():
            self.load_failure_kind = "missing_model_file"
            return
        try:
            if self.model_path.stat().st_size <= 0:
                self.load_failure_kind = "empty_model_file"
                return
        except OSError:
            self.load_failure_kind = "missing_model_file"
            return
        if xgb and self.model_path.exists():
            try:
                model = xgb.Booster()
                model.load_model(str(self.model_path))
            except Exception as exc:
                self.load_failure_kind = "model_load_failed"
                self.load_error = str(exc)
                logger.warning(
                    f"Failed to load XGBoost model from {self.model_path}: {exc}"
                )
            else:
                self.model = model
                self.model_loaded = True
                logger.info(f"Loaded XGBoost model from {self.model_path}")

    def predict(self, snapshot: FeatureSnapshot) -> PredictionResult:
        if not snapshot.valid or not snapshot.values:
            return PredictionResult(
                symbol=snapshot.symbol,
                up_probability=0.0,
                feature_count=0,
                model_version="invalid_features",
                model_id="invalid_features",
            )

        if self.model is not None and xgb is not None:
            feature_names, values = self._aligned_features(snapshot)
            matrix = xgb.DMatrix([values], feature_names=feature_names)
            probability = float(self.model.predict(matrix)[0])
            model_version = self.model_path.name
            model_id = self.model_id or model_version
        elif self.enable_fallback:
            probability = self._fallback_probability(snapshot)
            model_version = "fallback_v2"
            model_id = model_version
            feature_names = list(snapshot.values.keys())
        else:
            raise RuntimeError("XGBoost model unavailable and fallback disabled")

        return PredictionResult(
            symbol=snapshot.symbol,
            up_probability=max(0.0, min(1.0, probability)),
            feature_count=len(feature_names),
            model_version=model_version,
            model_id=model_id,
        )

    def _aligned_features(self, snapshot: FeatureSnapshot) -> tuple[list[str], list[float]]:
        feature_map = {
            str(name): float(value)
            for name, value in snapshot.values.items()
        }
        model_feature_names = list(getattr(self.model, "feature_names", []) or [])
        if not model_feature_names:
            return list(feature_map.keys()), list(feature_map.values())
        return model_feature_names, [
            float(feature_map.get(name, 0.0)) for name in model_feature_names
        ]

    def _resolve_model_id(self) -> str:
        meta_path = self.model_path.with_suffix(".meta.json")
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
        model_id = str(metadata.get("model_id") or "").strip()
        if model_id:
            return model_id
        signature = self.model_signature
        if signature[1] is None:
            return self.model_path.name
        return f"{self.model_path.name}@{signature[1]}"

    @staticmethod
    def _fallback_probability(snapshot: FeatureSnapshot) -> float:
        values = snapshot.values
        score = 0.5

        if values["close_1h"] > values["ma20_1h"]:
            score += 0.08
        if values["close_4h"] > values["ma50_4h"]:
            score += 0.10
        if values["close_1d"] > values["ma200_1d"]:
            score += 0.12
        if values["rsi_1h"] < 35:
            score += 0.06
        if values["rsi_1h"] > 70:
            score -= 0.08
        if values["macd_1h"] > 0:
            score += 0.05
        if values["volume_ratio_1h"] > 1.0:
            score += 0.04
        if values["return_24h"] < -0.03:
            score -= 0.08
        score += values["sentiment_value"] * 0.05
        score += values["market_regime_score"] * 0.04
        score -= min(values["volatility_20d"], 0.1) * 0.25

        return score
