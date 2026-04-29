"""Strategy profile helpers shared across entry and position management."""
from __future__ import annotations

from dataclasses import dataclass


OPEN_BIAS_REASONS = {
    "extreme_fear_offensive_setup",
    "extreme_fear_quant_override",
}
ENTRY_THESIS_OPEN_BIAS = {
    "high_conviction_long",
    "extreme_fear_reversal",
    "fast_alpha_short_horizon",
}
ENTRY_THESIS_SET = {
    "paper_canary_primary",
    "paper_canary_offensive",
    "paper_canary_soft",
    "high_conviction_long",
    "extreme_fear_reversal",
    "fast_alpha_short_horizon",
    "generic_long",
}
STRONG_ENTRY_QUALITY_REASONS = {
    "extreme_fear_offensive_setup",
    "extreme_fear_quant_override",
}
THESIS_REASON_EXTREME_FEAR = {
    "extreme_fear_offensive_setup",
    "extreme_fear_offensive_open",
    "extreme_fear_quant_override",
    "extreme_fear_quant_override_open",
    "quant_repairing_setup",
    "quant_repairing_setup_open",
}
THESIS_REASON_HIGH_CONVICTION = {
    "xgb_strong",
    "liquidity_supportive",
    "trend_supportive",
}


@dataclass(frozen=True)
class StrategyProfile:
    pipeline_mode: str
    canary_mode: str
    entry_thesis: str
    entry_thesis_strength: str
    entry_open_bias: bool
    has_entry_thesis: bool
    is_current_open_bias: bool
    strong_entry_quality: bool


def _normalize_reasons(reasons: list[str] | set[str] | tuple[str, ...] | None) -> set[str]:
    return {
        str(reason).strip()
        for reason in (reasons or [])
        if str(reason).strip()
    }


def derive_entry_profile(
    *,
    pipeline_mode: str,
    raw_action: str,
    reviewed_action: str,
    review_score: float,
    review_reasons: list[str] | set[str] | tuple[str, ...] | None,
) -> StrategyProfile:
    reasons = _normalize_reasons(review_reasons)
    normalized_pipeline_mode = str(pipeline_mode or "").strip()
    normalized_raw_action = str(raw_action or "").upper()
    normalized_reviewed_action = str(reviewed_action or "").upper()

    canary_mode = ""
    strength = "weak"
    thesis = "generic_long"
    if normalized_pipeline_mode == "fast_alpha":
        thesis = "fast_alpha_short_horizon"
        strength = "strong" if float(review_score) >= 0.20 else "moderate"
    elif normalized_pipeline_mode == "paper_canary":
        canary_mode = "soft"
        if "offensive_review" in reasons:
            canary_mode = "offensive"
        elif normalized_reviewed_action == "OPEN_LONG":
            canary_mode = "primary"
        thesis = f"paper_canary_{canary_mode}"
    elif reasons & THESIS_REASON_EXTREME_FEAR:
        thesis = "extreme_fear_reversal"
    elif reasons & THESIS_REASON_HIGH_CONVICTION:
        thesis = "high_conviction_long"

    if thesis in {"paper_canary_primary", "paper_canary_offensive"}:
        strength = "strong"
    elif thesis == "paper_canary_soft":
        strength = "moderate"
    elif (
        normalized_reviewed_action == "OPEN_LONG"
        or normalized_raw_action == "OPEN_LONG"
        or thesis in {"high_conviction_long", "extreme_fear_reversal"}
    ):
        strength = "strong" if float(review_score) >= 0.20 else "moderate"

    entry_open_bias = bool(
        normalized_reviewed_action == "OPEN_LONG"
        or normalized_raw_action == "OPEN_LONG"
        or thesis in ENTRY_THESIS_OPEN_BIAS
    )
    return StrategyProfile(
        pipeline_mode=normalized_pipeline_mode,
        canary_mode=canary_mode,
        entry_thesis=thesis,
        entry_thesis_strength=strength,
        entry_open_bias=entry_open_bias,
        has_entry_thesis=entry_open_bias or thesis in ENTRY_THESIS_SET,
        is_current_open_bias=normalized_raw_action == "OPEN_LONG" or bool(reasons & OPEN_BIAS_REASONS),
        strong_entry_quality=bool(
            canary_mode in {"primary", "offensive"}
            or normalized_reviewed_action == "OPEN_LONG"
            or (
                normalized_raw_action == "OPEN_LONG"
                and float(review_score) >= 0.15
            )
            or bool(reasons & STRONG_ENTRY_QUALITY_REASONS)
            or strength == "strong"
        ),
    )


def profile_from_trade_metadata(
    metadata: dict | None,
    *,
    current_raw_action: str = "",
    current_review_reasons: list[str] | set[str] | tuple[str, ...] | None = None,
) -> StrategyProfile:
    payload = dict(metadata or {})
    reasons = _normalize_reasons(payload.get("review_reasons") or [])
    review_score = float(payload.get("review_score") or 0.0)
    base = derive_entry_profile(
        pipeline_mode=str(payload.get("pipeline_mode") or "").strip(),
        raw_action=str(payload.get("raw_action") or "").upper(),
        reviewed_action=str(payload.get("reviewed_action") or "").upper(),
        review_score=review_score,
        review_reasons=list(reasons),
    )
    current_reasons = _normalize_reasons(current_review_reasons)
    is_current_open_bias = (
        str(current_raw_action or "").upper() == "OPEN_LONG"
        or bool(current_reasons & OPEN_BIAS_REASONS)
    )
    return StrategyProfile(
        pipeline_mode=base.pipeline_mode,
        canary_mode=base.canary_mode,
        entry_thesis=base.entry_thesis,
        entry_thesis_strength=base.entry_thesis_strength,
        entry_open_bias=base.entry_open_bias,
        has_entry_thesis=base.has_entry_thesis,
        is_current_open_bias=is_current_open_bias,
        strong_entry_quality=base.strong_entry_quality,
    )
