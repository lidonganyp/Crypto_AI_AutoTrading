"""Lightweight language helpers for active user-facing output."""
from __future__ import annotations

from config import Settings, get_settings


LANGUAGE_STATE_KEY = "app_language"


def normalize_language(value: str | None) -> str:
    text = (value or "").strip().lower()
    if text.startswith("en"):
        return "en"
    return "zh"


def get_default_language(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    return normalize_language(getattr(settings.app, "language", "zh"))


def get_runtime_language(storage=None, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    if storage is not None and hasattr(storage, "get_state"):
        value = storage.get_state(LANGUAGE_STATE_KEY)
        if value:
            return normalize_language(value)
    return get_default_language(settings)


def text_for(lang: str, zh: str, en: str) -> str:
    return zh if normalize_language(lang) == "zh" else en
