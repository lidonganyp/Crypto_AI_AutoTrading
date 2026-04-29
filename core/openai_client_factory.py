"""Helpers for constructing OpenAI clients with stable local networking behavior."""
from __future__ import annotations

import httpx
from openai import OpenAI


def build_openai_client(
    *,
    api_key: str,
    base_url: str,
    timeout_seconds: float = 18.0,
    connect_timeout_seconds: float = 5.0,
) -> OpenAI:
    # Force LLM traffic to ignore host-level proxy env vars such as
    # HTTP_PROXY / HTTPS_PROXY / ALL_PROXY. This keeps local SOCKS proxy
    # settings from breaking client initialization or redirecting requests.
    http_client = httpx.Client(
        trust_env=False,
        follow_redirects=True,
        timeout=httpx.Timeout(
            max(1.0, float(timeout_seconds)),
            connect=max(1.0, float(connect_timeout_seconds)),
        ),
    )
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
        max_retries=0,
    )
