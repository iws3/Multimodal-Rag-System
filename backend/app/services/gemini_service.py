"""
Lightweight Gemini service stub.

This file provides a simple interface `generate_text` that currently acts as a local stub
when no GEMINI_API_KEY is configured. Replace the stub implementation with a real
HTTP client (or official Google client) when you have API credentials.
"""
import os
from typing import Optional

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def generate_text(prompt: str, model: Optional[str] = None) -> str:
    """Generate text using Gemini (stub).

    - If GEMINI_API_KEY is not set, returns a deterministic stub response for development.
    - If GEMINI_API_KEY is set, this function currently raises NotImplementedError and
      should be implemented to call the real Gemini API (to avoid shipping credentials
      or hard-coding provider-specific SDKs in this repo).

    Replace the body with your preferred client (google.generativeai, httpx request,
    or SDK) per your environment.
    """
    if not GEMINI_API_KEY:
        # Local stubbed response for dev and tests
        return f"[gemini-stub] Echo: {prompt[:500]}"

    # Placeholder: at runtime, developer can implement the actual API call.
    raise NotImplementedError(
        "GEMINI_API_KEY is set but the Gemini HTTP client is not implemented. "
        "Please implement API calls in backend/app/services/gemini_service.py or unset GEMINI_API_KEY to use the local stub."
    )
