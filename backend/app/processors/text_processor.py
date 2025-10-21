"""
Text processor helpers.

This module contains minimal helpers to read plain text and clean it.
It's intentionally small: PDF/DOCX extraction should be added by the
project owner when needed so heavy dependencies are optional.
"""

from typing import Optional


def clean_text(text: Optional[str]) -> str:
	"""Basic cleaning: normalize whitespace and trim."""
	if not text:
		return ""
	# Replace common weird whitespace and collapse multiple spaces/newlines
	cleaned = " ".join(text.split())
	return cleaned.strip()


def read_plain_text(file_bytes: bytes, encoding: str = "utf-8") -> str:
	"""Decode a plain text file's bytes to a string.

	This is a tiny helper to keep upload handlers simple for beginners.
	"""
	try:
		return file_bytes.decode(encoding)
	except Exception:
		# Fall back to latin-1 for legacy files
		return file_bytes.decode("latin-1", errors="ignore")