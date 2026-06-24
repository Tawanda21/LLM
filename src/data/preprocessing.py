"""Text preprocessing utilities used before tokenization."""

import re
import unicodedata
from typing import Optional


def normalize_unicode(text: str) -> str:
    """NFC-normalize unicode to reduce vocabulary fragmentation.

    NFC merges composed and decomposed forms of the same character
    (e.g. é as one codepoint vs. e + combining accent).
    """
    return unicodedata.normalize("NFC", text)


def remove_control_characters(text: str) -> str:
    """Strip non-printable control characters, preserving newline and tab."""
    return "".join(
        ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t"
    )


def normalize_whitespace(text: str) -> str:
    """Collapse all runs of whitespace (tabs, multiple spaces) to one space."""
    return re.sub(r"[^\S\n]+", " ", text).strip()


def clean(
    text: str,
    do_unicode_norm: bool = True,
    do_remove_control: bool = True,
    min_length: int = 10,
) -> Optional[str]:
    """Apply a standard pre-tokenization cleaning pipeline.

    Args:
        text:               raw input string
        do_unicode_norm:    apply NFC unicode normalisation
        do_remove_control:  strip non-printable control characters
        min_length:         return None for texts shorter than this

    Returns:
        Cleaned string, or None if the text is too short after cleaning.
    """
    if do_unicode_norm:
        text = normalize_unicode(text)
    if do_remove_control:
        text = remove_control_characters(text)
    text = text.strip()
    return text if len(text) >= min_length else None
