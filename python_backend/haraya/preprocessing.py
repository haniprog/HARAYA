from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


TOKEN_PATTERN = re.compile(r"[^a-z0-9\s]")
SPACE_PATTERN = re.compile(r"\s+")


def preprocess_text(text: str) -> str:
    # Linisin ang text: lowercase, remove the symbols, at ayusin ang spaces.
    cleaned = text.lower()
    cleaned = TOKEN_PATTERN.sub(" ", cleaned)
    cleaned = SPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


@dataclass(frozen=True)
class PreprocessedConversation:
    original_messages: list[str]
    cleaned_messages: list[str]
    context: str


def preprocess_conversation(messages: Iterable[str]) -> PreprocessedConversation:
    # I-prepare ang buong usapan para maging isang context string.
    original_messages = [message for message in messages if message]
    cleaned_messages = [preprocess_text(message) for message in original_messages]
    # Ang [SEP] ay ginagamit para maipakita ang pagitan ng bawat message.
    context = " [SEP] ".join(message for message in cleaned_messages if message)
    return PreprocessedConversation(
        original_messages=original_messages,
        cleaned_messages=cleaned_messages,
        context=context,
    )
