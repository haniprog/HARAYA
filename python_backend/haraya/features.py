from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .preprocessing import preprocess_text


# Layer 4: Behavioral Feature Extraction
# Dito kinukuha ang mga extra signal bukod sa BERT.
POSITIVE_WORDS = {
    "good",
    "great",
    "thanks",
    "thank",
    "okay",
    "ok",
    "nice",
    "hello",
    "love",
    "safe",
}

NEGATIVE_WORDS = {
    "hate",
    "ugly",
    "stupid",
    "idiot",
    "trash",
    "fuck",
    "fucking",
    "shit",
    "bastard",
    "asshole",
    "kill",
    "annoying",
    "useless",
    "bitch",
    "shut",
}


@dataclass(frozen=True)
class BehavioralFeatures:
    frequency: float
    repetition: float
    sentiment: float


def extract_message_frequency(messages: Iterable[str]) -> float:
    # Bilang ng messages sa conversation.
    return float(len([message for message in messages if message]))


def count_repetition(messages: Iterable[str]) -> float:
    #  kung ilang salita ang paulit-ulit sa usapan.
    tokens: list[str] = []
    for message in messages:
        cleaned = preprocess_text(message)
        tokens.extend(token for token in cleaned.split() if len(token) > 2)

    counts = Counter(tokens)
    repeated = sum(max(0, count - 1) for count in counts.values())
    return float(repeated)


def analyze_sentiment(messages: Iterable[str]) -> float:
    #  sentiment score: positive minus negative words.
    tokens: list[str] = []
    for message in messages:
        cleaned = preprocess_text(message)
        tokens.extend(cleaned.split())

    if not tokens:
        return 0.0

    positive = sum(1 for token in tokens if token in POSITIVE_WORDS)
    negative = sum(1 for token in tokens if token in NEGATIVE_WORDS)

    if positive == 0 and negative == 0:
        return 0.0

    return float(np.clip((positive - negative) / (positive + negative), -1.0, 1.0))


def build_behavioral_features(messages: Iterable[str]) -> BehavioralFeatures:
    # Pinagsasama ang frequency, repetition, at sentiment.
    message_list = [message for message in messages if message]
    return BehavioralFeatures(
        frequency=extract_message_frequency(message_list),
        repetition=count_repetition(message_list),
        sentiment=analyze_sentiment(message_list),
    )


def to_feature_vector(features: BehavioralFeatures) -> np.ndarray:
    # Ginagawang numeric vector para maidugtong sa BERT output.
    return np.array([features.frequency, features.repetition, features.sentiment], dtype=np.float32)
