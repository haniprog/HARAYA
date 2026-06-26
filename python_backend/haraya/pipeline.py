from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

from .config import CONFIG
from .features import build_behavioral_features, to_feature_vector
from .model import load_model, load_tokenizer, softmax_probabilities
from .preprocessing import preprocess_conversation


# Layer 6: Classification Output
# Ito ang kabuuang proseso mula input hanggang final label.
@dataclass(frozen=True)
class HarayaResponse:
    conversation: str
    label: str
    finalScore: float
    confidence: float
    features: dict
    reasons: list[str]
    legalBasis: list[str]
    recommendations: list[str]


def _rule_based_label(behavioral: dict, text: str) -> tuple[int, float]:
    # Fallback guide kapag wala pang trained weights ang model.
    lowered = text.lower()

    harassment_cues = (
        "fuck",
        "fucking",
        "shit",
        "bastard",
        "asshole",
        "idiot",
        "stupid",
        "ugly",
        "bitch",
        "kill",
    )
    potential_cues = (
        "leave me alone",
        "stop messaging",
        "why are you ignoring",
        "reply now",
        "call me",
    )

    if any(cue in lowered for cue in harassment_cues):
        return 2, 0.95

    if any(cue in lowered for cue in potential_cues):
        return 1, 0.72

    if behavioral["frequency"] >= 4 or behavioral["repetition"] >= 2 or behavioral["sentiment"] <= -0.25:
        return 1, 0.58

    return 0, 0.88


def _risk_score_from_label(label_index: int, score: float) -> float:
    # I-map ang class output sa risk score na mas malinaw para sa UI.
    if label_index == 0:
        return float(np.clip(1.0 - score, 0.0, 0.35))

    if label_index == 1:
        return float(np.clip(0.4 + (score * 0.25), 0.4, 0.69))

    return float(np.clip(0.7 + (score * 0.3), 0.7, 0.99))


def _build_reasons(label: str, confidence: float, features: dict) -> list[str]:
    # Mga paliwanag na ipapakita sa user kung bakit ganoon ang label.
    reasons: list[str] = []

    if features["behavioral"]["frequency"] >= 4:
        reasons.append("Multiple messages suggest escalating interaction pressure.")

    if features["behavioral"]["repetition"] >= 2:
        reasons.append("Repeated wording was detected across the conversation.")

    if features["behavioral"]["sentiment"] <= -0.2:
        reasons.append("Conversation tone is negative.")

    if confidence >= 0.8:
        reasons.append("The classifier is highly confident in this output.")

    if not reasons:
        reasons.append(f"Model output for {label} is based on fused BERT and behavioral signals.")

    return reasons


def _build_legal_basis(label: str) -> list[str]:
    if label == "Safe Interaction":
        return [
            "Safe Spaces Act (RA 11313): the interaction does not appear unwelcome, intimidating, intrusive, or humiliating.",
            "Revised Penal Code / unjust vexation: no persistent annoyance, intimidation, or distressing conduct is evident.",
            "No indicators of psychological violence under RA 9262 are present in the current sequence.",
        ]

    if label == "Potential Harassment":
        return [
            "RA 9262 may apply where repeated unwanted attention or intimidation suggests psychological violence.",
            "Revised Penal Code / unjust vexation may apply if the conduct irritates, annoys, or distresses the user.",
            "Safe Spaces Act (RA 11313) may become relevant if intrusive behavior continues or intensifies.",
        ]

    return [
        "Safe Spaces Act (RA 11313): unwanted remarks, intimidation, and intrusive conduct may constitute harassment.",
        "RA 7877 may apply when conduct creates an intimidating, hostile, or offensive environment.",
        "RA 9262 may apply to threats, severe psychological abuse, or related violence against women.",
    ]


def _build_recommendations(label: str) -> list[str]:
    if label == "Harassment":
        return [
            "Move to a safer location and avoid direct confrontation.",
            "Contact trusted support or nearby authorities immediately.",
            "Preserve chat records or details for documentation.",
        ]

    if label == "Potential Harassment":
        return [
            "Set clear boundaries and avoid continued engagement.",
            "Stay alert for escalation in tone or behavior.",
            "Document key messages if the pattern continues.",
        ]

    return [
        "No immediate threat pattern detected.",
        "Continue monitoring future messages for changes in tone.",
        "Trust your instincts if the interaction becomes uncomfortable.",
    ]


class HarayaPipeline:
    def __init__(self, model_path: Optional[Path] = None):
        # tokenizer at model kapag nagsisimula ang API.
        self.tokenizer = load_tokenizer()
        self.model = load_model(weights_path=model_path)

    # Layer 6A: Apply preprocessing and feature fusion before inference.
    def predict(self, messages: Iterable[str]) -> HarayaResponse:
        # 1) linisin ang messages, 2) kunin ang behavioral features, 3) run model.
        message_list = [message for message in messages if message]
        preprocessed = preprocess_conversation(message_list)
        behavioral = build_behavioral_features(message_list)
        feature_vector = to_feature_vector(behavioral)

        tokenized = self.tokenizer(
            preprocessed.context or " ",
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=CONFIG.max_length,
        )

        with torch.no_grad():
            logits = self.model(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                behavioral_features=torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0),
            )

        probabilities = softmax_probabilities(logits)
        model_label_index = int(np.argmax(probabilities))
        model_label = CONFIG.label_names[model_label_index]
        model_score = float(probabilities[model_label_index])

        # Kung trained na ang weights, model ang mas sinusunod; kung hindi, may deterministic fallback.
        rule_label_index, rule_score = _rule_based_label(
            {
                "frequency": behavioral.frequency,
                "repetition": behavioral.repetition,
                "sentiment": behavioral.sentiment,
            },
            preprocessed.context,
        )

        if getattr(self.model, "has_trained_weights", False):
            label_index = model_label_index
            score = model_score
            confidence = float(np.clip(0.55 + score * 0.4, 0.5, 0.98))
        else:
            # Blend the untrained model with a deterministic language prior so startup predictions remain sane.
            label_index = rule_label_index
            score = float(np.clip((model_score * 0.35) + (rule_score * 0.65), 0, 1))
            confidence = float(np.clip(0.55 + score * 0.4, 0.5, 0.98))

        label = CONFIG.label_names[label_index]
        risk_score = _risk_score_from_label(label_index, score)

        features = {
            "bert": {
                "context": preprocessed.context,
                "embeddingSource": CONFIG.model_name,
                "modelStatus": "trained" if getattr(self.model, "has_trained_weights", False) else "untrained_fallback",
            },
            "behavioral": {
                "frequency": behavioral.frequency,
                "repetition": behavioral.repetition,
                "sentiment": behavioral.sentiment,
            },
            "probabilities": {
                CONFIG.label_names[0]: float(probabilities[0]),
                CONFIG.label_names[1]: float(probabilities[1]),
                CONFIG.label_names[2]: float(probabilities[2]),
            },
            "decision": {
                "modelLabel": model_label,
                "modelScore": round(model_score, 3),
                "ruleLabel": CONFIG.label_names[rule_label_index],
                "ruleScore": round(rule_score, 3),
            },
        }

        # Final package ng result na ibabalik sa frontend.
        return HarayaResponse(
            conversation=preprocessed.context,
            label=label,
            finalScore=round(risk_score, 3),
            confidence=round(confidence, 2),
            features=features,
            reasons=_build_reasons(label, confidence, features),
            legalBasis=_build_legal_basis(label),
            recommendations=_build_recommendations(label),
        )
