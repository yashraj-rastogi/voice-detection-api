"""
Ensemble Detector — Combines multiple backbone models for superior detection.
Supports late fusion, learned fusion, and confidence-weighted strategies.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class LateFusionEnsemble(nn.Module):
    """
    Late Fusion: Average (or weighted average) of per-model probabilities.
    Simple, effective, and robust.
    """

    def __init__(self, models: List[nn.Module], weights: List[float] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        assert len(weights) == len(models)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def forward(self, input_values: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        all_probs = []
        for model in self.models:
            logits = model(input_values, attention_mask)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs)

        # Stack and compute weighted average
        stacked = torch.stack(all_probs, dim=0)  # (num_models, batch, num_labels)
        weights = self.weights.to(stacked.device).view(-1, 1, 1)
        fused = (stacked * weights).sum(dim=0)  # (batch, num_labels)

        # Return log probabilities (compatible with loss functions)
        return torch.log(fused + 1e-10)


class LearnedFusionEnsemble(nn.Module):
    """
    Learned Fusion: Small MLP trained on concatenated model outputs.
    More expressive than late fusion but requires end-to-end training.
    """

    def __init__(self, models: List[nn.Module], num_labels: int = 2):
        super().__init__()
        self.models = nn.ModuleList(models)
        total_labels = num_labels * len(models)

        self.fusion_head = nn.Sequential(
            nn.Linear(total_labels, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_labels),
        )

    def forward(self, input_values: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        all_logits = []
        for model in self.models:
            logits = model(input_values, attention_mask)
            all_logits.append(logits)

        concatenated = torch.cat(all_logits, dim=-1)  # (batch, num_labels * num_models)
        return self.fusion_head(concatenated)


class ConfidenceWeightedEnsemble(nn.Module):
    """
    Confidence-Weighted: Each model's vote is weighted by how confident it is.
    Models that are uncertain contribute less to the final prediction.
    """

    def __init__(self, models: List[nn.Module], temperature: float = 1.0):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.temperature = temperature

    def forward(self, input_values: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        all_probs = []
        confidences = []

        for model in self.models:
            logits = model(input_values, attention_mask)
            probs = F.softmax(logits / self.temperature, dim=-1)
            confidence = probs.max(dim=-1)[0]  # Max probability as confidence
            all_probs.append(probs)
            confidences.append(confidence)

        # Normalize confidences to sum to 1
        conf_stack = torch.stack(confidences, dim=0)  # (num_models, batch)
        conf_weights = F.softmax(conf_stack, dim=0).unsqueeze(-1)  # (num_models, batch, 1)

        prob_stack = torch.stack(all_probs, dim=0)  # (num_models, batch, num_labels)
        fused = (prob_stack * conf_weights).sum(dim=0)  # (batch, num_labels)

        return torch.log(fused + 1e-10)


class EnsembleDetector:
    """Factory for creating ensemble models from config."""

    @staticmethod
    def create(models: List[nn.Module], strategy: str = "late_fusion",
               weights: List[float] = None, num_labels: int = 2) -> nn.Module:
        """
        Create an ensemble from a list of individual models.

        Args:
            models: List of DeepfakeClassifier instances
            strategy: "late_fusion", "learned_fusion", or "confidence_weighted"
            weights: Optional weights for late_fusion
            num_labels: Number of output classes

        Returns:
            Ensemble nn.Module
        """
        if strategy == "late_fusion":
            logger.info(f"Creating Late Fusion Ensemble ({len(models)} models)")
            return LateFusionEnsemble(models, weights)
        elif strategy == "learned_fusion":
            logger.info(f"Creating Learned Fusion Ensemble ({len(models)} models)")
            return LearnedFusionEnsemble(models, num_labels)
        elif strategy == "confidence_weighted":
            logger.info(f"Creating Confidence-Weighted Ensemble ({len(models)} models)")
            return ConfidenceWeightedEnsemble(models)
        else:
            raise ValueError(f"Unknown ensemble strategy: {strategy}")
