"""
Backbone Loader — Provides a unified interface for loading SSL audio models.
Supports: Wav2Vec2, HuBERT, WavLM with configurable layer freezing.
"""
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoFeatureExtractor,
    Wav2Vec2Model,
    HubertModel,
    WavLMModel,
)
import logging

logger = logging.getLogger(__name__)

# Registry of supported models and their HF classes
MODEL_REGISTRY = {
    "wav2vec2": {
        "default": "facebook/wav2vec2-large-xlsr-53",
        "class": Wav2Vec2Model,
    },
    "hubert": {
        "default": "facebook/hubert-large-ls960",
        "class": HubertModel,
    },
    "wavlm": {
        "default": "microsoft/wavlm-large",
        "class": WavLMModel,
    },
}


class BackboneLoader:
    """Loads and configures SSL audio backbones with layer freezing."""

    @staticmethod
    def load(model_type: str, model_name: str = None, freeze_layers: int = 0,
             device: str = "cpu") -> tuple:
        """
        Load a backbone model and its feature extractor.

        Args:
            model_type: One of "wav2vec2", "hubert", "wavlm"
            model_name: HuggingFace model ID (uses default if None)
            freeze_layers: Number of initial transformer layers to freeze
            device: Target device

        Returns:
            (model, feature_extractor, hidden_size)
        """
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}. "
                             f"Choose from {list(MODEL_REGISTRY.keys())}")

        reg = MODEL_REGISTRY[model_type]
        name = model_name or reg["default"]

        logger.info(f"Loading backbone: {name} (type={model_type})")

        # Load model and feature extractor
        model = AutoModel.from_pretrained(name)
        feature_extractor = AutoFeatureExtractor.from_pretrained(name)

        # Get hidden size from config
        hidden_size = model.config.hidden_size
        logger.info(f"Hidden size: {hidden_size}")

        # Freeze layers for efficient fine-tuning
        if freeze_layers > 0:
            BackboneLoader._freeze_layers(model, freeze_layers, model_type)

        model = model.to(device)
        return model, feature_extractor, hidden_size

    @staticmethod
    def _freeze_layers(model, num_layers: int, model_type: str):
        """Freeze the feature extractor and first N transformer layers."""
        # Always freeze the CNN feature extractor (low-level features)
        if hasattr(model, "feature_extractor"):
            for param in model.feature_extractor.parameters():
                param.requires_grad = False
            logger.info("  Froze CNN feature extractor")

        if hasattr(model, "feature_projection"):
            for param in model.feature_projection.parameters():
                param.requires_grad = False
            logger.info("  Froze feature projection")

        # Freeze the first N encoder layers
        if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            total_layers = len(model.encoder.layers)
            freeze_count = min(num_layers, total_layers)
            for i in range(freeze_count):
                for param in model.encoder.layers[i].parameters():
                    param.requires_grad = False
            logger.info(f"  Froze {freeze_count}/{total_layers} transformer layers")

        # Count trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"  Trainable params: {trainable:,} / {total:,} "
                    f"({trainable/total*100:.1f}%)")
