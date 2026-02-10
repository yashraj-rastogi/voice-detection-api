"""
Classification Heads — Pooling strategies and MLP classifier for deepfake detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentiveStatsPooling(nn.Module):
    """
    Attentive Statistics Pooling.
    Learns which frames are most important, then computes weighted mean + std.
    Used in ECAPA-TDNN and top speaker verification systems.
    """

    def __init__(self, hidden_size: int, attention_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )
        self.output_size = hidden_size * 2  # mean + std concatenated

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, time, hidden_size)
            mask: optional (batch, time) boolean mask

        Returns:
            (batch, hidden_size * 2) — weighted mean and std
        """
        # Compute attention weights
        attn_weights = self.attention(x).squeeze(-1)  # (batch, time)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1).unsqueeze(-1)  # (batch, time, 1)

        # Weighted mean
        mean = torch.sum(x * attn_weights, dim=1)  # (batch, hidden)

        # Weighted std
        var = torch.sum(attn_weights * (x - mean.unsqueeze(1)) ** 2, dim=1)
        std = torch.sqrt(var.clamp(min=1e-6))

        return torch.cat([mean, std], dim=-1)  # (batch, hidden*2)


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-Head Attention Pooling.
    Applies multi-head self-attention then pools via learned query vector.
    """

    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.mha = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.output_size = hidden_size

        nn.init.xavier_uniform_(self.query)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, time, hidden_size)
        Returns:
            (batch, hidden_size)
        """
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)  # (batch, 1, hidden)
        out, _ = self.mha(query, x, x)  # (batch, 1, hidden)
        return out.squeeze(1)  # (batch, hidden)


class MeanPooling(nn.Module):
    """Simple mean pooling over the time axis."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.output_size = hidden_size

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
            return x.sum(dim=1) / mask.sum(dim=1, keepdim=True).float()
        return x.mean(dim=1)


class DeepfakeClassifier(nn.Module):
    """
    Full classification model = Backbone + Pooling + MLP Head.
    """

    def __init__(self, backbone: nn.Module, hidden_size: int,
                 num_labels: int = 2, classifier_hidden: int = 256,
                 dropout: float = 0.3, pooling_type: str = "attentive_stats"):
        super().__init__()
        self.backbone = backbone

        # Select pooling strategy
        if pooling_type == "attentive_stats":
            self.pooling = AttentiveStatsPooling(hidden_size)
        elif pooling_type == "multi_head":
            self.pooling = MultiHeadAttentionPooling(hidden_size)
        elif pooling_type == "mean":
            self.pooling = MeanPooling(hidden_size)
        else:
            raise ValueError(f"Unknown pooling: {pooling_type}")

        pool_output_size = self.pooling.output_size

        # MLP classification head with batch norm
        self.classifier = nn.Sequential(
            nn.Linear(pool_output_size, classifier_hidden),
            nn.BatchNorm1d(classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, classifier_hidden // 2),
            nn.BatchNorm1d(classifier_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(classifier_hidden // 2, num_labels),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, input_values: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            input_values: (batch, time) raw waveform
            attention_mask: (batch, time) attention mask

        Returns:
            logits: (batch, num_labels)
        """
        # Extract features from backbone
        outputs = self.backbone(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)

        # Pool across time
        pooled = self.pooling(hidden_states)  # (batch, pool_dim)

        # Classify
        logits = self.classifier(pooled)  # (batch, num_labels)
        return logits

    def extract_embeddings(self, input_values: torch.Tensor) -> torch.Tensor:
        """Extract embeddings (before classification head) for analysis."""
        outputs = self.backbone(input_values)
        hidden_states = outputs.last_hidden_state
        return self.pooling(hidden_states)
