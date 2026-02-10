"""
Training Engine — Research-grade training loop for deepfake audio detection.
Features: Focal Loss, Cosine Annealing, Mixed Precision, EMA, full logging.
"""
import os
import sys
import copy
import yaml
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import librosa
import logging
import argparse
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, LinearLR, SequentialLR

from augment import AugmentationPipeline, AugConfig
from models.backbone import BackboneLoader
from models.classifier import DeepfakeClassifier
from models.ensemble import EnsembleDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ===============================================================
#  Focal Loss (handles class imbalance better than cross-entropy)
# ===============================================================

class FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced classification."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()

        pt = (probs * targets_one_hot).sum(dim=-1)
        focal_weight = (1 - pt) ** self.gamma

        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        loss = self.alpha * focal_weight * ce_loss
        return loss.mean()


# ===============================================================
#  EMA (Exponential Moving Average) for smoother final weights
# ===============================================================

class EMAModel:
    """Maintains an exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self, model: nn.Module):
        """Replace model params with EMA params."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        """Restore original params."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# ===============================================================
#  Audio Dataset
# ===============================================================

class AudioDataset(Dataset):
    """Loads audio files and applies augmentations on the fly."""

    def __init__(self, metadata_csv: str, cfg: dict,
                 augment: bool = False, max_samples: int = None):
        self.df = pd.read_csv(metadata_csv)
        if max_samples:
            self.df = self.df.head(max_samples)
        self.sr = cfg["data"]["sample_rate"]
        self.max_len = int(cfg["data"]["max_duration_sec"] * self.sr)
        self.augmenter = None
        if augment and cfg["augmentation"]["enabled"]:
            self.augmenter = AugmentationPipeline(
                AugConfig.from_dict(cfg), sr=self.sr
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["file"]
        label = int(row["label"])

        # Load audio
        try:
            y, _ = librosa.load(filepath, sr=self.sr, mono=True)
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}. Returning silence.")
            y = np.zeros(self.sr, dtype=np.float32)

        # Apply augmentations (training only)
        if self.augmenter is not None:
            y = self.augmenter(y)

        # Pad or truncate to fixed length
        if len(y) > self.max_len:
            y = y[:self.max_len]
        elif len(y) < self.max_len:
            y = np.pad(y, (0, self.max_len - len(y)), mode="constant")

        # Normalize
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak

        return {
            "input_values": torch.tensor(y, dtype=torch.float32),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ===============================================================
#  Training Loop
# ===============================================================

def set_seed(seed: int):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def build_model(cfg: dict, device: str):
    """Build model from config."""
    model_cfg = cfg["model"]
    mode = model_cfg["mode"]

    if mode == "ensemble":
        # Build all three backbones
        models = []
        for backbone_type in ["wav2vec2", "hubert", "wavlm"]:
            bb_cfg = model_cfg["backbones"][backbone_type]
            backbone, feat_ext, hidden = BackboneLoader.load(
                backbone_type, bb_cfg["name"], bb_cfg["freeze_layers"], device
            )
            clf = DeepfakeClassifier(
                backbone, hidden,
                num_labels=2,
                classifier_hidden=model_cfg["classifier"]["hidden_dim"],
                dropout=model_cfg["classifier"]["dropout"],
                pooling_type=model_cfg["classifier"]["pooling"],
            ).to(device)
            models.append(clf)

        ensemble = EnsembleDetector.create(
            models,
            strategy=model_cfg["ensemble"]["strategy"],
            weights=model_cfg["ensemble"].get("weights"),
        ).to(device)
        return ensemble, feat_ext  # Use last feature extractor

    else:
        # Single backbone
        bb_cfg = model_cfg["backbones"][mode]
        backbone, feat_ext, hidden = BackboneLoader.load(
            mode, bb_cfg["name"], bb_cfg["freeze_layers"], device
        )
        model = DeepfakeClassifier(
            backbone, hidden,
            num_labels=2,
            classifier_hidden=model_cfg["classifier"]["hidden_dim"],
            dropout=model_cfg["classifier"]["dropout"],
            pooling_type=model_cfg["classifier"]["pooling"],
        ).to(device)
        return model, feat_ext


def evaluate(model, dataloader, criterion, device):
    """Run evaluation and return loss + accuracy."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # P(AI_GENERATED)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)

    # Compute EER
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "eer": eer,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }


def train(cfg: dict):
    """Main training function."""
    train_cfg = cfg["training"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🖥️  Device: {device}")

    set_seed(train_cfg["seed"])

    # Paths
    metadata_dir = os.path.join(cfg["paths"]["output_dir"], "metadata")
    train_csv = os.path.join(metadata_dir, "train.csv")
    val_csv = os.path.join(metadata_dir, "val.csv")

    if not os.path.exists(train_csv):
        logger.error("❌ Training metadata not found. Run prepare_data.py first!")
        return

    # Datasets
    logger.info("📂 Loading datasets...")
    train_dataset = AudioDataset(train_csv, cfg, augment=True)
    val_dataset = AudioDataset(val_csv, cfg, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg["batch_size"],
        shuffle=True, num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg["batch_size"],
        shuffle=False, num_workers=0, pin_memory=True,
    )

    logger.info(f"   Train samples: {len(train_dataset)}")
    logger.info(f"   Val samples:   {len(val_dataset)}")

    # Model
    logger.info("🏗️  Building model...")
    model, feature_extractor = build_model(cfg, device)

    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"   Parameters: {trainable:,} trainable / {total:,} total")

    # Loss
    if train_cfg["loss"] == "focal":
        criterion = FocalLoss(
            gamma=train_cfg["focal_gamma"],
            alpha=train_cfg["focal_alpha"],
        )
        logger.info("   Loss: Focal Loss")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("   Loss: Cross Entropy")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    # Scheduler
    total_steps = len(train_loader) * train_cfg["num_epochs"]
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)

    if train_cfg["lr_scheduler"] == "cosine_with_restarts":
        main_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader), T_mult=2)
    else:
        main_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)

    scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[warmup_steps])

    # Mixed precision
    use_fp16 = train_cfg["fp16"] and device == "cuda"
    scaler = GradScaler() if use_fp16 else None
    logger.info(f"   Mixed precision: {'ON' if use_fp16 else 'OFF'}")

    # EMA
    ema = None
    if train_cfg["ema"]["enabled"]:
        ema = EMAModel(model, decay=train_cfg["ema"]["decay"])
        logger.info(f"   EMA: ON (decay={train_cfg['ema']['decay']})")

    # Training state
    output_dir = cfg["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    best_metric = float("inf") if train_cfg["metric_for_best_model"] == "eer" else 0
    patience_counter = 0
    history = []

    grad_accum = train_cfg["gradient_accumulation_steps"]

    # ============ Training Loop ============
    logger.info("=" * 60)
    logger.info("  🔥 TRAINING STARTED")
    logger.info("=" * 60)

    for epoch in range(train_cfg["num_epochs"]):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0

        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            if use_fp16:
                with autocast():
                    logits = model(inputs)
                    loss = criterion(logits, labels) / grad_accum
                scaler.scale(loss).backward()

                if (step + 1) % grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                logits = model(inputs)
                loss = criterion(logits, labels) / grad_accum
                loss.backward()

                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            # EMA update
            if ema is not None:
                ema.update(model)

            # Track metrics
            epoch_loss += loss.item() * grad_accum * labels.size(0)
            preds = logits.argmax(dim=-1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += labels.size(0)

        # Epoch summary
        train_loss = epoch_loss / max(epoch_total, 1)
        train_acc = epoch_correct / max(epoch_total, 1)

        # Validation
        if ema is not None:
            ema.apply_shadow(model)

        val_metrics = evaluate(model, val_loader, criterion, device)

        if ema is not None:
            ema.restore(model)

        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch+1}/{train_cfg['num_epochs']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
            f"EER: {val_metrics['eer']:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_eer": val_metrics["eer"],
            "lr": current_lr,
        })

        # Best model check
        metric_key = train_cfg["metric_for_best_model"]
        current_val = val_metrics.get(metric_key, val_metrics["accuracy"])
        is_better = (current_val < best_metric) if metric_key == "eer" else (current_val > best_metric)

        if is_better:
            best_metric = current_val
            patience_counter = 0
            # Save best model
            best_path = os.path.join(output_dir, "best_model")
            os.makedirs(best_path, exist_ok=True)

            if ema is not None:
                ema.apply_shadow(model)

            torch.save(model.state_dict(), os.path.join(best_path, "model.pt"))

            if ema is not None:
                ema.restore(model)

            logger.info(f"   ✅ New best! {metric_key}={current_val:.4f} → saved to {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= train_cfg["early_stopping_patience"]:
                logger.info(f"   ⏹️  Early stopping after {patience_counter} epochs without improvement")
                break

    # Save training history
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    logger.info("=" * 60)
    logger.info("  🎉 TRAINING COMPLETE!")
    logger.info(f"  Best {train_cfg['metric_for_best_model']}: {best_metric:.4f}")
    logger.info(f"  Model saved to: {os.path.join(output_dir, 'best_model')}")
    logger.info("=" * 60)


# ===============================================================
#  Main Entry Point
# ===============================================================

def main():
    parser = argparse.ArgumentParser(description="World-Class Deepfake Audio Detection Trainer")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Limit training for testing (overrides epochs)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.max_steps:
        cfg["training"]["num_epochs"] = 1
        logger.info(f"⚠️  Debug mode: limited to {args.max_steps} steps")

    train(cfg)


if __name__ == "__main__":
    main()
