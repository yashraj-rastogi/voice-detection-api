"""
Export Engine — Convert trained models to production-ready formats.
Supports: HuggingFace Hub, ONNX, TorchScript, and auto-integration with the API.
"""
import os
import json
import yaml
import torch
import torch.nn as nn
import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def export_huggingface(model, cfg, export_dir):
    """Export model as a HuggingFace-compatible checkpoint."""
    hf_dir = os.path.join(export_dir, "huggingface")
    os.makedirs(hf_dir, exist_ok=True)

    # Save model state dict
    torch.save(model.state_dict(), os.path.join(hf_dir, "pytorch_model.bin"))

    # Save config
    model_config = {
        "model_type": cfg["model"]["mode"],
        "num_labels": 2,
        "id2label": {0: "HUMAN", 1: "AI_GENERATED"},
        "label2id": {"HUMAN": 0, "AI_GENERATED": 1},
        "classifier_hidden_dim": cfg["model"]["classifier"]["hidden_dim"],
        "pooling_type": cfg["model"]["classifier"]["pooling"],
        "dropout": cfg["model"]["classifier"]["dropout"],
        "sample_rate": cfg["data"]["sample_rate"],
        "max_duration_sec": cfg["data"]["max_duration_sec"],
    }

    # Add backbone info
    mode = cfg["model"]["mode"]
    if mode != "ensemble":
        model_config["backbone_name"] = cfg["model"]["backbones"][mode]["name"]
        model_config["freeze_layers"] = cfg["model"]["backbones"][mode]["freeze_layers"]

    with open(os.path.join(hf_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    # Create model card
    card = f"""---
language:
  - en
  - ta
  - hi
  - te
  - ml
tags:
  - audio-classification
  - deepfake-detection
  - voice-detection
license: mit
---

# AI Voice Detection Model

Custom fine-tuned model for detecting AI-generated speech.

## Model Details
- **Base Model**: {model_config.get('backbone_name', 'ensemble')}
- **Task**: Binary classification (HUMAN vs AI_GENERATED)
- **Pooling**: {model_config['pooling_type']}
- **Sample Rate**: {model_config['sample_rate']} Hz

## Usage
```python
# Load with the voice-detection-api
from app.core.model import VoiceDetector
detector = VoiceDetector()
result = detector.predict(audio_array)
```
"""
    with open(os.path.join(hf_dir, "README.md"), "w") as f:
        f.write(card)

    logger.info(f"✅ HuggingFace export saved to {hf_dir}")

    # Optional: push to hub
    repo_id = cfg["export"].get("hf_repo_id")
    if repo_id:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_folder(
                folder_path=hf_dir,
                repo_id=repo_id,
                repo_type="model",
                commit_message="Upload fine-tuned voice detection model",
            )
            logger.info(f"✅ Pushed to HuggingFace Hub: {repo_id}")
        except Exception as e:
            logger.warning(f"⚠️  HF Hub push failed: {e}. Files saved locally.")

    return hf_dir


def export_onnx(model, cfg, export_dir):
    """Export model to ONNX format for optimized CPU inference."""
    onnx_dir = os.path.join(export_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device
    sr = cfg["data"]["sample_rate"]
    max_len = int(cfg["data"]["max_duration_sec"] * sr)

    # Dummy input
    dummy = torch.randn(1, max_len).to(device)

    onnx_path = os.path.join(onnx_dir, "model.onnx")

    try:
        torch.onnx.export(
            model, dummy,
            onnx_path,
            input_names=["input_values"],
            output_names=["logits"],
            dynamic_axes={
                "input_values": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"},
            },
            opset_version=cfg["export"].get("onnx_opset", 14),
            do_constant_folding=True,
        )

        # Optional quantization
        if cfg["export"].get("quantize_onnx", False):
            try:
                from onnxruntime.quantization import quantize_dynamic, QuantType
                quantized_path = onnx_path.replace(".onnx", "_quantized.onnx")
                quantize_dynamic(onnx_path, quantized_path, weight_type=QuantType.QUInt8)
                logger.info(f"✅ Quantized ONNX saved to {quantized_path}")
            except ImportError:
                logger.warning("⚠️  onnxruntime not installed. Skipping quantization.")

        logger.info(f"✅ ONNX export saved to {onnx_path}")
    except Exception as e:
        logger.error(f"❌ ONNX export failed: {e}")

    return onnx_dir


def export_torchscript(model, cfg, export_dir):
    """Export model as TorchScript for portable inference."""
    ts_dir = os.path.join(export_dir, "torchscript")
    os.makedirs(ts_dir, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device
    sr = cfg["data"]["sample_rate"]
    max_len = int(cfg["data"]["max_duration_sec"] * sr)

    dummy = torch.randn(1, max_len).to(device)

    try:
        traced = torch.jit.trace(model, dummy)
        ts_path = os.path.join(ts_dir, "model.pt")
        traced.save(ts_path)
        logger.info(f"✅ TorchScript export saved to {ts_path}")
    except Exception as e:
        logger.error(f"❌ TorchScript export failed: {e}")

    return ts_dir


def auto_integrate(cfg, export_dir):
    """
    Update the main API's config.py to point to the new custom model.
    Creates a backup of the original config first.
    """
    api_config_path = os.path.join(os.path.dirname(export_dir), "..", "app", "config.py")
    api_config_path = os.path.abspath(api_config_path)

    if not os.path.exists(api_config_path):
        logger.warning(f"⚠️  API config not found at {api_config_path}. Skipping auto-integration.")
        return

    # Create backup
    backup_path = api_config_path + ".backup"
    if not os.path.exists(backup_path):
        shutil.copy2(api_config_path, backup_path)
        logger.info(f"📋 Backed up original config to {backup_path}")

    # Read current config
    with open(api_config_path, "r") as f:
        content = f.read()

    # Point to new model
    hf_model_dir = os.path.join(export_dir, "huggingface")
    if os.path.exists(hf_model_dir):
        new_model_path = os.path.abspath(hf_model_dir).replace("\\", "/")
        content = content.replace(
            'MODEL_NAME = os.getenv("MODEL_NAME", "mo-thecreator/Deepfake-audio-detection")',
            f'MODEL_NAME = os.getenv("MODEL_NAME", "{new_model_path}")'
        )

        with open(api_config_path, "w") as f:
            f.write(content)

        logger.info(f"✅ API config updated to use custom model: {new_model_path}")
    else:
        logger.warning("⚠️  HuggingFace export not found. Run with --format huggingface first.")


def main():
    parser = argparse.ArgumentParser(description="Export trained model to production formats")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--format", type=str, nargs="+",
                        default=None, help="Export formats: huggingface, onnx, torchscript")
    parser.add_argument("--integrate", action="store_true",
                        help="Auto-update API config to use the new model")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    export_dir = cfg["paths"]["export_dir"]
    os.makedirs(export_dir, exist_ok=True)

    formats = args.format or cfg["export"]["formats"]

    # Load the trained model
    from train import build_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = build_model(cfg, device)

    model_path = os.path.join(cfg["paths"]["output_dir"], "best_model", "model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"✅ Loaded best model from {model_path}")
    else:
        logger.warning("⚠️  No trained model found. Exporting untrained model.")

    model.eval()

    # Export
    for fmt in formats:
        if fmt == "huggingface":
            export_huggingface(model, cfg, export_dir)
        elif fmt == "onnx":
            export_onnx(model, cfg, export_dir)
        elif fmt == "torchscript":
            export_torchscript(model, cfg, export_dir)
        else:
            logger.warning(f"⚠️  Unknown format: {fmt}")

    if args.integrate:
        auto_integrate(cfg, export_dir)

    logger.info("🎉 Export complete!")


if __name__ == "__main__":
    main()
