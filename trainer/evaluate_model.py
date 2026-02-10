"""
Evaluation Engine — Comprehensive model evaluation with industry-standard metrics.
Produces: EER, AUC-ROC, per-language breakdown, calibration curves, confusion matrix.
"""
import os
import json
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import logging
import time
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_curve, auc, precision_recall_fscore_support,
    confusion_matrix, accuracy_score, classification_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def compute_eer(labels, scores):
    """Compute Equal Error Rate."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    return eer, eer_threshold, fpr, tpr


def compute_calibration(labels, probs, n_bins=10):
    """Compute Expected Calibration Error (ECE) and reliability diagram data."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_data = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        bin_count = mask.sum()
        bin_data.append({
            "bin_center": (lo + hi) / 2,
            "accuracy": float(bin_acc),
            "confidence": float(bin_conf),
            "count": int(bin_count),
        })

    # ECE
    total = len(labels)
    ece = sum(
        (b["count"] / total) * abs(b["accuracy"] - b["confidence"])
        for b in bin_data
    )
    return ece, bin_data


def evaluate_model(cfg: dict):
    """Full model evaluation pipeline."""
    from train import AudioDataset, build_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = cfg["paths"]["output_dir"]
    metadata_dir = os.path.join(output_dir, "metadata")
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    # Load test set
    test_csv = os.path.join(metadata_dir, "test.csv")
    if not os.path.exists(test_csv):
        logger.error("❌ test.csv not found. Run prepare_data.py first.")
        return

    test_dataset = AudioDataset(test_csv, cfg, augment=False)
    test_loader = DataLoader(
        test_dataset, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=0
    )

    # Load model
    model, _ = build_model(cfg, device)
    model_path = os.path.join(output_dir, "best_model", "model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"✅ Loaded model from {model_path}")
    else:
        logger.warning("⚠️  No saved model found. Using randomly initialized model.")

    model.eval()

    # ============ Collect Predictions ============
    all_labels = []
    all_probs = []
    all_preds = []
    latencies = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["input_values"].to(device)
            labels = batch["labels"]

            start = time.time()
            logits = model(inputs)
            latency = (time.time() - start) / inputs.size(0)

            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()  # P(AI_GENERATED)
            preds = logits.argmax(dim=-1).cpu().numpy()

            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
            all_preds.extend(preds)
            latencies.append(latency)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    # ============ Compute Metrics ============
    report = {}

    # 1. EER
    eer, eer_threshold, fpr, tpr = compute_eer(all_labels, all_probs)
    report["eer"] = round(float(eer), 4)
    report["eer_threshold"] = round(float(eer_threshold), 4)

    # 2. AUC-ROC
    auc_roc = auc(fpr, tpr)
    report["auc_roc"] = round(float(auc_roc), 4)

    # 3. Accuracy, Precision, Recall, F1
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", pos_label=1
    )
    report["accuracy"] = round(float(accuracy), 4)
    report["precision"] = round(float(precision), 4)
    report["recall"] = round(float(recall), 4)
    report["f1"] = round(float(f1), 4)

    # 4. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    report["confusion_matrix"] = cm.tolist()

    # 5. Calibration
    if cfg["evaluation"]["calibration_curve"]:
        ece, bin_data = compute_calibration(all_labels, all_probs)
        report["ece"] = round(float(ece), 4)
        report["calibration_bins"] = bin_data

    # 6. Latency
    if cfg["evaluation"]["latency_benchmark"]:
        avg_latency = np.mean(latencies)
        report["avg_latency_ms"] = round(float(avg_latency * 1000), 2)

    # 7. Per-language breakdown (if language column exists)
    if cfg["evaluation"]["per_language"]:
        test_df = pd.read_csv(test_csv)
        if "language" in test_df.columns:
            languages = test_df["language"].unique()
            lang_report = {}
            for lang in languages:
                mask = test_df["language"] == lang
                if mask.sum() < 2:
                    continue
                l_labels = all_labels[mask.values]
                l_probs = all_probs[mask.values]
                l_preds = all_preds[mask.values]
                l_eer, _, _, _ = compute_eer(l_labels, l_probs)
                l_acc = accuracy_score(l_labels, l_preds)
                lang_report[lang] = {
                    "samples": int(mask.sum()),
                    "eer": round(float(l_eer), 4),
                    "accuracy": round(float(l_acc), 4),
                }
            report["per_language"] = lang_report

    # ============ Print Report ============
    print("\n" + "=" * 60)
    print("  📊  MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"  Samples tested    : {len(all_labels)}")
    print(f"  EER               : {report['eer']:.4f}  (threshold={report['eer_threshold']:.4f})")
    print(f"  AUC-ROC           : {report['auc_roc']:.4f}")
    print(f"  Accuracy          : {report['accuracy']:.4f}")
    print(f"  Precision         : {report['precision']:.4f}")
    print(f"  Recall            : {report['recall']:.4f}")
    print(f"  F1 Score          : {report['f1']:.4f}")
    if "ece" in report:
        print(f"  ECE (Calibration) : {report['ece']:.4f}")
    if "avg_latency_ms" in report:
        print(f"  Avg Latency       : {report['avg_latency_ms']:.1f} ms/sample")
    print()
    print("  Confusion Matrix:")
    print(f"    {'':12s} Pred HUMAN  Pred AI")
    print(f"    {'True HUMAN':12s}   {cm[0][0]:6d}    {cm[0][1]:6d}")
    print(f"    {'True AI':12s}   {cm[1][0]:6d}    {cm[1][1]:6d}")

    if "per_language" in report:
        print("\n  Per-Language Breakdown:")
        for lang, metrics in report["per_language"].items():
            print(f"    {lang:10s}: EER={metrics['eer']:.4f}  Acc={metrics['accuracy']:.4f}  "
                  f"(n={metrics['samples']})")

    print("=" * 60 + "\n")

    # ============ Save Report ============
    report_path = os.path.join(eval_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"📋 Full report saved to {report_path}")

    # Save classification report
    cls_report = classification_report(
        all_labels, all_preds,
        target_names=["HUMAN", "AI_GENERATED"],
        output_dict=True
    )
    with open(os.path.join(eval_dir, "classification_report.json"), "w") as f:
        json.dump(cls_report, f, indent=2)

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained deepfake detection model")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    evaluate_model(cfg)


if __name__ == "__main__":
    main()
