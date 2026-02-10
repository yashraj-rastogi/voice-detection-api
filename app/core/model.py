
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from app.config import settings
from app.core.forensics import forensic_engine
from app.core.audio import segment_audio
import logging
import gc
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoiceDetector:
    """
    World-class voice detection engine.
    Combines neural model inference with forensic analysis for maximum accuracy.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VoiceDetector, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.feature_extractor = None
            cls._instance.device = "cpu"
            cls._instance.load_model()
        return cls._instance

    def load_model(self):
        try:
            logger.info(f"Loading model {settings.MODEL_NAME} on {self.device}...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                settings.MODEL_NAME
            )
            self.model = AutoModelForAudioClassification.from_pretrained(
                settings.MODEL_NAME,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
            gc.collect()
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    def _infer_single(self, audio_array: np.ndarray) -> tuple:
        """Run model inference on a single audio segment."""
        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=settings.SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = F.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_idx].item()

        # Get model label
        id2label = self.model.config.id2label
        label = str(id2label[pred_idx]).lower()

        # Map to binary: is it AI?
        is_ai = False
        if "fake" in label or "spoof" in label:
            is_ai = True
        elif "real" in label or "bonafide" in label:
            is_ai = False
        else:
            is_ai = (pred_idx == 1)

        # Return P(AI) score (0=human, 1=AI)
        if is_ai:
            ai_score = confidence
        else:
            ai_score = 1.0 - confidence

        return ai_score, confidence, is_ai

    def predict(self, audio_array: np.ndarray, audio_profile: dict = None,
                detailed: bool = False) -> dict:
        """
        Full detection pipeline:
        1. Multi-segment neural model inference
        2. Forensic analysis (spectral, temporal, formant, artifact)
        3. Score fusion for final verdict

        Returns a rich result dict.
        """
        if self.model is None:
            self.load_model()

        start_time = time.time()

        try:
            sr = settings.SAMPLE_RATE

            # ====== Stage 1: Multi-Segment Neural Inference ======
            segments = segment_audio(audio_array, sr, segment_sec=5.0, overlap_sec=1.0)
            segment_scores = []

            for seg in segments:
                ai_score, conf, is_ai = self._infer_single(seg)
                segment_scores.append(ai_score)

            # Aggregate: use mean with outlier dampening
            if len(segment_scores) > 2:
                # Remove highest and lowest, then average (trimmed mean)
                sorted_scores = sorted(segment_scores)
                trimmed = sorted_scores[1:-1]
                neural_score = float(np.mean(trimmed))
            else:
                neural_score = float(np.mean(segment_scores))

            neural_confidence = max(neural_score, 1.0 - neural_score)
            neural_verdict = "AI_GENERATED" if neural_score >= 0.5 else "HUMAN"

            logger.info(
                f"Neural: {neural_verdict} (score={neural_score:.4f}, "
                f"segments={len(segments)}, per-seg={[round(s, 3) for s in segment_scores]})"
            )

            # ====== Stage 2: Forensic Analysis ======
            forensic_results = forensic_engine.analyze(audio_array, sr)
            forensic_score = forensic_engine.compute_forensic_score(forensic_results)
            all_artifacts = forensic_engine.get_all_artifacts(forensic_results)

            logger.info(
                f"Forensics: score={forensic_score:.4f}, "
                f"artifacts={len(all_artifacts)} found"
            )

            # ====== Stage 3: Score Fusion ======
            # Neural model gets higher weight (it's trained on actual data)
            # Forensics provide supporting evidence and catch edge cases
            NEURAL_WEIGHT = 0.70
            FORENSIC_WEIGHT = 0.30

            fused_score = (neural_score * NEURAL_WEIGHT) + (forensic_score * FORENSIC_WEIGHT)

            # Boost confidence if neural and forensics agree
            neural_says_ai = neural_score >= 0.5
            forensic_says_ai = forensic_score >= 0.4
            agreement = (neural_says_ai == forensic_says_ai)

            if agreement:
                # Both agree → push score further from 0.5
                fused_score = fused_score * 1.1 if fused_score >= 0.5 else fused_score * 0.9
                fused_score = max(0.0, min(1.0, fused_score))

            # Final verdict
            final_verdict = "AI_GENERATED" if fused_score >= 0.5 else "HUMAN"
            final_confidence = max(fused_score, 1.0 - fused_score)

            # Ensure minimum confidence floor
            final_confidence = max(final_confidence, 0.51)

            inference_time = round((time.time() - start_time) * 1000, 1)

            logger.info(
                f"FINAL: {final_verdict} (confidence={final_confidence:.4f}, "
                f"fused={fused_score:.4f}, neural={neural_score:.4f}, "
                f"forensic={forensic_score:.4f}, time={inference_time}ms)"
            )

            # ====== Build Response ======
            result = {
                "classification": final_verdict,
                "confidence": round(final_confidence, 4),
                "fused_score": round(fused_score, 4),
                "inference_time_ms": inference_time,
                "analyzers_agree": agreement,
            }

            if detailed:
                result["forensics"] = {
                    "neural_model": {
                        "score": round(neural_score, 4),
                        "verdict": neural_verdict,
                        "segments_analyzed": len(segments),
                        "per_segment_scores": [round(s, 4) for s in segment_scores],
                    },
                    **forensic_results,
                }
                result["artifacts_summary"] = all_artifacts

            if audio_profile:
                result["audio_profile"] = audio_profile

            return result

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise RuntimeError(f"Prediction failed: {e}")


voice_detector = VoiceDetector()
