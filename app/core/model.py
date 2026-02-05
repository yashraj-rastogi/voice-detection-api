
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from app.config import settings
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceDetector:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VoiceDetector, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.feature_extractor = None
            cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._instance.load_model()
        return cls._instance

    def load_model(self):
        try:
            logger.info(f"Loading model {settings.MODEL_NAME} on {self.device}...")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(settings.MODEL_NAME)
            self.model = AutoModelForAudioClassification.from_pretrained(settings.MODEL_NAME)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    def calibrate_confidence(self, probs, temperature=1.5):
        """
        Apply temperature scaling to calibrate confidence scores.
        This makes the model less overconfident and more reliable.
        
        Temperature > 1.0 makes predictions less confident (more realistic)
        Temperature < 1.0 makes predictions more confident
        """
        # Apply temperature scaling to logits before softmax
        logits = torch.log(probs + 1e-10)  # Convert back to logits
        scaled_logits = logits / temperature
        calibrated_probs = F.softmax(scaled_logits, dim=-1)
        return calibrated_probs
    
    def predict(self, audio_array):
        """
        Enhanced prediction with confidence calibration.
        Predicts whether the audio is REAL or FAKE (AI Generated).
        """
        if self.model is None:
            self.load_model()
            
        try:
            # Prepare input
            inputs = self.feature_extractor(
                audio_array, 
                sampling_rate=settings.SAMPLE_RATE, 
                return_tensors="pt", 
                padding=True
            )
            
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            # Inference
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            # Softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Apply confidence calibration (Phase 1 Enhancement)
            # Temperature scaling makes predictions more reliable
            calibrated_probs = self.calibrate_confidence(probs, temperature=1.3)
            
            # Get model labels
            id2label = self.model.config.id2label
            
            # Get the predicted class index
            pred_idx = torch.argmax(calibrated_probs, dim=-1).item()
            label = id2label[pred_idx]
            confidence = calibrated_probs[0][pred_idx].item()
            
            # Get both class probabilities for better decision making
            all_probs = calibrated_probs[0].cpu().numpy()
            
            # Calculate prediction certainty (margin between top 2 classes)
            sorted_probs = np.sort(all_probs)[::-1]
            certainty_margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
            
            # Log for debugging
            logger.info(f"Prediction: {label} (confidence: {confidence:.4f}, margin: {certainty_margin:.4f})")
            
            # Standardizing output as per requirement
            # If label contains "fake" or "generated", map to "AI_GENERATED"
            # If label contains "real" or "bonafide", map to "HUMAN"
            
            result_label = "UNKNOWN"
            if "fake" in label.lower() or "spoof" in label.lower():
                result_label = "AI_GENERATED"
            elif "real" in label.lower() or "bonafide" in label.lower():
                result_label = "HUMAN"
            else:
                # Fallback based on index if labels are ambiguous
                # For `mo-thecreator/Deepfake-audio-detection`:
                # label 0: real, label 1: fake
                result_label = label
            
            # Apply confidence adjustment based on certainty margin
            # If the model is very uncertain (close call), reduce reported confidence
            if certainty_margin < 0.2:  # Very close call
                confidence = confidence * 0.85  # Reduce confidence by 15%
                logger.info(f"Low certainty margin detected, adjusted confidence to {confidence:.4f}")
                
            return result_label, confidence

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

voice_detector = VoiceDetector()
