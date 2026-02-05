
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from app.config import settings
import logging
import gc

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
            # Force CPU to save memory on free tier
            cls._instance.device = "cpu"
            cls._instance.load_model()
        return cls._instance

    def load_model(self):
        try:
            logger.info(f"Loading model {settings.MODEL_NAME} on {self.device}...")
            
            # Clear memory before loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load with memory optimization
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                settings.MODEL_NAME
            )
            self.model = AutoModelForAudioClassification.from_pretrained(
                settings.MODEL_NAME,
                low_cpu_mem_usage=True,  # Memory optimization
                torch_dtype=torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
            
            # Clear unused memory
            gc.collect()
            
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
        Refined prediction for stability.
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
            
            # Use raw softmax for the base confidence
            probs = F.softmax(logits, dim=-1)
            
            # Get model labels from config
            id2label = self.model.config.id2label
            
            # Get the predicted class index
            pred_idx = torch.argmax(probs, dim=-1).item()
            label = str(id2label[pred_idx]).lower()
            confidence = probs[0][pred_idx].item()
            
            logger.info(f"Model Raw Output: Index={pred_idx}, Label={label}, Confidence={confidence:.4f}")
            
            # Robust Mapping Logic
            # mo-thecreator/Deepfake-audio-detection usually uses:
            # 0 -> REAL, 1 -> FAKE
            
            is_ai = False
            if "fake" in label or "spoof" in label:
                is_ai = True
            elif "real" in label or "bonafide" in label:
                is_ai = False
            else:
                # Direct index mapping fallback (very safe for this specific model)
                if pred_idx == 1:
                    is_ai = True
                else:
                    is_ai = False
            
            result_label = "AI_GENERATED" if is_ai else "HUMAN"
            
            # Stability check: If confidence is too low (< 0.6), 
            # the model is essentially guessing.
            if confidence < 0.6:
                logger.info(f"Low confidence ({confidence:.4f}) detected. Result might be uncertain.")

            return result_label, confidence

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

voice_detector = VoiceDetector()
