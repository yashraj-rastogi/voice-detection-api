"""
Augmentation Pipeline — State-of-the-art audio augmentations for robust training.
Each augmentation is independently controlled via config.yaml.
"""
import numpy as np
import librosa
import io
import random
import subprocess
import tempfile
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class AugConfig:
    """Parsed augmentation config."""
    gaussian_noise_prob: float = 0.5
    gaussian_noise_snr_range: Tuple[float, float] = (15, 40)
    speed_perturb_prob: float = 0.3
    speed_rate_range: Tuple[float, float] = (0.9, 1.1)
    pitch_shift_prob: float = 0.2
    pitch_semitone_range: Tuple[int, int] = (-2, 2)
    spec_augment_prob: float = 0.5
    spec_freq_mask: int = 20
    spec_time_mask: int = 40
    spec_num_freq: int = 2
    spec_num_time: int = 2
    codec_sim_prob: float = 0.3
    codec_types: List[str] = field(default_factory=lambda: ["mp3", "ogg"])
    codec_bitrates: List[int] = field(default_factory=lambda: [32, 64, 128])
    volume_perturb_prob: float = 0.4
    volume_gain_range: Tuple[float, float] = (-6, 6)

    @classmethod
    def from_dict(cls, d: dict) -> "AugConfig":
        aug = d.get("augmentation", {})
        return cls(
            gaussian_noise_prob=aug.get("gaussian_noise", {}).get("prob", 0.5),
            gaussian_noise_snr_range=tuple(aug.get("gaussian_noise", {}).get("snr_range_db", [15, 40])),
            speed_perturb_prob=aug.get("speed_perturb", {}).get("prob", 0.3),
            speed_rate_range=tuple(aug.get("speed_perturb", {}).get("rate_range", [0.9, 1.1])),
            pitch_shift_prob=aug.get("pitch_shift", {}).get("prob", 0.2),
            pitch_semitone_range=tuple(aug.get("pitch_shift", {}).get("semitone_range", [-2, 2])),
            spec_augment_prob=aug.get("spec_augment", {}).get("prob", 0.5),
            spec_freq_mask=aug.get("spec_augment", {}).get("freq_mask_param", 20),
            spec_time_mask=aug.get("spec_augment", {}).get("time_mask_param", 40),
            spec_num_freq=aug.get("spec_augment", {}).get("num_freq_masks", 2),
            spec_num_time=aug.get("spec_augment", {}).get("num_time_masks", 2),
            codec_sim_prob=aug.get("codec_simulation", {}).get("prob", 0.3),
            codec_types=aug.get("codec_simulation", {}).get("codecs", ["mp3", "ogg"]),
            codec_bitrates=aug.get("codec_simulation", {}).get("bitrates", [32, 64, 128]),
            volume_perturb_prob=aug.get("volume_perturb", {}).get("prob", 0.4),
            volume_gain_range=tuple(aug.get("volume_perturb", {}).get("gain_range_db", [-6, 6])),
        )


# ===============================================================
#  Waveform-Level Augmentations
# ===============================================================

def add_gaussian_noise(y: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian noise at a given SNR."""
    rms_signal = np.sqrt(np.mean(y ** 2))
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise = np.random.normal(0, rms_noise, y.shape)
    return y + noise


def speed_perturbation(y: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Change speed without changing pitch (time-stretch)."""
    return librosa.effects.time_stretch(y, rate=rate)


def pitch_shift(y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Shift pitch by N semitones."""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)


def volume_perturbation(y: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply volume gain in dB."""
    gain = 10 ** (gain_db / 20)
    return y * gain


def codec_simulation(y: np.ndarray, sr: int, codec: str = "mp3", bitrate: int = 64) -> np.ndarray:
    """Simulate lossy codec compression by encoding and decoding via ffmpeg.
    Falls back to identity if ffmpeg is unavailable."""
    try:
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            sf.write(tmp_in.name, y, sr)
            tmp_in_path = tmp_in.name

        ext = ".mp3" if codec == "mp3" else ".ogg"
        tmp_out_path = tmp_in_path.replace(".wav", ext)
        tmp_back_path = tmp_in_path.replace(".wav", "_back.wav")

        # Encode
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in_path, "-b:a", f"{bitrate}k", tmp_out_path],
            capture_output=True, timeout=10
        )
        # Decode back
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_out_path, "-ar", str(sr), tmp_back_path],
            capture_output=True, timeout=10
        )

        y_aug, _ = librosa.load(tmp_back_path, sr=sr, mono=True)

        # Cleanup
        for p in [tmp_in_path, tmp_out_path, tmp_back_path]:
            if os.path.exists(p):
                os.unlink(p)

        return y_aug
    except Exception:
        return y  # Fallback: return original


# ===============================================================
#  Spectrogram-Level Augmentation (SpecAugment)
# ===============================================================

def spec_augment(spec: np.ndarray, freq_mask: int, time_mask: int,
                 num_freq: int, num_time: int) -> np.ndarray:
    """Apply SpecAugment masking on a spectrogram (freq x time)."""
    spec = spec.copy()
    num_freq_bins, num_time_steps = spec.shape

    # Frequency masking
    for _ in range(num_freq):
        f = random.randint(0, min(freq_mask, num_freq_bins - 1))
        f0 = random.randint(0, num_freq_bins - f)
        spec[f0:f0 + f, :] = 0

    # Time masking
    for _ in range(num_time):
        t = random.randint(0, min(time_mask, num_time_steps - 1))
        t0 = random.randint(0, num_time_steps - t)
        spec[:, t0:t0 + t] = 0

    return spec


# ===============================================================
#  Augmentation Pipeline (Composes all transforms)
# ===============================================================

class AugmentationPipeline:
    """Applies a random subset of augmentations to each sample."""

    def __init__(self, cfg: AugConfig, sr: int = 16000):
        self.cfg = cfg
        self.sr = sr

    def __call__(self, y: np.ndarray) -> np.ndarray:
        """Apply waveform-level augmentations to a single audio sample."""
        c = self.cfg

        # 1. Speed perturbation (changes length, so apply first)
        if random.random() < c.speed_perturb_prob:
            rate = random.uniform(*c.speed_rate_range)
            y = speed_perturbation(y, self.sr, rate)

        # 2. Pitch shift
        if random.random() < c.pitch_shift_prob:
            semi = random.uniform(*c.pitch_semitone_range)
            y = pitch_shift(y, self.sr, semi)

        # 3. Volume perturbation
        if random.random() < c.volume_perturb_prob:
            gain = random.uniform(*c.volume_gain_range)
            y = volume_perturbation(y, gain)

        # 4. Additive Gaussian noise
        if random.random() < c.gaussian_noise_prob:
            snr = random.uniform(*c.gaussian_noise_snr_range)
            y = add_gaussian_noise(y, snr)

        # 5. Codec simulation (expensive, lower probability)
        if random.random() < c.codec_sim_prob:
            codec = random.choice(c.codec_types)
            bitrate = random.choice(c.codec_bitrates)
            y = codec_simulation(y, self.sr, codec, bitrate)

        # Normalize to prevent clipping
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak

        return y.astype(np.float32)

    def apply_spec_augment(self, spec: np.ndarray) -> np.ndarray:
        """Apply SpecAugment to a spectrogram tensor."""
        c = self.cfg
        if random.random() < c.spec_augment_prob:
            spec = spec_augment(
                spec, c.spec_freq_mask, c.spec_time_mask,
                c.spec_num_freq, c.spec_num_time
            )
        return spec
