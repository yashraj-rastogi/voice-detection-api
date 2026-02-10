"""
Forensic Analyzers — Four independent analysis engines that examine audio
for specific signatures of AI generation vs natural human speech.

Each analyzer returns a score (0=human, 1=AI) and a list of detected artifacts.
The final detection fuses all analyzer scores for maximum accuracy.
"""
import numpy as np
import librosa
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class AnalyzerResult:
    """Result from a single forensic analyzer."""
    name: str
    score: float  # 0.0 = definitely human, 1.0 = definitely AI
    verdict: str  # "HUMAN" or "AI_GENERATED"
    artifacts_found: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioProfile:
    """Technical profile of the audio sample."""
    duration_sec: float = 0.0
    snr_db: float = 0.0
    clipping_detected: bool = False
    silence_ratio: float = 0.0
    rms_energy: float = 0.0
    sample_rate: int = 16000
    num_segments: int = 1


# ===============================================================
#  Spectral Analyzer
# ===============================================================

class SpectralAnalyzer:
    """
    Detects AI signatures in the frequency domain:
    - Unnaturally smooth spectral envelope (TTS models produce cleaner spectra)
    - Missing or artificial harmonics
    - Sharp frequency cutoffs (vocoder artifacts)
    - Abnormal spectral flatness (AI speech is more tonal)
    """

    def analyze(self, y: np.ndarray, sr: int) -> AnalyzerResult:
        artifacts = []
        details = {}

        try:
            # 1. Spectral Flatness — AI speech tends to have lower flatness (more tonal)
            flatness = librosa.feature.spectral_flatness(y=y)[0]
            mean_flatness = float(np.mean(flatness))
            std_flatness = float(np.std(flatness))
            details["spectral_flatness_mean"] = round(mean_flatness, 4)
            details["spectral_flatness_std"] = round(std_flatness, 4)

            # Human speech has higher variance in spectral flatness
            if std_flatness < 0.02:
                artifacts.append("unnaturally_uniform_spectral_texture")
            if mean_flatness < 0.005:
                artifacts.append("overly_tonal_spectrum")

            # 2. Spectral Bandwidth — AI audio often has narrower bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            mean_bw = float(np.mean(bandwidth))
            std_bw = float(np.std(bandwidth))
            details["spectral_bandwidth_mean"] = round(mean_bw, 1)
            details["spectral_bandwidth_std"] = round(std_bw, 1)

            if std_bw < 200:
                artifacts.append("unnaturally_consistent_bandwidth")

            # 3. Spectral Centroid Variance — AI speech has more stable centroid
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            centroid_cv = float(np.std(centroid) / (np.mean(centroid) + 1e-10))
            details["spectral_centroid_cv"] = round(centroid_cv, 4)

            if centroid_cv < 0.15:
                artifacts.append("unnaturally_stable_spectral_centroid")

            # 4. Harmonic-to-Noise Ratio variation
            harmonic, percussive = librosa.effects.hpss(y)
            h_energy = np.sum(harmonic ** 2)
            p_energy = np.sum(percussive ** 2)
            hnr = float(10 * np.log10(h_energy / (p_energy + 1e-10)))
            details["hnr_db"] = round(hnr, 1)

            if hnr > 30:
                artifacts.append("unnaturally_clean_harmonics")

            # 5. High-frequency energy analysis — some vocoders cut at specific frequencies
            S = np.abs(librosa.stft(y))
            freq_bins = S.shape[0]
            high_freq_start = int(freq_bins * 0.75)
            high_energy = np.mean(S[high_freq_start:, :])
            low_energy = np.mean(S[:high_freq_start, :])
            hf_ratio = float(high_energy / (low_energy + 1e-10))
            details["high_freq_energy_ratio"] = round(hf_ratio, 4)

            if hf_ratio < 0.01:
                artifacts.append("sharp_high_frequency_cutoff")

            # Score: more artifacts = more likely AI
            score = min(1.0, len(artifacts) * 0.22)

        except Exception as e:
            logger.warning(f"SpectralAnalyzer error: {e}")
            score = 0.5
            artifacts = []
            details["error"] = str(e)

        return AnalyzerResult(
            name="spectral_analysis",
            score=round(score, 4),
            verdict="AI_GENERATED" if score >= 0.5 else "HUMAN",
            artifacts_found=artifacts,
            details=details,
        )


# ===============================================================
#  Temporal Analyzer
# ===============================================================

class TemporalAnalyzer:
    """
    Detects AI signatures in the time domain:
    - Robotic / metronomic pause timing (TTS pauses are uniform)
    - Missing micro-variations in energy (humans waver naturally)
    - Unnaturally smooth energy envelope
    - Consistent zero-crossing rate (humans vary more)
    """

    def analyze(self, y: np.ndarray, sr: int) -> AnalyzerResult:
        artifacts = []
        details = {}

        try:
            # 1. Energy contour smoothness
            # Human speech has natural energy fluctuations; AI is smoother
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)    # 10ms hop
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

            # Compute energy contour variability
            if len(rms) > 10:
                rms_diff = np.diff(rms)
                energy_roughness = float(np.std(rms_diff) / (np.mean(rms) + 1e-10))
                details["energy_roughness"] = round(energy_roughness, 4)

                if energy_roughness < 0.08:
                    artifacts.append("unnaturally_smooth_energy_contour")

            # 2. Zero-Crossing Rate consistency
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
            zcr_cv = float(np.std(zcr) / (np.mean(zcr) + 1e-10))
            details["zcr_coefficient_of_variation"] = round(zcr_cv, 4)

            if zcr_cv < 0.25:
                artifacts.append("unnaturally_consistent_zero_crossings")

            # 3. Pause regularity analysis
            # Detect silence segments and check if pauses are evenly spaced
            silence_threshold = np.percentile(np.abs(y), 10)
            is_silent = np.abs(y) < silence_threshold * 3
            # Find silence boundaries
            silent_changes = np.diff(is_silent.astype(int))
            pause_starts = np.where(silent_changes == 1)[0]
            pause_ends = np.where(silent_changes == -1)[0]

            if len(pause_starts) >= 3:
                pause_intervals = np.diff(pause_starts) / sr
                interval_cv = float(np.std(pause_intervals) / (np.mean(pause_intervals) + 1e-10))
                details["pause_interval_cv"] = round(interval_cv, 4)
                details["num_pauses"] = len(pause_starts)

                if interval_cv < 0.2 and len(pause_starts) > 3:
                    artifacts.append("metronomic_pause_timing")

            # 4. Micro-jitter analysis (human speech has natural pitch jitter)
            # We use autocorrelation-based local regularity
            if len(y) > sr:
                # Split into 100ms chunks and measure local energy variance
                chunk_size = int(0.1 * sr)
                chunks = [y[i:i+chunk_size] for i in range(0, len(y) - chunk_size, chunk_size)]
                chunk_energies = [np.sqrt(np.mean(c ** 2)) for c in chunks]

                if len(chunk_energies) > 4:
                    energy_autocorr = np.correlate(
                        chunk_energies - np.mean(chunk_energies),
                        chunk_energies - np.mean(chunk_energies), mode='full'
                    )
                    energy_autocorr = energy_autocorr / (energy_autocorr.max() + 1e-10)
                    mid = len(energy_autocorr) // 2
                    # High autocorrelation at small lags = too regular
                    if len(energy_autocorr) > mid + 3:
                        avg_autocorr = float(np.mean(energy_autocorr[mid+1:mid+4]))
                        details["energy_autocorrelation"] = round(avg_autocorr, 4)

                        if avg_autocorr > 0.7:
                            artifacts.append("repetitive_energy_pattern")

            # Score
            score = min(1.0, len(artifacts) * 0.28)

        except Exception as e:
            logger.warning(f"TemporalAnalyzer error: {e}")
            score = 0.5
            artifacts = []
            details["error"] = str(e)

        return AnalyzerResult(
            name="temporal_analysis",
            score=round(score, 4),
            verdict="AI_GENERATED" if score >= 0.5 else "HUMAN",
            artifacts_found=artifacts,
            details=details,
        )


# ===============================================================
#  Formant Analyzer
# ===============================================================

class FormantAnalyzer:
    """
    Detects AI signatures in formant structure:
    - Uniform formant transitions (TTS produces overly smooth formants)
    - Missing natural formant jitter
    - Abnormal F1/F2 distribution
    """

    def analyze(self, y: np.ndarray, sr: int) -> AnalyzerResult:
        artifacts = []
        details = {}

        try:
            # Use mel-frequency bands as a proxy for formant analysis
            # (True formant extraction requires specialized DSP; mel bands are a good proxy)

            # 1. MFCC stability — AI speech produces more stable MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_deltas = librosa.feature.delta(mfccs)

            # Coefficient of variation for each MFCC
            mfcc_cvs = []
            for i in range(1, 13):  # Skip MFCC-0 (energy)
                cv = float(np.std(mfccs[i]) / (np.abs(np.mean(mfccs[i])) + 1e-10))
                mfcc_cvs.append(cv)

            avg_mfcc_cv = float(np.mean(mfcc_cvs))
            details["avg_mfcc_cv"] = round(avg_mfcc_cv, 4)

            if avg_mfcc_cv < 0.5:
                artifacts.append("unnaturally_stable_formant_structure")

            # 2. Delta MFCC smoothness — AI transitions are too smooth
            delta_roughness = float(np.mean(np.abs(librosa.feature.delta(mfcc_deltas))))
            details["delta_mfcc_roughness"] = round(delta_roughness, 4)

            if delta_roughness < 0.3:
                artifacts.append("overly_smooth_formant_transitions")

            # 3. MFCC correlation between adjacent frames — AI is more correlated
            if mfccs.shape[1] > 10:
                frame_corrs = []
                for i in range(mfccs.shape[1] - 1):
                    corr = float(np.corrcoef(mfccs[1:, i], mfccs[1:, i+1])[0, 1])
                    if not np.isnan(corr):
                        frame_corrs.append(corr)

                if frame_corrs:
                    mean_corr = float(np.mean(frame_corrs))
                    details["inter_frame_correlation"] = round(mean_corr, 4)

                    if mean_corr > 0.95:
                        artifacts.append("excessive_inter_frame_correlation")

            # 4. Mel-band energy distribution — AI often has specific band patterns
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            band_stds = np.std(mel_db, axis=1)
            details["mel_band_std_range"] = round(float(np.max(band_stds) - np.min(band_stds)), 2)

            if np.max(band_stds) - np.min(band_stds) < 5.0:
                artifacts.append("uniform_mel_band_energy")

            # Score
            score = min(1.0, len(artifacts) * 0.28)

        except Exception as e:
            logger.warning(f"FormantAnalyzer error: {e}")
            score = 0.5
            artifacts = []
            details["error"] = str(e)

        return AnalyzerResult(
            name="formant_analysis",
            score=round(score, 4),
            verdict="AI_GENERATED" if score >= 0.5 else "HUMAN",
            artifacts_found=artifacts,
            details=details,
        )


# ===============================================================
#  Artifact Detector
# ===============================================================

class ArtifactDetector:
    """
    Detects synthesis artifacts in the raw waveform:
    - Phase discontinuities (vocoder glitches)
    - Abrupt volume changes (concatenation boundaries)
    - Click artifacts
    - Unnatural silence-to-speech transitions
    """

    def analyze(self, y: np.ndarray, sr: int) -> AnalyzerResult:
        artifacts = []
        details = {}

        try:
            # 1. Click / pop detection — sudden amplitude spikes
            threshold = np.std(y) * 5
            clicks = np.where(np.abs(np.diff(y)) > threshold)[0]
            num_clicks = len(clicks)
            click_rate = num_clicks / (len(y) / sr)
            details["click_rate_per_sec"] = round(click_rate, 2)

            if click_rate > 10:
                artifacts.append("synthesis_click_artifacts")

            # 2. Waveform symmetry — human speech is roughly symmetric
            pos_rms = np.sqrt(np.mean(y[y > 0] ** 2)) if np.any(y > 0) else 0
            neg_rms = np.sqrt(np.mean(y[y < 0] ** 2)) if np.any(y < 0) else 0
            symmetry = float(pos_rms / (neg_rms + 1e-10))
            details["waveform_symmetry"] = round(symmetry, 4)

            if abs(symmetry - 1.0) > 0.3:
                artifacts.append("asymmetric_waveform")

            # 3. Silence segment quality — AI silence is often "too perfect" (near-zero)
            silence_segments = np.abs(y) < 0.001
            if np.any(silence_segments):
                silent_vals = y[silence_segments]
                silence_noise_floor = float(np.std(silent_vals))
                details["silence_noise_floor"] = round(silence_noise_floor, 6)

                if silence_noise_floor < 1e-5 and np.sum(silence_segments) > sr * 0.05:
                    artifacts.append("digitally_perfect_silence")

            # 4. Transition sharpness — AI-to-silence boundaries are often too clean
            envelope = np.abs(librosa.util.frame(y, frame_length=int(0.01 * sr),
                                                  hop_length=int(0.005 * sr))).max(axis=0)
            if len(envelope) > 10:
                env_diff = np.abs(np.diff(envelope))
                sharp_transitions = np.sum(env_diff > np.percentile(env_diff, 98))
                details["sharp_transitions"] = int(sharp_transitions)

                if sharp_transitions > len(envelope) * 0.05:
                    artifacts.append("unnaturally_sharp_speech_boundaries")

            # 5. Periodicity check — AI waveforms can be excessively periodic
            if len(y) > sr:
                segment = y[:sr]  # First second
                autocorr = np.correlate(segment, segment, mode='full')
                autocorr = autocorr[len(autocorr) // 2:]
                autocorr = autocorr / (autocorr[0] + 1e-10)
                # Find peaks in autocorrelation
                peaks = []
                for i in range(1, len(autocorr) - 1):
                    if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.5:
                        peaks.append(i)
                    if len(peaks) > 5:
                        break
                details["strong_periodicity_peaks"] = len(peaks)

                if len(peaks) > 4:
                    artifacts.append("excessive_signal_periodicity")

            # Score
            score = min(1.0, len(artifacts) * 0.22)

        except Exception as e:
            logger.warning(f"ArtifactDetector error: {e}")
            score = 0.5
            artifacts = []
            details["error"] = str(e)

        return AnalyzerResult(
            name="artifact_detection",
            score=round(score, 4),
            verdict="AI_GENERATED" if score >= 0.5 else "HUMAN",
            artifacts_found=artifacts,
            details=details,
        )


# ===============================================================
#  Forensic Engine (orchestrates all analyzers)
# ===============================================================

class ForensicEngine:
    """
    Runs all forensic analyzers and produces a combined result.
    """

    def __init__(self):
        self.spectral = SpectralAnalyzer()
        self.temporal = TemporalAnalyzer()
        self.formant = FormantAnalyzer()
        self.artifact = ArtifactDetector()

    def analyze(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Run all analyzers and return combined forensic report."""
        results = {}

        # Run all 4 analyzers
        for analyzer in [self.spectral, self.temporal, self.formant, self.artifact]:
            result = analyzer.analyze(y, sr)
            results[result.name] = {
                "score": result.score,
                "verdict": result.verdict,
                "artifacts_found": result.artifacts_found,
                "details": result.details,
            }

        return results

    def compute_forensic_score(self, forensic_results: Dict[str, Any]) -> float:
        """
        Compute a weighted forensic score from all analyzers.
        Returns 0.0 (definitely human) to 1.0 (definitely AI).
        """
        weights = {
            "spectral_analysis": 0.30,
            "temporal_analysis": 0.25,
            "formant_analysis": 0.25,
            "artifact_detection": 0.20,
        }

        weighted_sum = 0.0
        total_weight = 0.0
        for name, result in forensic_results.items():
            w = weights.get(name, 0.25)
            weighted_sum += result["score"] * w
            total_weight += w

        return round(weighted_sum / (total_weight + 1e-10), 4)

    def get_all_artifacts(self, forensic_results: Dict[str, Any]) -> List[str]:
        """Collect all artifacts found across all analyzers."""
        all_artifacts = []
        for result in forensic_results.values():
            all_artifacts.extend(result.get("artifacts_found", []))
        return all_artifacts


# Singleton instance
forensic_engine = ForensicEngine()
