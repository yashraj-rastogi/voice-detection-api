"""
Explanation Engine — Generates evidence-based, human-readable explanations
from actual forensic analysis results.
"""
from typing import List, Dict, Any


# Human-readable names for detected artifacts
ARTIFACT_DESCRIPTIONS = {
    # Spectral
    "unnaturally_uniform_spectral_texture": "Spectral texture is unnaturally uniform across the sample",
    "overly_tonal_spectrum": "Frequency spectrum shows overly tonal characteristics typical of synthesis",
    "unnaturally_consistent_bandwidth": "Spectral bandwidth remains unusually consistent (humans vary more)",
    "unnaturally_stable_spectral_centroid": "Spectral center of mass is abnormally stable",
    "unnaturally_clean_harmonics": "Harmonics are unusually clean (lacking natural noise)",
    "sharp_high_frequency_cutoff": "Sharp cutoff in high frequencies detected (vocoder signature)",
    # Temporal
    "unnaturally_smooth_energy_contour": "Energy envelope is too smooth (missing natural micro-variations)",
    "unnaturally_consistent_zero_crossings": "Zero-crossing rate is abnormally consistent",
    "metronomic_pause_timing": "Pauses between speech segments are evenly spaced (robotic timing)",
    "repetitive_energy_pattern": "Energy pattern repeats in a machine-like fashion",
    # Formant
    "unnaturally_stable_formant_structure": "Formant structure lacks natural speaker variation",
    "overly_smooth_formant_transitions": "Transitions between speech sounds are unnaturally smooth",
    "excessive_inter_frame_correlation": "Adjacent audio frames are excessively correlated",
    "uniform_mel_band_energy": "Energy distribution across frequency bands is unnaturally uniform",
    # Artifacts
    "synthesis_click_artifacts": "Click/pop artifacts typical of synthesis boundaries detected",
    "asymmetric_waveform": "Waveform shows unusual positive/negative asymmetry",
    "digitally_perfect_silence": "Silence segments are digitally perfect (zero noise floor)",
    "unnaturally_sharp_speech_boundaries": "Speech-to-silence transitions are unnaturally abrupt",
    "excessive_signal_periodicity": "Signal shows excessive periodicity beyond natural speech",
}


def generate_explanation(result: Dict[str, Any]) -> str:
    """
    Generate a human-readable explanation based on the full detection result.

    Args:
        result: The full result dict from VoiceDetector.predict()

    Returns:
        A clear, evidence-based explanation string.
    """
    classification = result["classification"]
    confidence = result["confidence"]
    agreement = result.get("analyzers_agree", True)
    artifacts = result.get("artifacts_summary", [])

    if classification == "AI_GENERATED":
        return _explain_ai(confidence, agreement, artifacts)
    else:
        return _explain_human(confidence, agreement, artifacts)


def _explain_ai(confidence: float, agreement: bool, artifacts: List[str]) -> str:
    """Generate explanation for AI_GENERATED verdict."""
    parts = []

    # Opening based on confidence
    if confidence > 0.90:
        parts.append("Strong indicators of AI-generated speech detected")
    elif confidence > 0.75:
        parts.append("Multiple indicators of synthetic voice generation detected")
    elif confidence > 0.60:
        parts.append("Moderate indicators of AI voice synthesis detected")
    else:
        parts.append("Weak indicators of possible AI generation detected")

    # Cite specific evidence
    if artifacts:
        evidence = _describe_top_artifacts(artifacts, max_items=3)
        if evidence:
            parts.append(f"Evidence: {evidence}")

    # Agreement note
    if agreement:
        parts.append("Neural model and forensic analyzers are in agreement")
    else:
        parts.append("Note: forensic analyzers show mixed signals — result may warrant manual review")

    return ". ".join(parts) + "."


def _explain_human(confidence: float, agreement: bool, artifacts: List[str]) -> str:
    """Generate explanation for HUMAN verdict."""
    parts = []

    if confidence > 0.90:
        parts.append("Natural human voice characteristics confirmed with high confidence")
    elif confidence > 0.75:
        parts.append("Authentic human speech patterns and natural variations detected")
    elif confidence > 0.60:
        parts.append("Voice exhibits predominantly human characteristics")
    else:
        parts.append("Voice classified as human with moderate confidence")

    # Note any artifacts still found (for transparency)
    if artifacts:
        parts.append(
            f"Minor note: {len(artifacts)} analysis flag(s) present but within normal range"
        )

    if agreement:
        parts.append("All analysis channels confirm this assessment")

    return ". ".join(parts) + "."


def _describe_top_artifacts(artifacts: List[str], max_items: int = 3) -> str:
    """Convert artifact codes to human-readable descriptions."""
    descriptions = []
    for artifact in artifacts[:max_items]:
        desc = ARTIFACT_DESCRIPTIONS.get(artifact, artifact.replace("_", " "))
        descriptions.append(desc.lower())

    if not descriptions:
        return ""

    if len(descriptions) == 1:
        return descriptions[0]
    elif len(descriptions) == 2:
        return f"{descriptions[0]} and {descriptions[1]}"
    else:
        return f"{', '.join(descriptions[:-1])}, and {descriptions[-1]}"
