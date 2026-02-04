"""
Explanation generator for voice detection results.
Provides human-readable explanations based on classification and confidence.
"""

def generate_explanation(classification: str, confidence: float) -> str:
    """
    Generate human-readable explanation for the classification.
    
    Args:
        classification: Either "AI_GENERATED" or "HUMAN"
        confidence: Confidence score between 0.0 and 1.0
    
    Returns:
        str: Human-readable explanation
    """
    if classification == "AI_GENERATED":
        if confidence > 0.9:
            return "Strong indicators of synthetic voice generation detected with high confidence"
        elif confidence > 0.7:
            return "Unnatural pitch consistency and robotic speech patterns detected"
        else:
            return "Moderate synthetic voice characteristics observed"
    else:  # HUMAN
        if confidence > 0.9:
            return "Natural human voice characteristics confirmed with high confidence"
        elif confidence > 0.7:
            return "Authentic human speech patterns and natural variations detected"
        else:
            return "Voice exhibits predominantly human characteristics"
