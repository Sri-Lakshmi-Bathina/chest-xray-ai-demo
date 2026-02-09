import requests
import json
from typing import List

# ---------------------------
# Ollama configuration
# ---------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"

# Lightweight, free, local models suitable for this task
# You can change this to: "phi3:mini", "mistral:7b", etc.
OLLAMA_MODEL = "tinyllama"

# ---------------------------
# Safety-first system prompt
# ---------------------------
SYSTEM_PROMPT = """
You are an AI assistant explaining the behavior of a machine learning model
that analyzes chest X-ray images.

IMPORTANT RULES:
- This system is for EDUCATIONAL and RESEARCH purposes only.
- You must NOT provide medical diagnoses or medical advice.
- You must NOT suggest treatment or clinical decisions.
- Explain what the model focused on, NOT what a patient has.
- Use plain, non-alarming language.
"""

# ---------------------------
# Utility: check if Ollama is running
# ---------------------------
def ollama_available() -> bool:
    try:
        requests.get(OLLAMA_BASE_URL, timeout=2)
        return True
    except Exception:
        return False

# ---------------------------
# Local LLM explanation (Ollama)
# ---------------------------
def generate_llm_explanation(predictions, labels: List[str]) -> str:
    """
    Uses a local Ollama LLM to generate a plain-English explanation.
    This function should ONLY be called if ollama_available() is True.
    """

    findings = [
        f"{labels[i]} (confidence {predictions[i]:.2f})"
        for i in range(len(labels))
        if predictions[i] > 0.5
    ]

    findings_text = (
        ", ".join(findings)
        if findings
        else "No findings exceeded the confidence threshold."
    )

    prompt = f"""
{SYSTEM_PROMPT}

Model output:
{findings_text}

Task:
Explain what the highlighted regions in the Grad-CAM visualization represent.
Do NOT interpret this as a medical diagnosis.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(
        OLLAMA_GENERATE_URL,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=30
    )

    response.raise_for_status()
    return response.json().get("response", "").strip()

# ---------------------------
# Safe fallback (HF Spaces / no Ollama)
# ---------------------------
def fallback_explanation(predictions, labels: List[str]) -> str:
    """
    Deterministic, non-LLM explanation used when Ollama is unavailable
    (e.g., on Hugging Face Spaces).
    """

    high_level_findings = [
        labels[i]
        for i in range(len(labels))
        if predictions[i] > 0.5
    ]

    if not high_level_findings:
        return (
            "The highlighted regions indicate areas of the image that the model "
            "considered while making its prediction. No features strongly influenced "
            "the model above the confidence threshold."
        )

    return (
        "The highlighted regions show parts of the X-ray image that most influenced "
        "the model’s predictions for the following categories: "
        + ", ".join(high_level_findings)
        + ". These visualizations help understand model behavior and are not "
          "medical diagnoses."
    )
