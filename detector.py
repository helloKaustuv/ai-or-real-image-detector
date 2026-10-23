"""
Core detection logic — kept separate from the website so it can be
tested from the command line and reused anywhere.

Model: dima806/deepfake_vs_real_image_detection (Hugging Face)
A Vision Transformer fine-tuned for binary Real vs Fake/AI-generated
image classification. Loaded once and cached by the caller.
"""

from PIL import Image

MODEL_ID = "dima806/deepfake_vs_real_image_detection"

# Words that mean "this label is the AI / fake class", whatever exact
# wording a given checkpoint uses ("Fake", "deepfake", "AI", ...).
_FAKE_WORDS = ("fake", "deepfake", "ai", "generated", "artificial", "synthetic")


def load_detector():
    """Load the pretrained classifier. First run downloads ~350 MB."""
    from transformers import pipeline

    return pipeline("image-classification", model=MODEL_ID)


def _is_fake_label(label: str) -> bool:
    text = label.lower()
    return any(word in text for word in _FAKE_WORDS)


def classify(detector, image: Image.Image) -> dict:
    """
    Classify one PIL image.

    Returns:
        {
            "verdict": "AI-generated" or "Real",
            "confidence": float between 0 and 1 (for the verdict),
            "scores": {"AI-generated": float, "Real": float},
        }
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    raw = detector(image)  # [{"label": ..., "score": ...}, ...]

    scores = {"AI-generated": 0.0, "Real": 0.0}
    for item in raw:
        key = "AI-generated" if _is_fake_label(item["label"]) else "Real"
        scores[key] = max(scores[key], float(item["score"]))

    # Normalise in case a checkpoint only returns the top label.
    total = scores["AI-generated"] + scores["Real"]
    if total > 0:
        scores = {k: v / total for k, v in scores.items()}

    verdict = "AI-generated" if scores["AI-generated"] >= scores["Real"] else "Real"
    return {
        "verdict": verdict,
        "confidence": scores[verdict],
        "scores": scores,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python detector.py <image-file>")
        raise SystemExit(1)

    print(f"Loading model ({MODEL_ID}) — first run downloads it...")
    det = load_detector()
    result = classify(det, Image.open(sys.argv[1]))
    print(f"Verdict:    {result['verdict']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(
        f"Scores:     AI-generated {result['scores']['AI-generated']:.1%} | "
        f"Real {result['scores']['Real']:.1%}"
    )
