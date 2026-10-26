
import numpy as np


def route_for_review(probabilities, threshold):
    """
    Route uncertain predictions for human review.

    A prediction is sent for review when its maximum class
    probability is below the selected threshold.
    """
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 2 or probabilities.shape[1] != 6:
        raise ValueError("Expected probabilities with shape (N, 6).")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities must be finite.")

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Probabilities must be between 0 and 1.")

    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Each row must sum to 1.")

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    confidence = np.max(probabilities, axis=1)
    predicted_class = np.argmax(probabilities, axis=1)
    needs_review = confidence < threshold

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "needs_review": needs_review,
    }
