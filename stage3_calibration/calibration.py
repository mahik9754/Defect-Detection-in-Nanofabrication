
import numpy as np


def softmax(logits, temperature=1.0):
    """Convert logits into probabilities using temperature scaling."""
    logits = np.asarray(logits, dtype=np.float64)

    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    scaled_logits = logits / temperature
    scaled_logits -= np.max(scaled_logits, axis=-1, keepdims=True)

    exp_values = np.exp(scaled_logits)
    probabilities = exp_values / np.sum(
        exp_values, axis=-1, keepdims=True
    )

    return probabilities
