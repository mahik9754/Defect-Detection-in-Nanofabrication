
import numpy as np

from calibration import softmax
from review_router import route_for_review


def test_temperature_scaling():
    logits = np.array([[2.0, 1.0, 0.0, -1.0, -2.0, -3.0]])

    probabilities = softmax(logits, temperature=2.0)

    assert probabilities.shape == (1, 6)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_review_routing():
    probabilities = np.array([
        [0.90, 0.02, 0.02, 0.02, 0.02, 0.02],
        [0.10, 0.15, 0.20, 0.40, 0.10, 0.05],
    ])

    result = route_for_review(probabilities, threshold=0.60)

    assert result["predicted_class"].tolist() == [0, 3]
    assert result["needs_review"].tolist() == [False, True]


def test_threshold_boundary():
    probabilities = np.array([
        [0.60, 0.08, 0.08, 0.08, 0.08, 0.08]
    ])

    result = route_for_review(probabilities, threshold=0.60)

    assert result["needs_review"].tolist() == [False]


if __name__ == "__main__":
    test_temperature_scaling()
    test_review_routing()
    test_threshold_boundary()
    print("All Stage 3 tests passed.")
