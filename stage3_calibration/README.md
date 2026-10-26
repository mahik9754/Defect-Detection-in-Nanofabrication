# Stage 3: Confidence Calibration and Human-Review Routing

## Overview

Stage 3 adds confidence calibration and a human-review routing mechanism to the existing SEM wafer-defect classification model.

The existing MobileNetV2 checkpoint is reused without retraining. Temperature scaling adjusts the model's output probabilities, while a confidence threshold determines which predictions are accepted automatically and which are flagged for human review.

## Methodology

### 1. Temperature scaling

Temperature scaling is fitted using the validation split only. The model's pre-softmax logits are reconstructed from the features entering its final Dense layer and that layer's weights and bias.

The scaled probabilities are calculated as:

`softmax(logits / temperature)`

A temperature is selected by minimizing validation negative log-likelihood (NLL). The search evaluates 500 candidate temperatures between 0.05 and 10.0.

Selected temperature: **1.559644**

### 2. Human-review routing

The calibrated prediction confidence is the maximum probability across the six classes.

- Confidence greater than or equal to 0.90: accept the prediction automatically.
- Confidence below 0.90: send the prediction for human review.

The threshold is selected using validation data, minimizing accepted-prediction error among candidate thresholds that retain at least 80% validation coverage. Ties prefer higher coverage.

The selected threshold is **0.90**.

### 3. Evaluation protocol

The validation split is used to fit the temperature and select the review threshold. The held-out test split is used only for final evaluation. The selected settings are kept fixed during test evaluation.

The model checkpoint, class mapping, and existing processed images are reused.

## Validation Results

| Metric | Before calibration | After calibration |
|---|---:|---:|
| NLL | 0.028821 | 0.023814 |
| ECE (15 bins) | 0.007576 | 0.004368 |

Validation routing results:

- Validation images: 688
- Selected threshold: 0.90
- Automatically accepted: 677
- Sent for review: 11
- Coverage: 98.40%
- Accepted-prediction error: approximately 0.15%

## Held-Out Test Results

| Metric | Raw model | Calibrated model |
|---|---:|---:|
| Accuracy | 99.71% | 99.71% |
| NLL | 0.009493 | 0.012464 |
| ECE (15 bins) | 0.003227 | 0.005473 |

Test routing results:

- Test images: 688
- Automatically accepted: 670
- Sent for review: 18
- Automatic-acceptance coverage: 97.38%
- Accepted-prediction error: 0 observed errors among 670 accepted predictions

Although accuracy remained unchanged, test NLL and ECE increased after temperature scaling. The calibration improvement observed on validation data therefore did not carry over to these test calibration metrics. This limitation is reported rather than treating validation improvement as a guaranteed test improvement.

The zero observed errors among accepted test predictions is a result for this test sample, not a guarantee of future performance.

## Generated Files

- `fit_temperature.py` — fits temperature scaling and selects the review threshold.
- `calibration.py` — implements the temperature-scaled softmax.
- `review_router.py` — routes predictions according to confidence.
- `evaluate_test.py` — evaluates the frozen calibration settings on test data.
- `calibration_config.json` — stores the selected temperature, threshold, and validation metrics.
- `calibration_metrics.csv` — stores calibration and validation-routing metrics.
- `threshold_candidates.csv` — records validation coverage and accepted-prediction error for candidate thresholds.
- `reliability_diagram.png` — compares confidence and accuracy before and after calibration on validation data.
- `coverage_error_curve.png` — plots validation coverage against accepted-prediction error.
- `test_evaluation_metrics.csv` — records final test metrics.

## Reproducibility

Run the following commands from the project root using the configured virtual environment:

```powershell
.\.venv\python.exe .\stage3_calibration\fit_temperature.py
.\.venv\python.exe .\stage3_calibration\evaluate_test.py
```

The first command fits calibration settings from validation data and generates the validation plots. The second applies the saved temperature and threshold to the held-out test set.

## Limitations

- Calibration results depend on the validation distribution.
- Confidence is not a guarantee that a prediction is correct.
- The 80% minimum validation coverage is a design constraint chosen for threshold selection.
- Test calibration metrics worsened after temperature scaling in this evaluation.
- Human-review routing flags uncertain predictions; it does not itself perform human review.
