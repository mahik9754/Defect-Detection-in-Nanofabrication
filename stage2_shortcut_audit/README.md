# Stage 2: Shortcut Learning Audit for SEM Defect Classification

## 1. Objective

The objective of this stage is to investigate whether the trained SEM defect classification model relies on information from the defect regions, the surrounding background, or both.

The audit compares model predictions on the same held-out test images under three input conditions: original images, defect-only images, and background-only images.

## 2. Dataset and Model

- **Dataset:** Carinthia-S SEM defect dataset
- **Evaluation split:** Existing held-out test split
- **Number of test images:** 688
- **Number of classes:** 6
- **Model:** Previously trained classification model checkpoint
- **Retraining:** Not performed during this audit

All three conditions use the same test images and ground-truth labels.

## 3. Experimental Conditions

### Original

The complete, unmodified SEM image is supplied to the model.

### Defect-only

Pixels inside the supplied defect mask are retained with their original grayscale intensities. Pixels outside the mask are set to black.

### Background-only

Pixels outside the supplied defect mask are retained with their original grayscale intensities. Pixels inside the mask are set to black.

The region extraction code uses a provisional mask convention in which nonzero mask pixels are treated as belonging to the selected region.

## 4. Evaluation Metrics

The audit calculates:

- **Accuracy:** Fraction of all predictions that are correct.
- **Balanced accuracy:** Average recall across classes, giving each class equal weight.
- **Macro F1-score:** Average F1-score across classes, giving each class equal weight.
- **Mean confidence:** Average maximum predicted-class probability.
- **Prediction change rate:** Fraction of test images whose predicted class changes relative to the original condition.
- **Per-class metrics:** Precision, recall, F1-score, and support for each class.

Confusion matrices are used to inspect class-specific prediction patterns.

## 5. Results

| Condition       | Accuracy | Balanced accuracy | Macro F1 | Mean confidence |
| --------------- | -------: | ----------------: | -------: | --------------: |
| Original        |   99.71% |            99.94% |   98.80% |          99.62% |
| Defect-only     |   12.06% |            60.42% |   48.23% |          99.50% |
| Background-only |   58.87% |            71.40% |   61.93% |          88.10% |

### Prediction changes relative to original images

| Condition       | Changed predictions | Prediction change rate |
| --------------- | ------------------: | ---------------------: |
| Defect-only     |           604 / 688 |                 87.79% |
| Background-only |           281 / 688 |                 40.84% |

### Key observations

1. The model performs very well on the original test images.
2. The defect-only condition produces a substantial performance decline, and 87.79% of predictions differ from the original predictions.
3. The background-only condition also produces a performance decline, with 40.84% of predictions changing.
4. The defect-only condition has a mean confidence of 99.50% despite an accuracy of 12.06%. This indicates that the model can remain highly confident when its predictions are incorrect on these transformed inputs.
5. In the confusion matrices, class 3 recall decreases from 99.67% for original images to 0% for defect-only images and 53.41% for background-only images.

## 6. Visualizations and Outputs

- `predictions.csv`: Predictions and class probabilities for each image under each condition.
- `shortcut_metrics.csv`: Overall metrics, per-class metrics, and prediction-change statistics.
- `plot_confusion_matrices.py`: Script for generating confusion matrices.
- `confusion_matrices.png`: Comparison of class-wise confusion matrices.
- `create_visualizations.py`: Script for visualizing region extraction.
- `create_example_predictions.py`: Script for visualizing predictions under the three conditions.
- `visualizations/region_comparison.png`: Side-by-side comparison of original, defect-only, and background-only images.
- `visualizations/example_predictions.png`: Example images with predicted classes and confidence scores.

## 7. Limitations

1. **Mask validity:** The nonzero-pixel convention is provisional. Visual inspection is useful, but the semantic correctness of the supplied masks should be verified before drawing strong conclusions.
2. **Distribution shift:** Masking creates images that differ from the original training distribution. Performance degradation may therefore reflect sensitivity to the transformation, not necessarily shortcut learning alone.
3. **Class imbalance:** The test set is heavily imbalanced, with class 3 accounting for 601 of the 688 images. Classes 2 and 5 each have only one test example, so their individual metrics are highly unstable.
4. **Confidence interpretation:** High confidence on transformed images does not establish that the model is calibrated or that a particular shortcut is being used.
5. **Artifact-focused condition:** No artifact-focused experiment is reported here because it requires a separately supplied artifact annotation mask.

## 8. Conclusion

The audit reveals substantial changes in model predictions when defect regions or surrounding backgrounds are masked. The original model performs strongly on the held-out test images, whereas both masked conditions show reduced performance and changes in predicted classes.

These findings indicate sensitivity to the input regions and motivate further investigation into the features used by the classifier. However, because masking can introduce distribution shift and the mask convention remains provisional, the results should be treated as evidence for further investigation rather than definitive proof of shortcut learning.
