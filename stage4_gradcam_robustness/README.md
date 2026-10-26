# Stage 4: Grad-CAM Interpretability and Robustness Analysis

## Overview

Stage 4 investigates how the nanofabrication defect-classification model makes predictions and how its performance changes under different image conditions.

The experiments focus on three aspects:

1. **Grad-CAM interpretability:** Visualizing image regions that contribute to model predictions.
2. **Defect-mask evaluation:** Measuring the overlap between Grad-CAM heatmaps and annotated defect masks.
3. **Robustness analysis:** Comparing the original and filtered models under blur, contrast changes, and noise.

## Experiments

### 1. Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) generates heatmaps highlighting image regions that influence a model's predicted class.

The experiments include:
- Grad-CAM visualizations for selected test images.
- Comparisons between original images, defect-only images, and background-only images.
- Verification that Grad-CAM uses the intended model predictions.

**Purpose:** To investigate whether the model's attention aligns with visible defects or includes substantial background information.

### 2. Defect-Mask Evaluation

Grad-CAM heatmaps are compared against annotated defect masks using Intersection over Union (IoU) and Dice coefficient.

| Heatmap threshold | Mean IoU | Mean Dice |
|---:|---:|---:|
| 0.2 | 0.0951 | 0.1604 |
| 0.3 | 0.1051 | 0.1754 |
| 0.4 | 0.1212 | 0.1982 |
| 0.5 | 0.1396 | 0.2233 |
| 0.6 | 0.1617 | 0.2527 |

The highest overlap among the tested thresholds occurred at 0.6, with a mean IoU of approximately 0.162 and mean Dice of approximately 0.253 across 267 evaluated images.

**Interpretation:** The overlap between Grad-CAM regions and annotated defect masks was limited. This suggests that the highlighted regions do not closely match the annotated defects. However, Grad-CAM is a coarse explanation method, so low overlap alone does not prove that the model relies on shortcuts or ignores defects.

### 3. Robustness Comparison

The original and filtered models were evaluated on 170 images per condition.

| Condition | Original accuracy | Filtered accuracy |
|---|---:|---:|
| Clean | 100.00% | 99.41% |
| Blur (radius 2) | 95.29% | 96.47% |
| Contrast (0.5) | 98.82% | 98.82% |
| Contrast (1.5) | 98.82% | 99.41% |
| Noise (standard deviation 15) | 94.12% | 92.35% |

**Observations:**
- Both models maintained relatively high accuracy across the tested conditions.
- The filtered model performed better under blur and increased contrast.
- Both models performed equally under reduced contrast.
- The original model performed better under noise.
- The differences were generally small, and neither model was consistently superior.

These findings apply to the tested subset and transformations; they should not be generalized to all images or real-world conditions.

## Project Structure

```text
stage4_gradcam_robustness/
├── check_masks.py
├── compare_robustness.py
├── gradcam.py
├── gradcam_three.py
├── mask_evaluation.py
├── robustness_filtered.py
├── robustness_original.py
├── run_analysis.py
├── verify_label_masks.py
└── results/
    ├── comparison/
    ├── gradcam/
    ├── gradcam_three_conditions/
    ├── mask_evaluation/
    ├── mask_evaluation_corrected/
    ├── robustness_filtered/
    └── robustness_original/
```

## Running the Experiments

Run commands from the repository root after activating the project environment and ensuring that the required dataset and model files are available.

```bash
python stage4_gradcam_robustness/gradcam.py
python stage4_gradcam_robustness/gradcam_three.py
python stage4_gradcam_robustness/mask_evaluation.py
python stage4_gradcam_robustness/robustness_original.py
python stage4_gradcam_robustness/robustness_filtered.py
python stage4_gradcam_robustness/compare_robustness.py
```

The scripts may require paths or configuration values to match your local environment. Run `check_masks.py` and `verify_label_masks.py` when checking mask availability and label-mask consistency.

## Limitations

- Grad-CAM provides approximate visual explanations, not definitive proof of model reasoning.
- Low mask overlap may be affected by heatmap resolution, threshold selection, and annotation quality.
- The robustness experiments use a limited subset of 170 images per condition.
- Artificial blur, contrast changes, and noise do not represent every real-world imaging variation.
- Class imbalance and differences between defect classes may affect the evaluation.

## Conclusion

Stage 4 evaluates model interpretability and robustness using Grad-CAM visualizations, annotated defect masks, and controlled image transformations. The robustness results show that both models performed well on the tested conditions, with small differences depending on the transformation. The limited Grad-CAM-to-mask overlap motivates further investigation into model attention and defect localization.

These results complement the classification metrics from earlier project stages and provide additional evidence for evaluating the reliability and interpretability of the defect-classification system.
