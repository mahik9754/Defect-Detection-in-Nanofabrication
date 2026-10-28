# SEM-Based Wafer-Defect Classification with MobileNetV2
Project Overview:

This repository documents a four-stage investigation of a six-class SEM wafer-defect classification model using the Carinthia-S dataset. The stages cover data and mask preparation, shortcut-learning analysis, confidence calibration with human-review routing, and Grad-CAM interpretability and robustness testing.
Stages
1. Stage 1 — Data Pipeline, Masks, and Shared Inference
2. Stage 2 — Shortcut Learning Audit for SEM Defect Classification
3. Stage 3 — Confidence Calibration and Human-Review Routing
4. Stage 4 — Grad-CAM Interpretability and Robustness Analysis
The project reuses the existing trained model for the audit and calibration stages. Keep the original training pipeline, checkpoint, split assignments, and baseline results unchanged unless a documented compatibility issue requires a change.
Dataset
The project uses the Carinthia-S SEM defect dataset, available from Zenodo. The recorded manifest contains 4,591 images across six classes, split into 3,215 training, 688 validation, and 688 test images. Dataset files may need to be obtained separately and placed in the paths documented below.
Key recorded results
- Original-condition test accuracy in the Stage 2 audit: 99.71% on 688 images.
- Stage 2 masking experiments show substantial prediction changes for defect-only and background-only images; these results indicate sensitivity to the transformations but do not prove shortcut learning by themselves.
- Stage 3 selected temperature: 1.559644; selected review threshold: 0.90. Test accuracy remained 99.71%, while test NLL and ECE worsened after calibration.
- Stage 4 Grad-CAM overlap was limited. In the robustness subset, neither the original nor filtered model was consistently better across all tested transformations.
Stage 1 — Data Pipeline, Masks, and Shared Inference
1. Purpose
This module prepares the data and shared utilities for the SEM wafer-defect shortcut-learning audit.

It provides:
- Dataset and image-mask audit outputs.
- A manifest linking SEM images to their expert masks.
- Region extraction for original, defect-only, and background-only inputs.
- An inference interface returning the predicted class, confidence, and six-class probabilities.
- Reproduction instructions and documented limitations for the next project stage.
The original training pipeline and baseline results should remain unchanged.
2. Required project structure
Run the commands in this README from the project root, which contains config.py and preprocessing.py.

Expected structure:
Defect-Detection-in-Nanofabrication/
├── config.py
├── preprocessing.py
├── model_builder.py
├── train_model.py
├── dataset/
│   └── carinthia_s/
│       └── raw/
│           ├── images/
│           ├── masks/
│           └── carinthia-s.csv
└── stage1_data_pipeline/
    ├── __init__.py
    ├── audit_dataset.py
    ├── mask_utils.py
    ├── inference.py
    ├── region_extraction.py
    ├── data_audit.csv
    ├── mask_manifest.csv
    ├── duplicate_split_conflicts.csv
    ├── examples/
    └── README.md
The dataset and example files shown above may be excluded from the Git repository due to size or redistribution restrictions. If the dataset is not committed, each team member must obtain an authorized copy and place it in the documented location.

Dataset path distinction: the original project configuration uses dataset/raw/ for its baseline pipeline. The new Stage 1 audit and region-extraction commands use dataset/carinthia_s/raw/. Do not move or rename the original baseline data without coordinating with the team.
3. Dataset and manifest
The Carinthia-S CSV uses a semicolon (;) delimiter and contains the following columns:
- image_path
- mask_path
- filename
- label
The dataset audit generated the following files:
File	Purpose
data_audit.csv	Per-record audit results
mask_manifest.csv	Image-mask associations and mask metadata
duplicate_split_conflicts.csv	Duplicate entries potentially crossing data splits


Recorded audit summary
- CSV records: 4,591
- Audit status: 4,591 records marked ok
- Image-mask pairs aligned in dimensions: 4,591 / 4,591
- Exact duplicate image entries: 0
- Duplicate split conflicts: 0
- Empty masks: 226; all belong to class 6 in the audited data
- Full masks: 0
- Masks with intermediate grayscale values: 395
- Baseline split availability: train 3,215; validation 688; test 688
These are the recorded audit results. Empty masks are consistent with the class-6 no-defect examples; they should not automatically be treated as errors.
4. Region extraction
region_extraction.py generates controlled image inputs using an SEM image and its corresponding defect mask.
Conditions
- Original: the unchanged source image.
- Defect-only: retains pixels where the defect mask is nonzero; all other pixels are black.
- Background-only: retains pixels where the defect mask is zero; the masked region is black.
- Artifact-focused: retains pixels inside a separately supplied artifact mask. This requires an artifact annotation mask; the defect mask is not automatically treated as an artifact mask.
The current implementation interprets every nonzero mask pixel as foreground. Since some masks contain intermediate grayscale values, this convention remains provisional until the mask encoding and intended semantics are confirmed.
Command
From the project root, run:
python -m stage1_data_pipeline.region_extraction --image "dataset/carinthia_s/raw/images/0c8af9b8e62f4cb498392e806236e5e0.jpg" --mask "dataset/carinthia_s/raw/masks/0c8af9b8e62f4cb498392e806236e5e0.png"
Replace the example filenames with the image and corresponding mask you want to process.

The script saves three PNG files in:
stage1_data_pipeline/region_examples/
The output filenames end with _original.png, _defect_only.png, and _background_only.png.

An optional artifact mask can be supplied with --artifact-mask, provided it is a separate, correctly aligned annotation:
python -m stage1_data_pipeline.region_extraction --image "PATH_TO_IMAGE" --mask "PATH_TO_DEFECT_MASK" --artifact-mask "PATH_TO_ARTIFACT_MASK"
Masked-out pixels are black. This can introduce distribution shift, so changes in model predictions should not be interpreted as proof of causal shortcut reliance.
5. Shared inference interface
inference.py provides a common interface for loading a trained model and predicting one SEM image.

The model checkpoint and run configuration must belong to the same training run. The run configuration must contain the recorded six-class mapping.

The interface returns:
- Predicted class index and class name.
- Predicted-class confidence.
- Probabilities for all six classes.
- The preprocessing condition used.
The function reuses the existing preprocessing.py image-processing function. The model handles its own MobileNetV2 rescaling.
Example usage
Run this from the project root, using a checkpoint and run-configuration file that actually exist:
from stage1_data_pipeline.inference import (
    load_inference_model,
    predict_image,
)

checkpoint = "models/original/original_seed42.keras"
run_config = "results/original/original_seed42_run_config.json"

model, class_names = load_inference_model(
    checkpoint,
    run_config,
)

result = predict_image(
    "dataset/carinthia_s/raw/images/0c8af9b8e62f4cb498392e806236e5e0.jpg",
    model,
    class_names,
)

print(result)
The checkpoint paths above are examples based on the project's naming convention. Confirm that the selected files exist before running the example.

A successful region-extraction run has been recorded. End-to-end inference with a matching checkpoint and run configuration must also be tested before claiming that inference compatibility has been fully validated.
6. Reproducibility and split discipline
- Preserve the existing baseline scripts, checkpoints, and results.
- Do not change the train/validation/test assignments for this handoff.
- Use identical image IDs and labels when comparing original and masked conditions.
- Record the checkpoint, run configuration, class mapping, preprocessing, and split used for every experiment.
- Fit calibration parameters and select review thresholds using validation data only.
- Reserve test data for final evaluation.
- Do not retrain or alter the baseline unless a documented compatibility issue requires it.
7. Known limitations
1. Mask semantics: nonzero pixels are currently treated as foreground. Intermediate grayscale values exist, so this interpretation needs confirmation.
2. Artifact annotations: artifact-focused extraction requires a separate artifact mask; it is not produced from the defect mask.
3. Distribution shift: blacked-out regions can change image statistics and model behaviour.
4. Inference compatibility: use a checkpoint and run configuration from the same training run.
5. Dataset availability: the dataset is expected at dataset/carinthia_s/raw/, but may need to be obtained separately if it is not included in the repository.
6. Scientific interpretation: prediction changes after masking are evidence to investigate, not proof that the model causally relies on a particular feature.
8. Handoff to Member 2
Member 2 should receive:
- mask_utils.py
- inference.py
- region_extraction.py
- audit_dataset.py
- data_audit.csv
- mask_manifest.csv
- duplicate_split_conflicts.csv
- This README
- Representative mask-overlay and region-extraction examples, where available
Member 2 must also have access to the dataset, a compatible trained checkpoint, and its matching run configuration.

The next stage can use this interface to compare predictions for original, defect-only, and background-only images. Artifact-focused comparisons should be added only when a separate artifact-region definition or annotation is available.

Current status: dataset audit completed; region extraction successfully executed on one real image-mask pair; inference interface created; artifact-focused extraction requires a separate mask; mask semantics and end-to-end inference compatibility remain documented validation items.

# Stage 2: Shortcut Learning Audit for SEM Defect Classification

## 1. Objective

The objective of this stage is to investigate whether the trained SEM defect classification model relies on information from the defect regions, the surrounding background, or both.

The audit compares model predictions on the same held-out test images under three input conditions: original images, defect-only images, and background-only images.

## 2. Dataset and Model
- Dataset: Carinthia-S SEM defect dataset
- Evaluation split: Existing held-out test split
- Number of test images: 688
- Number of classes: 6
- Model: Previously trained classification model checkpoint
- Retraining: Not performed during this audit
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
- Accuracy: Fraction of all predictions that are correct.
- Balanced accuracy: Average recall across classes, giving each class equal weight.
- Macro F1-score: Average F1-score across classes, giving each class equal weight.
- Mean confidence: Average maximum predicted-class probability.
- Prediction change rate: Fraction of test images whose predicted class changes relative to the original condition.
- Per-class metrics: Precision, recall, F1-score, and support for each class.
Confusion matrices are used to inspect class-specific prediction patterns.

## 5. Results
Condition	Accuracy	Balanced accuracy	Macro F1	Mean confidence
Original	99.71%	99.94%	98.80%	99.62%
Defect-only	12.06%	60.42%	48.23%	99.50%
Background-only	58.87%	71.40%	61.93%	88.10%


### Prediction changes relative to original images
Condition	Changed predictions	Prediction change rate
Defect-only	604 / 688	87.79%
Background-only	281 / 688	40.84%


### Key observations

1. The model performs very well on the original test images.
2. The defect-only condition produces a substantial performance decline, and 87.79% of predictions differ from the original predictions.
3. The background-only condition also produces a performance decline, with 40.84% of predictions changing.
4. The defect-only condition has a mean confidence of 99.50% despite an accuracy of 12.06%. This indicates that the model can remain highly confident when its predictions are incorrect on these transformed inputs.
5. In the confusion matrices, class 3 recall decreases from 99.67% for original images to 0% for defect-only images and 53.41% for background-only images.

## 6. Visualizations and Outputs
- predictions.csv: Predictions and class probabilities for each image under each condition.
- shortcut_metrics.csv: Overall metrics, per-class metrics, and prediction-change statistics.
- plot_confusion_matrices.py: Script for generating confusion matrices.
- confusion_matrices.png: Comparison of class-wise confusion matrices.
- create_visualizations.py: Script for visualizing region extraction.
- create_example_predictions.py: Script for visualizing predictions under the three conditions.
- visualizations/region_comparison.png: Side-by-side comparison of original, defect-only, and background-only images.
- visualizations/example_predictions.png: Example images with predicted classes and confidence scores.
## 7. Limitations

1. Mask validity: The nonzero-pixel convention is provisional. Visual inspection is useful, but the semantic correctness of the supplied masks should be verified before drawing strong conclusions.
2. Distribution shift: Masking creates images that differ from the original training distribution. Performance degradation may therefore reflect sensitivity to the transformation, not necessarily shortcut learning alone.
3. Class imbalance: The test set is heavily imbalanced, with class 3 accounting for 601 of the 688 images. Classes 2 and 5 each have only one test example, so their individual metrics are highly unstable.
4. Confidence interpretation: High confidence on transformed images does not establish that the model is calibrated or that a particular shortcut is being used.
5. Artifact-focused condition: No artifact-focused experiment is reported here because it requires a separately supplied artifact annotation mask.

## 8. Conclusion

The audit reveals substantial changes in model predictions when defect regions or surrounding backgrounds are masked. The original model performs strongly on the held-out test images, whereas both masked conditions show reduced performance and changes in predicted classes.

These findings indicate sensitivity to the input regions and motivate further investigation into the features used by the classifier. However, because masking can introduce distribution shift and the mask convention remains provisional, the results should be treated as evidence for further investigation rather than definitive proof of shortcut learning.

# Stage 3: Confidence Calibration and Human-Review Routing

## Overview

Stage 3 adds confidence calibration and a human-review routing mechanism to the existing SEM wafer-defect classification model.

The existing MobileNetV2 checkpoint is reused without retraining. Temperature scaling adjusts the model's output probabilities, while a confidence threshold determines which predictions are accepted automatically and which are flagged for human review.

## Methodology

### 1. Temperature scaling

Temperature scaling is fitted using the validation split only. The model's pre-softmax logits are reconstructed from the features entering its final Dense layer and that layer's weights and bias.

The scaled probabilities are calculated as:

softmax(logits / temperature)

A temperature is selected by minimizing validation negative log-likelihood (NLL). The search evaluates 500 candidate temperatures between 0.05 and 10.0.

Selected temperature: 1.559644

### 2. Human-review routing

The calibrated prediction confidence is the maximum probability across the six classes.
- Confidence greater than or equal to 0.90: accept the prediction automatically.
- Confidence below 0.90: send the prediction for human review.
The threshold is selected using validation data, minimizing accepted-prediction error among candidate thresholds that retain at least 80% validation coverage. Ties prefer higher coverage.

The selected threshold is 0.90.

### 3. Evaluation protocol

The validation split is used to fit the temperature and select the review threshold. The held-out test split is used only for final evaluation. The selected settings are kept fixed during test evaluation.

The model checkpoint, class mapping, and existing processed images are reused.

## Validation Results
Metric	Before calibration	After calibration
NLL	0.028821	0.023814
ECE (15 bins)	0.007576	0.004368


Validation routing results:
- Validation images: 688
- Selected threshold: 0.90
- Automatically accepted: 677
- Sent for review: 11
- Coverage: 98.40%
- Accepted-prediction error: approximately 0.15%
## Held-Out Test Results
Metric	Raw model	Calibrated model
Accuracy	99.71%	99.71%
NLL	0.009493	0.012464
ECE (15 bins)	0.003227	0.005473


Test routing results:
- Test images: 688
- Automatically accepted: 670
- Sent for review: 18
- Automatic-acceptance coverage: 97.38%
- Accepted-prediction error: 0 observed errors among 670 accepted predictions
Although accuracy remained unchanged, test NLL and ECE increased after temperature scaling. The calibration improvement observed on validation data therefore did not carry over to these test calibration metrics. This limitation is reported rather than treating validation improvement as a guaranteed test improvement.

The zero observed errors among accepted test predictions is a result for this test sample, not a guarantee of future performance.

## Generated Files
- fit_temperature.py — fits temperature scaling and selects the review threshold.
- calibration.py — implements the temperature-scaled softmax.
- review_router.py — routes predictions according to confidence.
- evaluate_test.py — evaluates the frozen calibration settings on test data.
- calibration_config.json — stores the selected temperature, threshold, and validation metrics.
- calibration_metrics.csv — stores calibration and validation-routing metrics.
- threshold_candidates.csv — records validation coverage and accepted-prediction error for candidate thresholds.
- reliability_diagram.png — compares confidence and accuracy before and after calibration on validation data.
- coverage_error_curve.png — plots validation coverage against accepted-prediction error.
- test_evaluation_metrics.csv — records final test metrics.
## Reproducibility

Run the following commands from the project root using the configured virtual environment:
.\\.venv\python.exe .\stage3_calibration\fit_temperature.py
.\\.venv\python.exe .\stage3_calibration\evaluate_test.py
The first command fits calibration settings from validation data and generates the validation plots. The second applies the saved temperature and threshold to the held-out test set.

## Limitations
- Calibration results depend on the validation distribution.
- Confidence is not a guarantee that a prediction is correct.
- The 80% minimum validation coverage is a design constraint chosen for threshold selection.
- Test calibration metrics worsened after temperature scaling in this evaluation.
- Human-review routing flags uncertain predictions; it does not itself perform human review.
# Stage 4: Grad-CAM Interpretability and Robustness Analysis

## Overview

Stage 4 investigates how the nanofabrication defect-classification model makes predictions and how its performance changes under different image conditions.

The experiments focus on three aspects:

1. Grad-CAM interpretability: Visualizing image regions that contribute to model predictions.
2. Defect-mask evaluation: Measuring the overlap between Grad-CAM heatmaps and annotated defect masks.
3. Robustness analysis: Comparing the original and filtered models under blur, contrast changes, and noise.

## Experiments

### 1. Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) generates heatmaps highlighting image regions that influence a model's predicted class.

The experiments include:
- Grad-CAM visualizations for selected test images.
- Comparisons between original images, defect-only images, and background-only images.
- Verification that Grad-CAM uses the intended model predictions.
Purpose: To investigate whether the model's attention aligns with visible defects or includes substantial background information.

### 2. Defect-Mask Evaluation

Grad-CAM heatmaps are compared against annotated defect masks using Intersection over Union (IoU) and Dice coefficient.
Heatmap threshold	Mean IoU	Mean Dice
0.2	0.0951	0.1604
0.3	0.1051	0.1754
0.4	0.1212	0.1982
0.5	0.1396	0.2233
0.6	0.1617	0.2527


The highest overlap among the tested thresholds occurred at 0.6, with a mean IoU of approximately 0.162 and mean Dice of approximately 0.253 across 267 evaluated images.

Interpretation: The overlap between Grad-CAM regions and annotated defect masks was limited. This suggests that the highlighted regions do not closely match the annotated defects. However, Grad-CAM is a coarse explanation method, so low overlap alone does not prove that the model relies on shortcuts or ignores defects.

### 3. Robustness Comparison

The original and filtered models were evaluated on 170 images per condition.
Condition	Original accuracy	Filtered accuracy
Clean	100.00%	99.41%
Blur (radius 2)	95.29%	96.47%
Contrast (0.5)	98.82%	98.82%
Contrast (1.5)	98.82%	99.41%
Noise (standard deviation 15)	94.12%	92.35%


Observations:
- Both models maintained relatively high accuracy across the tested conditions.
- The filtered model performed better under blur and increased contrast.
- Both models performed equally under reduced contrast.
- The original model performed better under noise.
- The differences were generally small, and neither model was consistently superior.
These findings apply to the tested subset and transformations; they should not be generalized to all images or real-world conditions.

## Project Structure
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
## Running the Experiments

Run commands from the repository root after activating the project environment and ensuring that the required dataset and model files are available.
python stage4_gradcam_robustness/gradcam.py
python stage4_gradcam_robustness/gradcam_three.py
python stage4_gradcam_robustness/mask_evaluation.py
python stage4_gradcam_robustness/robustness_original.py
python stage4_gradcam_robustness/robustness_filtered.py
python stage4_gradcam_robustness/compare_robustness.py
The scripts may require paths or configuration values to match your local environment. Run check_masks.py and verify_label_masks.py when checking mask availability and label-mask consistency.

## Limitations
- Grad-CAM provides approximate visual explanations, not definitive proof of model reasoning.
- Low mask overlap may be affected by heatmap resolution, threshold selection, and annotation quality.
- The robustness experiments use a limited subset of 170 images per condition.
- Artificial blur, contrast changes, and noise do not represent every real-world imaging variation.
- Class imbalance and differences between defect classes may affect the evaluation.
## Conclusion

Stage 4 evaluates model interpretability and robustness using Grad-CAM visualizations, annotated defect masks, and controlled image transformations. The robustness results show that both models performed well on the tested conditions, with small differences depending on the transformation. The limited Grad-CAM-to-mask overlap motivates further investigation into model attention and defect localization.

These results complement the classification metrics from earlier project stages and provide additional evidence for evaluating the reliability and interpretability of the defect-classification system.
