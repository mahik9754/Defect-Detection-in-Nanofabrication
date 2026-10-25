# Stage 1 — Data Integrity, Expert Masks, and Shared Inference

## 1. Purpose

This module supports the SEM-Based Wafer-Defect Classification project by preparing the Carinthia-S image-mask data for subsequent region-based experiments.

It provides dataset audit results, image-mask metadata, reusable mask utilities, and a common inference interface for the existing six-class MobileNetV2 classifier.

The original baseline training pipeline, checkpoints, results, class mapping, and train/validation/test assignments must be preserved.

## 2. Project Structure

```text
stage1_data_pipeline/
├── __init__.py
├── audit_dataset.py
├── mask_utils.py
├── inference.py
├── data_audit.csv
├── mask_manifest.csv
├── duplicate_split_conflicts.csv
├── examples/
└── README.md
```

The existing project files remain in the repository root, including `config.py`, `preprocessing.py`, `model_builder.py`, and `train_model.py`.

## 3. Dataset

The additional dataset is Carinthia-S, stored at:

```text
dataset/carinthia_s/raw/
├── images/
├── masks/
└── carinthia-s.csv
```

The metadata CSV is semicolon-delimited and contains the fields `image_path`, `mask_path`, `filename`, and `label`.

The dataset contains 4,591 image-mask records across six labels.

## 4. Data Audit

Run the audit from the project root:

```bash
python -m stage1_data_pipeline.audit_dataset
```

The audit generates:

- `data_audit.csv` — record-level validation results.
- `mask_manifest.csv` — image-mask dimensions, mask statistics, hashes, labels, and baseline split assignments.
- `duplicate_split_conflicts.csv` — detected exact-image duplicates assigned to different baseline splits.
- `examples/` — representative image, mask, and overlay visualizations.

### Recorded audit results

| Check | Result |
|---|---:|
| Metadata records | 4,591 |
| Records passing structural audit | 4,591 |
| Aligned image-mask pairs | 4,591 |
| Exact duplicate image entries | 0 |
| Duplicate split conflicts | 0 |
| Empty masks | 226 |
| Masks with intermediate grayscale values | 395 |

All 226 empty masks belong to class 6, which represents the no-defect class in this project.

The existing baseline split availability is:

| Split | Images |
|---|---:|
| Training | 3,215 |
| Validation | 688 |
| Test | 688 |

These split assignments must not be changed for the new experiments.

**Interpretation:** The audit establishes structural consistency, readability, and image-mask dimension alignment. It does not, by itself, prove that every mask correctly identifies a defect or establish the semantic meaning of every pixel value.

## 5. Mask Handling

`mask_utils.py` contains the reusable mask-loading and validation utilities.

The audit records empty masks, unique pixel values, intermediate grayscale values, and foreground statistics.

An empty mask is not automatically considered invalid because class 6 represents no defect.

The current nonzero-pixel foreground interpretation is provisional until the dataset's annotation convention is confirmed. Intermediate grayscale values must be handled consistently with the documented mask encoding before region-based experiments are interpreted scientifically.

## 6. Shared Model Inference

`inference.py` provides a common interface for loading the existing trained classifier and predicting an SEM image.

Example:

```python
from stage1_data_pipeline.inference import (
    load_inference_model,
    predict_image,
)

model, class_names = load_inference_model(
    "models/original/original_seed42.keras",
    "results/original/original_seed42_run_config.json",
)

result = predict_image(
    "path/to/sem_image.png",
    model,
    class_names,
    filtered=False,
)

print(result["predicted_class"])
print(result["confidence"])
print(result["probabilities"])
```

Replace the example image path with an existing SEM image.

The model checkpoint and run configuration must belong to the same training run. The configuration supplies the class mapping. The shared interface reuses the project's existing `process_image` function, including grayscale conversion, resizing, RGB conversion, and optional median filtering.

The returned dictionary contains the predicted class, predicted-class confidence, all six class probabilities, and the preprocessing version.

Confidence is the highest model probability and must not be interpreted as a guarantee of correctness.

## 7. Region-Based Experiments

The next stage uses the same images, checkpoint, class mapping, and preprocessing to compare:

1. Original images.
2. Defect-only images.
3. Background-only images.
4. Artifact-focused images.

Region extraction must use the verified mask convention and preserve image-mask alignment. Artificially blacked-out regions can introduce distribution shift, so performance changes should be interpreted cautiously.

The region-extraction implementation and its reproducible example must be completed before this handoff is marked fully accepted.

## 8. Reproducibility

For each experiment, record:

- Model checkpoint and training run.
- Class mapping.
- Image IDs and split.
- Preprocessing version.
- Mask interpretation and region-extraction parameters.
- Random seeds, where applicable.
- Output filenames and metric definitions.

The test set must remain reserved for evaluation. It must not be used to fit calibration parameters or select confidence thresholds.

## 9. Limitations

- Structural mask validation does not establish annotation correctness.
- Empty masks are consistent with the no-defect class, but other mask exceptions require appropriate interpretation.
- Intermediate grayscale values may encode more than a simple binary region.
- Region masking can create artificial image patterns and distribution shift.
- Model predictions and confidence scores are decision-support outputs, not substitutes for expert inspection.

## 10. Handoff Status

**Completed:** Dataset audit, mask manifest generation, image-mask dimension checks, duplicate checks, baseline split recording, and reusable mask utilities.

**Inference interface:** Implemented; a reproducible prediction example and compatibility with the selected checkpoint must be confirmed.

**Remaining before full handoff:** Region extraction for the agreed experimental conditions, a documented example, and a successful end-to-end inference sanity check.

The receiving member should be able to run the documented example without changing the original baseline pipeline.
