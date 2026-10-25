# Stage 1 — Data Pipeline, Masks, and Shared Inference

## 1. Purpose

This module prepares the data and shared utilities for the SEM wafer-defect shortcut-learning audit.

It provides:
- Dataset and image-mask audit outputs.
- A manifest linking SEM images to their expert masks.
- Region extraction for original, defect-only, and background-only inputs.
- An inference interface returning the predicted class, confidence, and six-class probabilities.
- Reproduction instructions and documented limitations for the next project stage.

The original training pipeline and baseline results should remain unchanged.

## 2. Required project structure

Run the commands in this README from the **project root**, which contains `config.py` and `preprocessing.py`.

Expected structure:

```text
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
```

The dataset and example files shown above may be excluded from the Git repository due to size or redistribution restrictions. If the dataset is not committed, each team member must obtain an authorized copy and place it in the documented location.

**Dataset path distinction:** the original project configuration uses `dataset/raw/` for its baseline pipeline. The new Stage 1 audit and region-extraction commands use `dataset/carinthia_s/raw/`. Do not move or rename the original baseline data without coordinating with the team.

## 3. Dataset and manifest

The Carinthia-S CSV uses a semicolon (`;`) delimiter and contains the following columns:

- `image_path`
- `mask_path`
- `filename`
- `label`

The dataset audit generated the following files:

| File | Purpose |
|---|---|
| `data_audit.csv` | Per-record audit results |
| `mask_manifest.csv` | Image-mask associations and mask metadata |
| `duplicate_split_conflicts.csv` | Duplicate entries potentially crossing data splits |

### Recorded audit summary

- CSV records: 4,591
- Audit status: 4,591 records marked `ok`
- Image-mask pairs aligned in dimensions: 4,591 / 4,591
- Exact duplicate image entries: 0
- Duplicate split conflicts: 0
- Empty masks: 226; all belong to class 6 in the audited data
- Full masks: 0
- Masks with intermediate grayscale values: 395
- Baseline split availability: train 3,215; validation 688; test 688

These are the recorded audit results. Empty masks are consistent with the class-6 no-defect examples; they should not automatically be treated as errors.

## 4. Region extraction

`region_extraction.py` generates controlled image inputs using an SEM image and its corresponding defect mask.

### Conditions

- **Original:** the unchanged source image.
- **Defect-only:** retains pixels where the defect mask is nonzero; all other pixels are black.
- **Background-only:** retains pixels where the defect mask is zero; the masked region is black.
- **Artifact-focused:** retains pixels inside a separately supplied artifact mask. This requires an artifact annotation mask; the defect mask is not automatically treated as an artifact mask.

The current implementation interprets every nonzero mask pixel as foreground. Since some masks contain intermediate grayscale values, this convention remains provisional until the mask encoding and intended semantics are confirmed.

### Command

From the project root, run:

```cmd
python -m stage1_data_pipeline.region_extraction --image "dataset/carinthia_s/raw/images/0c8af9b8e62f4cb498392e806236e5e0.jpg" --mask "dataset/carinthia_s/raw/masks/0c8af9b8e62f4cb498392e806236e5e0.png"
```

Replace the example filenames with the image and corresponding mask you want to process.

The script saves three PNG files in:

```text
stage1_data_pipeline/region_examples/
```

The output filenames end with `_original.png`, `_defect_only.png`, and `_background_only.png`.

An optional artifact mask can be supplied with `--artifact-mask`, provided it is a separate, correctly aligned annotation:

```cmd
python -m stage1_data_pipeline.region_extraction --image "PATH_TO_IMAGE" --mask "PATH_TO_DEFECT_MASK" --artifact-mask "PATH_TO_ARTIFACT_MASK"
```

Masked-out pixels are black. This can introduce distribution shift, so changes in model predictions should not be interpreted as proof of causal shortcut reliance.

## 5. Shared inference interface

`inference.py` provides a common interface for loading a trained model and predicting one SEM image.

The model checkpoint and run configuration must belong to the same training run. The run configuration must contain the recorded six-class mapping.

The interface returns:
- Predicted class index and class name.
- Predicted-class confidence.
- Probabilities for all six classes.
- The preprocessing condition used.

The function reuses the existing `preprocessing.py` image-processing function. The model handles its own MobileNetV2 rescaling.

### Example usage

Run this from the project root, using a checkpoint and run-configuration file that actually exist:

```python
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
```

The checkpoint paths above are examples based on the project's naming convention. Confirm that the selected files exist before running the example.

A successful region-extraction run has been recorded. End-to-end inference with a matching checkpoint and run configuration must also be tested before claiming that inference compatibility has been fully validated.

## 6. Reproducibility and split discipline

- Preserve the existing baseline scripts, checkpoints, and results.
- Do not change the train/validation/test assignments for this handoff.
- Use identical image IDs and labels when comparing original and masked conditions.
- Record the checkpoint, run configuration, class mapping, preprocessing, and split used for every experiment.
- Fit calibration parameters and select review thresholds using validation data only.
- Reserve test data for final evaluation.
- Do not retrain or alter the baseline unless a documented compatibility issue requires it.

## 7. Known limitations

1. **Mask semantics:** nonzero pixels are currently treated as foreground. Intermediate grayscale values exist, so this interpretation needs confirmation.
2. **Artifact annotations:** artifact-focused extraction requires a separate artifact mask; it is not produced from the defect mask.
3. **Distribution shift:** blacked-out regions can change image statistics and model behaviour.
4. **Inference compatibility:** use a checkpoint and run configuration from the same training run.
5. **Dataset availability:** the dataset is expected at `dataset/carinthia_s/raw/`, but may need to be obtained separately if it is not included in the repository.
6. **Scientific interpretation:** prediction changes after masking are evidence to investigate, not proof that the model causally relies on a particular feature.

## 8. Handoff to Member 2

Member 2 should receive:

- `mask_utils.py`
- `inference.py`
- `region_extraction.py`
- `audit_dataset.py`
- `data_audit.csv`
- `mask_manifest.csv`
- `duplicate_split_conflicts.csv`
- This README
- Representative mask-overlay and region-extraction examples, where available

Member 2 must also have access to the dataset, a compatible trained checkpoint, and its matching run configuration.

The next stage can use this interface to compare predictions for original, defect-only, and background-only images. Artifact-focused comparisons should be added only when a separate artifact-region definition or annotation is available.

**Current status:** dataset audit completed; region extraction successfully executed on one real image-mask pair; inference interface created; artifact-focused extraction requires a separate mask; mask semantics and end-to-end inference compatibility remain documented validation items.
