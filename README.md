# DP-Det

Research implementation containing the supplied DP-Det method modules, model configuration, data preparation, training, evaluation and loss sensitivity settings.

## Release scope

This source release preserves the active framework files supplied by the authors. Twelve modified framework files and one custom model YAML are distributed as patches over **Ultralytics 8.4.7**; unchanged upstream files are installed from the pinned dependency. `patch-manifest.json` checks the upstream and modified source hashes before installation. Historical backup copies and duplicate scripts are omitted.

The launchers were consolidated for portable paths and explicit experiment settings. This is a source-and-configuration release, not an archive of every historical run. GPU training/inference and exact reproduction of manuscript scores have not been validated for this prepared release. The supplied current model YAML enhances P2 and P3. The manuscript loss setting is alpha=2.1, delta=3, scale bias=50; the supplied active loss file defaults to alpha=1.9, so the launcher explicitly selects the profile instead of relying on defaults. An additional H15 profile is retained separately and is not asserted to identify the historical main-result checkpoint.

## Installation

Use a fresh environment (Python 3.11 is a starting point, not a claim about the exact historical training environment). On Windows PowerShell, from this repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tools/install_framework_patches.py --apply
```

Use that same environment's Python for all commands below. On Linux/macOS use `.venv/bin/python`. Install a PyTorch build appropriate for your machine/GPU if necessary. The patch tool refuses unexpected local changes and requires a dedicated environment; do not apply it to another project environment. Preserve `LICENSE` (GNU AGPL-3.0) and upstream notices.

## Datasets and split

Download the original datasets from their providers and follow their terms and citation instructions:

- VisDrone: https://github.com/VisDrone/VisDrone-Dataset (object detection in images).
- Original 800-image NWPU VHR-10: https://gcheng-nwpu.github.io/#Datasets (not NWPU VHR-10.v2).

Dataset images/annotations are not redistributed here. Prepare data using:

```text
python scripts/prepare_visdrone.py --raw-root "PATH_TO_VISDRONE_RAW"
python scripts/prepare_nwpu.py --raw-root "PATH_TO_NWPU_RAW"
```

The VisDrone root contains `VisDrone2019-DET-train` and `VisDrone2019-DET-val`, each with `images` and `annotations`; test-dev is optional. NWPU root contains `positive image set`, `negative image set`, and `ground truth`. Output directories must be empty. Defaults match `configs/visdrone.yaml` and `configs/nwpu.yaml`; if changing output locations, update those dataset YAML paths. Relative dataset roots in the supplied YAML are resolved from the repository root by the training/evaluation launchers.

NWPU uses seed 2026 and an 80/20 split of all 800 images, including negative images: 640 training and 160 validation images. `splits/nwpu_train.txt` and `splits/nwpu_val.txt` list the prepared image names (positive/negative prefixes), recovered from the supplied project inventory. The supplied preparation logic reproduces this membership for the original dataset. These are file lists, not redistributed images or annotations.

## Training

Obtain the YOLOv8s pretrained weights from the upstream provider and place them at `weights/yolov8s.pt` (not bundled). Select initialization explicitly with `--pretrained` or `--scratch`:

```text
python scripts/train.py --data configs/visdrone.yaml --variant ours --profile a2p1_d3_b50 --pretrained weights/yolov8s.pt --name visdrone_ours
python scripts/train.py --data configs/visdrone.yaml --variant uniform --profile a2p1_d3_b50 --pretrained weights/yolov8s.pt --name visdrone_uniform
python scripts/train.py --data configs/visdrone.yaml --variant baseline --profile ciou --pretrained weights/yolov8s.pt --name visdrone_baseline
python scripts/train.py --data configs/nwpu.yaml --variant ours --profile a2p3_d2p5_b30 --pretrained weights/yolov8s.pt --warmup-epochs 3 --name nwpu_h15
```

Defaults: 300 maximum epochs, early-stopping patience 50, batch 8, image size 640, SGD, lr 0.01, momentum 0.937, weight decay 0.0005, seed 2026. Use `--device`, `--epochs`, etc. to configure your machine. Arguments, dependency versions, environment switches and pretrained-weight hash are recorded in `run-settings/`; model outputs go to `runs/`. Existing run names are rejected.

`ours` selects the learned prior (D), `uniform` selects uniform prior (B), and `fap2-only` selects uniform prior without DLKA or BS-WIoU (use profile `ciou`). These are implementation switches; they do not certify a mapping to every historical manuscript ablation row.

## Loss sensitivity experiments

`configs/loss-profiles.json` documents parameter values. The nine rows in `configs/sensitivity-grid.json` contain seven unique combinations; shared reference settings run once. Print commands first, then add `--run` to train:

```text
python scripts/run_sensitivity.py --data configs/visdrone.yaml --pretrained weights/yolov8s.pt
python scripts/run_sensitivity.py --data configs/visdrone.yaml --pretrained weights/yolov8s.pt --run
```

## Evaluation

The portable evaluator uses COCO bbox AP with IoU 0.50:0.95, image size 640 for inference, confidence threshold 0.001, NMS IoU 0.7, and max detections **300**. AP values are explicitly read at maxDets=300. Image ID 0 is handled correctly. Category IDs are validated rather than inferred from the minimum predicted class.

Build COCO GT from the prepared validation YOLO labels:

```text
python scripts/build_coco_gt.py --images datasets/VisDrone_YOLO/VisDrone2019-DET-val/images --labels datasets/VisDrone_YOLO/VisDrone2019-DET-val/labels --data configs/visdrone.yaml --output local-eval/visdrone_gt.json
python scripts/evaluate.py --gt local-eval/visdrone_gt.json --weights runs/visdrone_ours/weights/best.pt --data configs/visdrone.yaml --output local-eval/ours_metrics.json
```

Or evaluate an existing prediction file:

```text
python scripts/evaluate.py --gt local-eval/visdrone_gt.json --predictions PATH_TO_PREDICTIONS.json --output local-eval/metrics.json
```

The GT builder uses category IDs 1..10. If a prediction file uses IDs 0..9, explicitly add `--category-offset 1`. The pinned Ultralytics detector exports non-COCO datasets with category IDs starting at 1; use no offset for its normal output. For other exports, verify the mapping explicitly; the evaluator stops on IDs outside GT categories. Use `--mode B` when evaluating a uniform-prior model.

The supplied VisDrone converter excludes categories 0 and 11 and does not implement the complete official ignored-region protocol. Therefore these commands describe **COCO bbox evaluation of converted annotations**, not the official VisDrone challenge evaluator. For NWPU, use its validation image/label directories and `configs/nwpu.yaml` in the same GT-building workflow.

## Files

- `patches/`: exact supplied active framework modifications and custom model YAML.
- `tools/`: version/hash-checked patch installer.
- `scripts/`: consolidated data preparation, training, sensitivity and evaluation entries.
- `configs/`: dataset settings and explicit loss profiles/grid.
- `splits/`: NWPU train/validation membership.
- `requirements.txt`, `LICENSE`, `patch-manifest.json`: dependencies, license and patch checks.

Raw datasets, pretrained/trained weights, prediction dumps, caches, local path inventories and manuscript drafts are outside this source release. No new large inference-data archive is required to use the source package.
