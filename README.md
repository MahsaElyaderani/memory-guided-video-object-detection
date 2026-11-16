# Mobile VOD

PyTorch video object detection inspired by *Looking Fast and Slow: Memory-Guided
Mobile Video Object Detection*. The project combines pretrained image backbones,
recurrent memory, and SSD detection heads for the 30 ImageNet VID classes.
It supports timm MobileNetV2 and EfficientNet-B0/TinyNet-D, with standalone
and joint models. Both families use a shared ConvLSTM and SSD detection heads in
joint mode. Backbone weights are ImageNet-pretrained; SSD heads are trained here.

| Model family | Small backbone (160×160) | Big backbone (320×320) |
| --- | --- | --- |
| MobileNetV2 | `mobilenetv2_050.lamb_in1k` | `mobilenetv2_120d.ra_in1k` |
| EfficientNet / TinyNet | `tinynet_d.in1k` | `efficientnet_b0.ra_in1k` |

Backbones are implemented in `models/backbones.py`. Both families use a 1280-channel
LSTM input, a 256-channel LSTM output, and the same SSD head architecture.

## Setup

Create a separate environment (the launcher expects a Conda environment named `vision`):

```bash
git clone https://github.com/MahsaElyaderani/mobile_vod.git
cd mobile_vod
conda create -n vision python=3.11 -y
conda activate vision
python -m pip install -r requirement/requirements.txt \
  -r requirement/requirements_paired.txt -r requirement/requirements_timm.txt
```

The requirements pin PyTorch 2.6.0, TorchVision 0.21.0, and timm 1.0.22.
The default launcher requires CUDA; check that `torch.cuda.is_available()` is true.

Edit `path.py` for your machine, especially `DATASET_ROOT`. Training expects:

```text
ILSVRC2015/
  Annotations/imagenet_vid_train.json
  Annotations/imagenet_vid_val.json
  Data/VID/...
```

Annotations must be video-aware COCO-style JSON with `videos`, `images`,
`annotations`, and `categories`; raw ImageNet XML files are not loaded directly.
The included `splits/train_25pct.json` must match your training annotation file
(the loader checks its hash). Dataset files and trained checkpoints are not included.
Initial backbone training downloads pretrained weights through timm.

## Train and evaluate

Run commands from the project root. Choose one family; each launcher processes
small standalone, big standalone, then joint. It activates the `vision` environment.

New standalone classification uses backbone → global average pooling → a 30-class
linear head, with one prediction per image. Detection transfers the backbone and
starts with freshly initialized LSTM and SSD weights. Joint classification keeps
its three recurrent steps and transfers both backbones and the shared LSTM.

Train only (repeating a command resumes its default run):

```bash
bash launchers/run_lstm_experiments.sh mobilenetv2 train
bash launchers/run_lstm_experiments.sh efficientnet train
```

Resume an existing run, including one created by an older launcher. Replace each
`EXISTING_..._RUN` with its folder name directly under `RUNS_DIR`, without `/seed_0`:

```bash
RUN_NAME="EXISTING_MOBILENET_RUN" bash launchers/run_lstm_experiments.sh mobilenetv2 train
RUN_NAME="EXISTING_EFFICIENTNET_RUN" bash launchers/run_lstm_experiments.sh efficientnet train
```

Resume loads latest `classification.pt` and `detection.pt`, skips completed stages,
and trains missing experiments. Preserve the original subset, budget, batch sizes,
worker count, and validation settings; incompatible resumes are rejected.

Evaluate the same existing runs using their best detection checkpoints:

```bash
RUN_NAME="EXISTING_MOBILENET_RUN" bash launchers/run_lstm_experiments.sh mobilenetv2 eval
RUN_NAME="EXISTING_EFFICIENTNET_RUN" bash launchers/run_lstm_experiments.sh efficientnet eval
```

Omit `RUN_NAME` to evaluate a default run. Use `all` instead of `train` or `eval`
to train/resume and then evaluate; `all` is also the default when omitted.
Evaluation-only mode requires all three best detection checkpoints and never
runs training. Each evaluation creates a new results directory.

The launcher defaults to `TRAIN_SUBSET=0.25`. Set `0.5` for the fixed 50%
manifest; the run name is automatic:

```bash
TRAIN_SUBSET=0.5 \
  bash launchers/run_lstm_experiments.sh efficientnet all
# Later, evaluate the same run without training:
TRAIN_SUBSET=0.5 \
  bash launchers/run_lstm_experiments.sh efficientnet eval
```

Explicit JSON manifest paths are also accepted through `TRAIN_SUBSET`. These
fractions select the saved video subsets; they do not resample the dataset.

Optional `MAX_VAL_VIDEOS`, `VAL_FRAMES_PER_VIDEO`, and `SUBSET_SEED` environment
variables override 100, 20, and 7 for both training validation and final evaluation.
Use the same values across families for comparisons. When evaluating later, supply
any validation overrides again. Default run names include the family, subset filename,
and epoch budgets; nondefault validation settings add a suffix. Existing default
25% run names are preserved. Use `RUN_NAME` only to select an explicit folder or
start an independent repeat. Different manifests with the same filename need
separate `RUN_NAME` values; the trainer still rejects incompatible resumes. To evaluate
an existing run with different validation coverage, set `RUN_NAME` to that run explicitly. Continuing an existing
25% checkpoint on 50% with its optimizer/schedule is the separate
`launchers/run_joint_subset50.sh` workflow for joint MobileNetV2. It starts from the
best validated detector with cosine updates remaining. The new run inherits the
optimizer, update count and best score. Later invocations resume its latest checkpoint.

| Family | Small / big / joint experiments |
| --- | --- |
| `mobilenetv2` | `small_050`, `big_120d`, `joint_120d` |
| `efficientnet` | `small_tinynet_d`, `big_efficientnet_b0`, `joint_efficientnet_b0_tinynet_d` |

| Stage | Epochs | Batch size | Initial LR | Schedule |
| --- | ---: | ---: | ---: | --- |
| Classification | 5 | 8 images | 0.01 | Linear |
| Detection | 100 | 4 clips | 0.001 | Cosine |

This launcher selects `splits/train_25pct.json` by default, independently of
`MATCHED_TRAIN_SUBSET` in `path.py`. Joint classification repeats each image for three
recurrent steps; standalone classification uses one pass. Detection
uses six-frame clips. Validation runs every 5,000 updates on up
to 100 videos with 20 frames each (subset seed 7, training seed 0). Final evaluation
uses the same validation selection, big-only mode for standalone models and
interval 8 for joint models. Adjust batch sizes to GPU memory before starting.

New runs use separate family folders:
`RUNS_DIR/shared_<family>_subset25_class5_linear_det100_cosine/seed_0/<experiment>/`
for joint models on 25%, or the equivalent `subset50` folder for 50%.
Standalone defaults append `_backbone_pretrain` to the run name, before `/seed_0/`.
Explicit `RUN_NAME` uses that exact run folder for all models, including existing runs.
Old standalone classification checkpoints resume their saved recurrent architecture.
Previous experiment folders are not reused. Repeating the same command resumes
latest `classification.pt` and `detection.pt`, including mid-epoch progress.
Keep the saved training budget and settings unchanged when resuming.

Fresh detection starts from `classification_best.pt`. Best checkpoints use
highest classification micro AP (final recurrent step) or continuous detection
mAP@0.5. Classification trains with multi-label BCE on the single standalone output
or all three joint outputs. Joint
detection training samples a backbone independently per example and frame with
equal probability. Only big calls commit the shared memory. Final evaluation loads
`detection_best.pt`. Evaluation JSON files go to a timestamped family folder under
`RESULTS_DIR`. All paths are configured in `path.py`.

For individual commands and options:

```bash
python train_lstm_models.py --help
python -m evaluate.evaluate_paired_vod --help
```

The earlier `standalone_pretrain` / `standalone_detection` trainer is archived locally
under `archives/standalone_stages_20261116/` (source snapshot included).
Use `train_lstm_models.py` for current standalone and joint experiments.
Frozen-big training is archived under `archives/frozen_big_20261116/`.
Its checkpoints remain evaluable; resuming that training mode requires the archived source.
`prepare_joint_training.py` now prepares trainable joint runs by default.
`train_joint_pretrain.py` and `train_paired_vod.py` provide separate joint entry points.
The shared launcher replaces the older family launchers.

Show saved classification and detection history while training:

```bash
bash launchers/show_train_history.sh
```

Pass a run folder as the first argument to limit the report to that run.

Show classification and detection history for each family's default 25% run:

```bash
conda activate vision
RUNS="$(python path.py --get RUNS_DIR)"
bash launchers/show_train_history.sh "$RUNS/shared_mobilenetv2_subset25_class5_linear_det100_cosine/seed_0"
bash launchers/show_train_history.sh "$RUNS/shared_mobilenetv2_subset25_class5_linear_det100_cosine_backbone_pretrain/seed_0"
bash launchers/show_train_history.sh "$RUNS/shared_efficientnet_subset25_class5_linear_det100_cosine/seed_0"
bash launchers/show_train_history.sh "$RUNS/shared_efficientnet_subset25_class5_linear_det100_cosine_backbone_pretrain/seed_0"
```

For just one model, append its experiment name from the family table above:

```bash
bash launchers/show_train_history.sh "$RUNS/shared_efficientnet_subset25_class5_linear_det100_cosine/seed_0/joint_efficientnet_b0_tinynet_d"
```

For an older or custom run, replace the run folder with its actual name. For a
default 50% run, replace `subset25` with `subset50`. The report reads the latest
saved checkpoints, includes best validation scores, learning rates and remaining
updates, and skips models without checkpoints. It can run while training continues.

MobileNetV1 is no longer supported by the active code. Its checkpoints require
the archived version; existing MobileNetV2 and EfficientNet checkpoint formats
are preserved. Local archives, datasets, and checkpoint weights are not published.

EdgeNeXt source and tests are archived locally in `archives/edgenext_20261116/`.
Use that source snapshot to train, evaluate, or resume existing EdgeNeXt checkpoints.

The matched `big_100`, `big_110d`, `joint_100`, and `joint_110d` experiments are archived
in `archives/mobilenet_variants_20261116/`. Use that source snapshot for their training,
evaluation, or resume. Low-level MobileNetV2 backbone support remains available.

## Attention-based Memory

This detection experiment asks whether the last keyframe helps TinyNet-D detect
objects on intervening frames. Both variants use TinyNet-D through `blocks.4`
on a 160×160 frame. The student feature map is 64×10×10. The frozen
EfficientDet-D0 teacher processes a padded 512×512 keyframe and supplies its
64×32×32 P4 map. Each variant uses one 64→256 projection before the SSD pyramid.

| Trainer mode | Input to the projection | Prediction on keyframes |
| --- | --- | --- |
| `none` | Current student features | Teacher detections |
| `p4` | Current features plus a spatially weighted sum of cached P4 values | Teacher detections |

**Setup.** Set `DELTA_ROOT` to the Delta Distillation checkout containing the
project-specific `baseline/evaluate_tracking.py`, its sibling `tracking.py`,
and the fork's `lib/` directory. The adapter calls that baseline's `load_model`
and `records` functions. Set `DELTA_TEACHER_CHECKPOINT` to the released VID
`checkpoints/test/efficientdet_d0.pth`. The fork uses OmegaConf 2.3.0 and the
scorer needs pycocotools in the active environment. Set `P4_VALIDATION_MANIFEST`
to `splits/val100x20_seed7.json` and use its matching validation annotations.
See the [Delta Distillation release](https://github.com/Qualcomm-AI-research/delta-distillation/releases/tag/v1.0)
for the weights.

**Initialization.** In `launchers/run_attention_experiments.sh`, set `BEST` to the trained
joint EfficientNet/TinyNet-D detection checkpoint. On first use the launcher
pins it with a hard link at `RUNS_DIR/p4_student/joint_init.pt` so both variants
start from the same weights. The student inherits the TinyNet-D stem and blocks
through `blocks.4`, the SSD pyramid and predictors. Its 64→256 projection starts
fresh. The Python trainer also accepts a matched `small_tinynet_d` detection
checkpoint via `--init-detector`.

```bash
bash launchers/run_attention_experiments.sh train  # Start or resume both from their own detection.pt
bash launchers/run_attention_experiments.sh eval   # Evaluate each detection_best.pt on CUDA
```

**Training and selection.** The trainer reads `MATCHED_TRAIN_SUBSET` (25% by
default), divides videos into non-overlapping 10-frame windows, and samples one
student frame 1–9 frames after each keyframe. Both images share a crop and flip,
and the student frame supplies the detection targets. The default budget is
100 epochs with batch size 4 and cosine decay from 0.001. Validation runs every
5,000 optimizer updates and at the final update.

At validation, the cache resets at each video's selected window. Frame 1 and
every tenth frame after it use the teacher. **Best means highest COCO AP50 on
student frames** because those are the frames where the variants differ.
Evaluation also reports whole-sequence AP50 and continuous AP50. The trainer
stores each variant's history in `RUNS_DIR/p4_student/{none,p4}/detection.pt`.
Run the following command from the project root to compare the best student-frame
COCO AP50, epoch, and update for each variant with a saved checkpoint:

```bash
bash launchers/show_train_history.sh "$(python path.py --get RUNS_DIR)/p4_student"
```

**Reading the code.** Start with `models/attention_memory_detector.py` for fusion and
`models/frozen_efficientdet_teacher.py` for P4 extraction. In `train_attention_models.py`,
`TrainingPairs` defines sampling and `evaluate` defines routing and scoring.
The shared scorer and manifest reader live in `scripts/compare_vid.py`.

Local results, runs, checkpoints and tests are excluded from Git. Supported
launchers live in `launchers/`.

### File naming and compatibility

LSTM experiments start in `train_lstm_models.py` and `launchers/run_lstm_experiments.sh`.
Attention-memory experiments (including their no-memory baseline) start in
`train_attention_models.py` and `launchers/run_attention_experiments.sh`.
Standalone classification pretraining remains backbone-only. Model implementations
are in `models/joint_lstm_detector.py`, `models/lstm_ssd_detector.py`,
`models/classification_pretraining.py`, `models/attention_memory_detector.py`, and
`models/frozen_efficientdet_teacher.py`.

The previous entry points, launchers, and model imports remain compatibility aliases.
Run folders, experiment identifiers, checkpoint formats, and resume settings are unchanged.
