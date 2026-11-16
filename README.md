# Mobile VOD

Video object detection for the 30 ImageNet VID classes, with two memory approaches:
**LSTM memory** shared between small/big backbones, and **attention memory** retrieved
from a frozen teacher's last keyframe. Detection uses SSD feature pyramids and heads.

## Models

| LSTM family | Small backbone, 160×160 | Big backbone, 320×320 | Experiment names: small / big / joint |
| --- | --- | --- | --- |
| MobileNetV2 | `mobilenetv2_050.lamb_in1k` | `mobilenetv2_120d.ra_in1k` | `small_050` / `big_120d` / `joint_120d` |
| EfficientNet / TinyNet | `tinynet_d.in1k` | `efficientnet_b0.ra_in1k` | `small_tinynet_d` / `big_efficientnet_b0` / `joint_efficientnet_b0_tinynet_d` |

**Standalone LSTM models:** classification pretraining uses one backbone, global
average pooling, and a 30-class linear head. Detection transfers the backbone and
trains it with a newly initialized LSTM and SSD heads.

**Joint LSTM models:** classification pretraining uses both backbones, a shared
LSTM, and a classification head, repeating each image for three recurrent steps.
Detection transfers both backbones and the LSTM. Both backbones produce
1280×10×10 features; the LSTM outputs 256 channels. Training randomly selects a
backbone per sample/frame; only big-backbone calls commit joint memory. Standalone
models update their own memory every frame. Detection trains on six-frame clips.

**Attention models:** TinyNet-D through `blocks.4` produces 64×10×10 features at
160×160. A frozen EfficientDet-D0 teacher processes 512×512 padded keyframes and
caches its 64×32×32 P4 features. The student uses spatial attention to retrieve
teacher features, adds them to its current features, then applies a 64→256
projection and SSD heads. There is no LSTM.

| Attention experiment | Student features before projection | Keyframe predictions |
| --- | --- | --- |
| `--memory none` | Current features only: no-memory baseline | Frozen teacher |
| `--memory p4` | Current features + retrieved P4 features | Frozen teacher |

These are student-training experiments; the teacher is not jointly optimized.
Evaluation calls the teacher every 10 frames and the student on intervening frames.

## Setup

Run commands from the project root. The launchers activate Conda environment
`vision` and use CUDA. Python entry points also accept `--device cpu`.

```bash
conda create -n vision python=3.11 -y
conda activate vision
python -m pip install -r requirement/requirements.txt \
  -r requirement/requirements_paired.txt -r requirement/requirements_timm.txt
```

Set paths in `path.py`, especially `DATASET_ROOT`. The dataset must provide:

```text
ILSVRC2015/Annotations/imagenet_vid_train.json
ILSVRC2015/Annotations/imagenet_vid_val.json
ILSVRC2015/Data/VID/...
```

The JSON annotations require `videos`, `images`, `annotations`, and `categories`.
Training subset manifests must match the annotations' hash. Dataset files and
trained checkpoints are not included; initial backbone weights download through timm.

## Primary workflow: launchers

### LSTM models

Train or resume classification and detection for all three models in a family:

```bash
bash launchers/run_lstm_experiments.sh mobilenetv2 train
bash launchers/run_lstm_experiments.sh efficientnet train
```

Evaluate their best detection checkpoints without starting training:

```bash
bash launchers/run_lstm_experiments.sh mobilenetv2 eval
bash launchers/run_lstm_experiments.sh efficientnet eval
```

Evaluation requires all three best checkpoints. It uses 100 videos × up to 20
consecutive frames, seed 7; joint models use interval 8 and standalone models use
their sole backbone. Each invocation creates a new results directory. Use `all`
instead of `train` or `eval` to train/resume and then evaluate. Training defaults:

| Stage | Epochs | Batch | Initial LR | Schedule |
| --- | ---: | ---: | ---: | --- |
| Multi-label classification | 5 | 8 images | 0.01 | Linear |
| Detection | 100 | 4 clips | 0.001 | Cosine |

Both stages use RMSprop. The launcher selects the fixed 25% training subset and
validates every 5,000 updates on 100 videos × up to 20 consecutive frames, seed 7.
Set `TRAIN_SUBSET=0.5` for the fixed 50% subset, or supply a manifest path.
Repeat the same `TRAIN_SUBSET` and any `RUN_NAME` overrides when evaluating.
`MAX_VAL_VIDEOS`, `VAL_FRAMES_PER_VIDEO`, and `SUBSET_SEED` control validation
selection; keep them identical for comparable runs.

The launcher resumes latest `classification.pt` / `detection.pt` automatically.
Keep the saved training settings unchanged. Best classification checkpoints maximize
micro AP; best detection checkpoints maximize continuous mAP@0.5.

Default launcher run names encode family, subset, and budgets; standalone run names
append `_backbone_pretrain`. The launcher prints both locations. `RUN_NAME=your_run`
selects an explicit folder under `RUNS_DIR` for all three models.

### Attention models

Configure `DELTA_ROOT`, `DELTA_TEACHER_CHECKPOINT`, `MATCHED_TRAIN_SUBSET`, and
`P4_VALIDATION_MANIFEST` in `path.py`. Required local dependencies are the Delta
fork's `baseline/evaluate_tracking.py`, `tracking.py`, and `lib/`; the released VID
EfficientDet-D0 teacher checkpoint; and `scripts/compare_vid.py` for manifests and
scoring. The local scripts are not bundled in Git. The environment also needs
OmegaConf 2.3.0 and pycocotools.

Export the fixed validation subset once if its manifest does not exist:

```bash
python scripts/compare_vid.py export --manifest splits/val100x20_seed7.json
```

Set `P4_VALIDATION_MANIFEST` to that file. In the attention launcher, set `BEST` to
your trained joint EfficientNet/TinyNet detection checkpoint. Both variants inherit
its TinyNet-D features and SSD weights; their 64→256 projection is newly initialized.
The launcher pins one source checkpoint so both variants start from the same weights.

Train or resume both the no-memory baseline and attention-memory student:

```bash
bash launchers/run_attention_experiments.sh train
```

Evaluate both best checkpoints without training:

```bash
bash launchers/run_attention_experiments.sh eval
```

Defaults are 100 detection epochs, batch 4, RMSprop with cosine decay from 0.001,
and validation every 5,000 updates. Training samples one student frame after each
keyframe from non-overlapping 10-frame windows. The launcher resumes each variant's
latest checkpoint under `RUNS_DIR/p4_student/{none,p4}` and evaluates its best one.
Best selection uses **student-frame COCO AP50**; evaluation also prints whole-sequence
COCO AP50 and continuous AP50.

**Current evaluation scope:** attention checkpoints require the exact saved
validation manifest, including during evaluation. A checkpoint trained with the
100-video manifest cannot currently be evaluated on a different full-validation
manifest through this CLI. The manifest exporter also fixes the selection to
100 videos × up to 20 frames; full-set attention evaluation needs a code change.

## Direct Python: individual models and full validation

Use these commands when you need one model rather than the launcher's complete
experiment set. Activate `vision` first.

### One LSTM model

To train **one standalone or joint model**, choose any experiment from the table:

```bash
EXP=joint_120d
RUNS="$(python path.py --get RUNS_DIR)"
DIR="$RUNS/my_experiment/$EXP"
COMMON=(--experiment "$EXP" --video-ids splits/train_25pct.json
        --num-workers 4 --max-val-videos 100 --val-frames-per-video 20
        --subset-seed 7 --seed 0 --device cuda)

python train_lstm_models.py "${COMMON[@]}" --task classification \
  --epochs 5 --batch-size 8 --learning-rate 0.01 --lr-schedule linear \
  --checkpoint-out "$DIR/classification.pt"
python train_lstm_models.py "${COMMON[@]}" --task detection \
  --epochs 100 --batch-size 4 --learning-rate 0.001 --lr-schedule cosine \
  --init-checkpoint "$DIR/classification_best.pt" --checkpoint-out "$DIR/detection.pt"
```

For full-training-set runs, omit `--video-ids` and leave training selection limits
unset in `path.py`. Use a fresh output folder when changing the dataset or budget.

To resume either stage, repeat its command with `--resume` pointing to that
stage's latest checkpoint, keeping the saved settings unchanged.

### One attention variant

```bash
python train_attention_models.py train --memory p4 \
  --init-detector /path/to/detection_best.pt --run-dir /path/to/new_run --device cuda
python train_attention_models.py eval --memory p4 \
  --checkpoint /path/to/new_run/detection_best.pt --device cuda
```

Use `--memory none` for the baseline. To resume training, replace `--init-detector`
with `--checkpoint /path/to/new_run/detection.pt` and keep the same run directory.

### Full LSTM validation set

Omit video/frame limits to evaluate all validation videos and frames. Use
`--mode fixed` for joint models or `--mode big` for either standalone model.

```bash
python -m evaluate.evaluate_paired_vod \
  --checkpoint /path/to/detection_best.pt --mode fixed --interval 8 --seed 0 \
  --score-threshold 0 --max-detections 100 --warmup-frames 10 \
  --no-losses --no-visuals --device cuda --output-dir /path/to/new_full_eval
```

Leave evaluation `video_ids` unset in `path.py` and use a fresh output directory.
For one model on the standard subset, add `--max-videos 100 --frames-per-video 20
--subset-seed 7`. Results include continuous mAP@0.5, per-class AP, FPS, and operation
counts. Compare FPS on the same hardware/device with identical settings.
Attention full-validation limitations are described above.

## Main modules and monitoring

| Purpose | Main files |
| --- | --- |
| LSTM training and classification | `train_lstm_models.py`, `engine/classification_training.py`, `models/classification_pretraining.py` |
| LSTM detection | `models/joint_lstm_detector.py`, `models/lstm_ssd_detector.py`, `models/bottleneck_lstm.py`, `engine/paired_trainer.py` |
| Attention experiments | `train_attention_models.py`, `models/attention_memory_detector.py`, `models/frozen_efficientdet_teacher.py` |
| Shared backbones and heads | `models/backbones.py`, `models/ssd_head.py`, `models/anchors.py` |
| Data and preprocessing | `datasets/imagenet_vid.py`, `datasets/classification.py`, `datasets/transforms.py`, `datasets/dataloader.py` |
| Evaluation and checkpoints | `evaluate/evaluate_paired_vod.py`, `engine/paired_evaluator.py`, `utils/lstm_checkpoints.py`, `utils/runtime.py` |

```bash
bash launchers/show_train_history.sh                    # All discovered runs
bash launchers/show_train_history.sh /path/to/run       # One run
```

History reports saved training/validation metrics, best scores, learning rates,
and remaining updates. It can be read while training continues.
