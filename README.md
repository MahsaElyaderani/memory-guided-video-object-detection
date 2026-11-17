# Mobile VOD

## Train

Video object detection for the 30 ImageNet VID classes, using either LSTM memory
or attention to a frozen teacher's cached keyframe features.

Configure dataset and output paths in `path.py`. Launchers run from the project
root, activate the `vision` Conda environment, and use CUDA. Select the launcher's
`train` action to start training or resume its latest checkpoints.

| Models | Launcher | Selection | Training |
| --- | --- | --- | --- |
| MobileNetV2: small, big, joint | [run_lstm_experiments.sh](launchers/run_lstm_experiments.sh) | `mobilenetv2` | Small: `mobilenetv2_050.lamb_in1k`; big: `mobilenetv2_120d.ra_in1k`. Runs both standalone models and their joint model. |
| EfficientNet/TinyNet: small, big, joint | [run_lstm_experiments.sh](launchers/run_lstm_experiments.sh) | `efficientnet` | Small: `tinynet_d.in1k`; big: `efficientnet_b0.ra_in1k`. Runs both standalone models and their joint model. |
| Attention memory and no-memory baseline | [run_attention_experiments.sh](launchers/run_attention_experiments.sh) | Both variants automatically | Trains a TinyNet-D student with or without cached EfficientDet-D0 P4 features. The teacher stays frozen. |

**LSTM models** use 160×160 small inputs and 320×320 big inputs. Standalone
classification pretraining trains the backbone and a classification head; joint
pretraining trains both backbones, their shared LSTM, and a classification head.
Detection transfers the pretrained components and adds SSD heads. Standalone
models update their own memory each frame; only big-backbone calls update joint
memory.

**Attention models** use TinyNet-D features from the current 160×160 frame to query
cached teacher features. Retrieved features are added before a projection and SSD
heads. The baseline uses current features alone. Set `BEST` in the attention
launcher to a trained joint EfficientNet/TinyNet detector; both students inherit
its backbone and SSD weights. Configure the external Delta fork, teacher weights,
and validation manifest in `path.py`; the local `scripts/compare_vid.py` utility,
OmegaConf, and pycocotools are also required.

| Launcher defaults | LSTM experiments | Attention experiments |
| --- | --- | --- |
| Training subset | Fixed 25%; `TRAIN_SUBSET` accepts `0.25`, `0.5`, or a manifest path | `MATCHED_TRAIN_SUBSET` in `path.py` |
| Classification | 5 epochs, batch 8, LR 0.01, linear decay | No separate classification stage |
| Detection | 100 epochs, batch 4, LR 0.001, cosine decay | 100 epochs, batch 4, LR 0.001, cosine decay |
| Optimizer | RMSprop | RMSprop |
| Validation frequency | Every 5,000 updates and at budget completion | Every 5,000 updates and at budget completion |

Resume uses latest checkpoints; keep the saved training settings unchanged.
LSTM launchers print the selected run folders; `RUN_NAME` selects an explicit run.
Attention runs are stored under `RUNS_DIR/p4_student/{none,p4}`.
[show_train_history.sh](launchers/show_train_history.sh) reports saved training and
validation history, best scores, learning rates, and remaining updates.

## Eval

Select the same launcher's `eval` action to evaluate best detection checkpoints
without starting training. Use the same run selection as training.

| Models | Launcher | Evaluation |
| --- | --- | --- |
| MobileNetV2: small, big, joint | [run_lstm_experiments.sh](launchers/run_lstm_experiments.sh), selection `mobilenetv2` | Evaluates all three models; requires all three best detection checkpoints. |
| EfficientNet/TinyNet: small, big, joint | [run_lstm_experiments.sh](launchers/run_lstm_experiments.sh), selection `efficientnet` | Evaluates all three models; requires all three best detection checkpoints. |
| Attention memory and no-memory baseline | [run_attention_experiments.sh](launchers/run_attention_experiments.sh) | Evaluates both best student checkpoints with the frozen teacher. |

**Validation selection:** LSTM launchers default to 100 videos × up to 20 consecutive
frames, subset seed 7. `MAX_VAL_VIDEOS`, `VAL_FRAMES_PER_VIDEO`, and `SUBSET_SEED`
control this selection. Repeat any training run-selection overrides for evaluation.
Attention evaluation requires the exact validation manifest recorded in its checkpoint.

**Inference:** standalone models use their sole backbone each frame. Joint LSTM
models call the big backbone every 8 frames. Attention experiments use teacher
detections every 10 frames and student detections between keyframes.

**Metrics and best checkpoints:** LSTM classification selects highest micro AP;
LSTM detection selects highest continuous mAP@0.5. Attention selects highest
student-frame COCO AP50 and also reports whole-sequence COCO AP50 and continuous
AP50. Compare models using identical validation frames and metric definitions;
compare FPS using the same device and hardware.

Both launchers save evaluation results in a new directory under `RESULTS_DIR`.
Attention exports include metrics, per-frame predictions, and ground-truth/prediction
videos for both `none` and `p4`, with teacher keyframes labeled. The saved predictions
can also be compared side by side using `evaluate.compare_detection_runs`, without
rerunning inference. Full-validation evaluation is not exposed by these
launchers; attention evaluation additionally enforces its saved manifest.
