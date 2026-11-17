"""Entry point for standalone and joint LSTM experiments: classification pretraining and detection training."""
import argparse
from copy import copy
from dataclasses import asdict
from types import SimpleNamespace
from pathlib import Path
import math
import uuid

import torch
from torch import nn
from datasets.dataset_imagenetvid import ClassificationDataset
from engine import trainer_lstm as trainer
from models.classification_pretraining import JointClassifier
from models.lstm_ssd_detector import LSTMSSD
from models.joint_lstm_detector import JointLSTMDetector
from models.ssd_head import SSDFeatureMaps, SSDBoxPredictor
from lstm_config import PairedConfig
from path import parse_paths, RUNS_DIR
from routing.data import data_signature, guard_held_out, selected_dataset, validate_native_categories
from datasets.transforms import TrainTransform
from utils.evaluation_data import select_evaluation_frames
from utils.checkpoints import restore_rng
from utils.joint_initialization import make_joint_config
from utils.lstm_checkpoints import lstm_metadata
from utils.runtime import run_lock, load_checkpoint, atomic_save_checkpoint, seed_training
from routing.predictions import sha256_file
from utils.resume import resume_defaults, saved_arguments, restore_optimizer

EXPERIMENTS = {'small_050': .5, 'big_120d': 1.2, 'joint_120d': 1.2,
               'small_tinynet_d': 1., 'big_efficientnet_b0': 1.,
               'joint_efficientnet_b0_tinynet_d': 1.}


class MatchedDetector(JointLSTMDetector):
    def __init__(self, experiment, seed=0):
        if experiment not in EXPERIMENTS:
            raise ValueError(f'Unknown or archived experiment: {experiment}')
        family = ('tinynet_d' if experiment == 'small_tinynet_d' else
                  'efficientnet_b0' if 'efficientnet_b0' in experiment else 'timm_mobilenet_v2')
        cfg = make_joint_config(family, EXPERIMENTS[experiment])
        paired = PairedConfig(fast_width_mult=.5 if family == 'timm_mobilenet_v2' else 1.)
        self.standalone = None if experiment.startswith('joint_') else ('small' if experiment.startswith('small_') else 'big')
        if self.standalone is None:
            super().__init__(LSTMSSD(cfg), paired)
        else:
            nn.Module.__init__(self)
            self.big = LSTMSSD(cfg)
            self.paired_config = paired
            self.fast_backbone = nn.Identity()
            self.small_projection = nn.Identity()
            if self.standalone == 'small':
                self.big.backbone.set_output_stride16()
        # Separate initialization stream: changing backbone depth cannot change shared weights.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.big.lstm.reset_parameters()
            self.big.feature_maps = SSDFeatureMaps()
            self.big.predictor = SSDBoxPredictor(cfg.feature_channels, cfg.num_classes,
                self.ssd_cfg.num_anchors_per_location, self.ssd_cfg.include_background_class)
        self.configure_detector_training()

    def forward_big(self, frames, state=None):
        if self.standalone == 'small':
            return self.big._forward_from_backbone(self.small_projection(self.big.backbone(frames.small)), state, True,
                                                   anchor_dtype=frames.dtype)
        return super().forward_big(frames, state)

    def forward_small(self, frames, state=None):
        return self.forward_big(frames, state) if self.standalone else super().forward_small(frames, state)

    def forward_selected(self, frames, state, use_big, small_commit=None):
        if self.standalone:
            return self.forward_big(frames, state)
        return super().forward_selected(frames, state, use_big, small_commit)


def build_model(experiment, task, seed):
    detector = MatchedDetector(experiment, seed)
    if task == 'detection':
        return detector
    if task != 'classification':
        raise ValueError('Unknown matched task.')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed + 1)
        return JointClassifier(detector.model_cfg, detector.paired_config, 30, detector=detector)


def model_from_matched(payload):
    if payload.get('matched_format_version') != 1 or payload.get('stage') not in ('matched_classification', 'matched_detection'):
        raise ValueError('Expected a matched classification/detection checkpoint.')
    if payload.get('class_to_idx') != {str(i): i - 1 for i in range(1, 31)}:
        raise ValueError('Matched class mapping differs.')
    task = payload['stage'].removeprefix('matched_')
    architecture = payload.get('classification_architecture', 'recurrent')
    expected_architecture = ('backbone_only' if task == 'classification' and
                             not payload['experiment'].startswith('joint_') else 'recurrent')
    if architecture != expected_architecture:
        raise ValueError('Incompatible classification architecture metadata.')
    model = build_model(payload['experiment'], task, payload['initialization_seed'])
    expected = lstm_metadata(model)
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError('Matched architecture/configuration differs.')
    if payload.get('protocol') != {'bn_stats': 'frozen', 'standalone_memory': 'update', 'joint_memory': 'big_only'}:
        raise ValueError('Matched training protocol differs.')
    model.load_state_dict(payload['model'], strict=True)
    return model


def parse_args(argv=None):
    from routing.data import add_data_arguments
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--val-annotations")
    parser.add_argument("--val-frames-root")
    parser.add_argument("--val-video-ids")
    parser.add_argument("--max-val-videos", type=int)
    parser.add_argument("--val-frames-per-video", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--clip-length", type=int, default=6)
    parser.add_argument("--clip-stride", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--sharing-strategy", choices=("file_descriptor", "file_system"),
                        help="CPU tensor transfer strategy; file_system avoids per-storage cached file descriptors.")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--grad-clip", type=float, default=10.)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--big-call-interval", type=int, default=8,
                        help="Call the big backbone every N frames during fixed-policy validation; not validation frequency.")
    parser.add_argument("--validation-interval", dest="big_call_interval", type=int,
                        default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-out")
    parser.add_argument("--resume")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--log-every", type=int, default=50,
                        help="Print a persistent loss line every N detector batches, plus first batch and epoch mean.")
    parser.add_argument('--task', choices=('classification', 'detection'), required=True)
    parser.add_argument('--experiment', choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument('--lr-schedule', choices=('linear', 'cosine'), help='Defaults to linear for classification and cosine for detection.')
    parser.add_argument('--validate-every', type=int, default=5000, help='Budgeted validation interval in optimizer updates.')
    parser.add_argument('--early-stopping', action=argparse.BooleanOptionalAction, default=None,
                        help='Stop after 10 validations without improvement; resume inherits the saved setting')
    parser.add_argument('--steps', type=int, default=3,
                        help='Recurrent classification steps; backbone-only standalone models use one pass.')
    parser.add_argument('--init-checkpoint')
    parser.add_argument('--expand-subset', action='store_true',
                        help='Continue joint cosine detection on a larger subset in a fresh output directory.')
    args = parse_paths(parser, 'train_matched', argv)
    args.lr_schedule = args.lr_schedule or ('linear' if args.task == 'classification' else 'cosine')
    args = resume_defaults(parser, args, argv)
    if args.lr_schedule not in ('linear', 'cosine'):
        parser.error('Constant-LR training is retired; use linear or cosine for a new run.')
    if args.validate_every < 1:
        parser.error('validate-every must be positive')
    if args.steps < 1 or args.learning_rate is not None and (not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error('steps and learning-rate must be positive and finite')
    if not args.val_annotations or not args.val_frames_root:
        parser.error('Matched experiments require validation annotations and frames root')
    directory = Path(RUNS_DIR) / 'matched' / f'seed_{args.seed}' / args.experiment
    args.checkpoint_out = args.checkpoint_out or str(directory / (args.task + '.pt'))
    if args.task == 'classification' and args.init_checkpoint:
        parser.error('Classification starts from official timm weights, not another experiment')
    if args.task == 'detection' and not args.resume:
        args.init_checkpoint = args.init_checkpoint or str(directory / 'classification.pt')
    args.memory_corruption_probability, args.memory_dropout, args.memory_noise_std = 0., .1, .1
    return args


def checkpoint_position(payload):
    return payload['epoch'], payload.get('budget', {}).get('batch', 0)


def expand_subset_checkpoint(resume, definition, signatures, args):
    """Fork a detector's training state; only the dataset and sampling position change."""
    if (not resume or args.task != 'detection' or not args.experiment.startswith('joint_') or
            args.lr_schedule != 'cosine' or args.learning_rate is not None):
        raise ValueError('Expansion requires joint detection --resume with cosine and no LR override.')
    old = resume['run_definition']
    if any(definition[k] != old[k] for k in ('arguments', 'validation', 'num_workers')):
        raise ValueError('Expansion must preserve training options and validation.')
    before, after = old['training'], definition['training']
    if (any(before[k] != after[k] for k in before if k not in ('video_ids', 'image_ids')) or
            not set(before['video_ids']) < set(after['video_ids']) or
            not set(before['image_ids']) < set(after['image_ids'])):
        raise ValueError('Expansion requires a strict superset from the same annotations.')
    saved = resume.get('budget', {})
    batches = saved.get('batches_per_epoch', saved.get('total_updates', 0) // args.epochs)
    if (saved.get('epochs') != args.epochs or saved.get('sampling_version') != 1 or
            not 0 <= saved.get('batch', -1) <= batches or
            saved.get('updates') != saved.get('origin_updates', 0) +
            (saved.get('epoch', 0) - saved.get('origin_epoch', 1)) * batches + saved.get('batch', -1) or
            not 0 < saved.get('updates', 0) < saved.get('total_updates', 0) or
            resume['scheduler']['last_epoch'] != saved['updates'] or
            resume['scheduler'].get('T_max') != saved['total_updates']):
        raise ValueError('Source must have valid, unfinished cosine progress and the original epoch target.')
    scores = [row['validation']['mAP@0.5'] for row in resume['history']]
    if (not scores or not all(math.isfinite(score) for score in scores) or
            resume['history'][-1].get('update') != saved['updates'] or scores[-1] != max(scores)):
        raise ValueError('Expand from the best validated detection checkpoint.')
    destination = Path(args.checkpoint_out)
    if destination.parent.resolve() == Path(args.resume).resolve().parent or list(destination.parent.glob('*.pt')):
        raise FileExistsError('Expansion requires a separate directory without checkpoints.')
    epoch = saved['epoch'] + 1
    return {**resume, 'run_id': uuid.uuid4().hex, 'epoch': epoch, 'run_definition': definition,
            'resume_arguments': saved_arguments(args), 'training_signatures': signatures,
            'sources': {**resume['sources'], 'continuation_sha256': sha256_file(args.resume)},
            'budget': {**saved, 'epoch': epoch, 'batch': 0, 'origin_epoch': epoch,
                       'origin_updates': saved['updates']}}


def build_training_data(args, model, metadata):
    if (min(args.epochs, args.batch_size, args.clip_length, args.clip_stride, args.prefetch_factor,
            args.big_call_interval) < 1 or args.num_workers < 0 or
            args.max_batches is not None and args.max_batches < 1 or
            not math.isfinite(args.grad_clip) or args.grad_clip <= 0):
        raise ValueError("Invalid training controls.")
    if bool(args.val_annotations) != bool(args.val_frames_root):
        raise ValueError("Supply both validation annotations and frames root.")
    print(f"Loading training annotations and indexing clips: {args.annotations}", flush=True)
    train = selected_dataset(args, model.model_cfg.image_size, args.clip_length, args.clip_stride, interpolation=model.model_cfg.input_interpolation, paired=model.model_cfg.uses_timm)
    print(f"Training index ready: {len(train.video_ids)} videos, {len(train)} clips.", flush=True)
    train.transform = TrainTransform(model.model_cfg.image_size, interpolation=model.model_cfg.input_interpolation,
                                     paired=model.model_cfg.uses_timm)
    if not metadata:
        metadata.update(category_id_to_label=train.category_id_to_label, label_to_name=train.label_to_name,
                        ssd_config=asdict(model.ssd_cfg))
    validate_native_categories(metadata, train)
    if len(train) < args.batch_size:
        raise ValueError("Training selection has fewer complete clips than a full batch.")
    val = None
    if args.val_annotations:
        print(f"Loading validation annotations: {args.val_annotations}", flush=True)
        val_args = SimpleNamespace(annotations=args.val_annotations, frames_root=args.val_frames_root,
                                   video_ids=args.val_video_ids, max_videos=args.max_val_videos, subset_seed=args.subset_seed)
        val = selected_dataset(val_args, model.model_cfg.image_size, interpolation=model.model_cfg.input_interpolation, paired=model.model_cfg.uses_timm)
        select_evaluation_frames(val, args.val_frames_per_video, args.subset_seed)
        validate_native_categories(metadata, val)
        guard_held_out([data_signature(train)], val)
    return train, val


def train(args):
    requested_stop = vars(args).pop('early_stopping', None)
    expanding = vars(args).pop('expand_subset', False)  # One-time action, never inherited on resume.
    resume = load_checkpoint(args.resume) if args.resume else None
    if resume and not resume.get('budget'):
        raise ValueError('Constant-LR resume is retired; use the archived trainer. Checkpoint evaluation remains supported.')
    early_stopping = resume.get('early_stopping', False) if resume and requested_stop is None else bool(requested_stop)
    source = load_checkpoint(args.init_checkpoint) if args.init_checkpoint and not resume else None
    if resume and (resume['experiment'] != args.experiment or resume['stage'] != 'matched_' + args.task):
        raise ValueError('Resume must use the same matched experiment and stage.')
    seed_training(args.seed)
    model = model_from_matched(resume) if resume else build_model(args.experiment, args.task, args.seed)
    sources, signatures = {}, list(resume['training_signatures']) if resume else []
    if source:
        if source.get('stage') != 'matched_classification' or source.get('experiment') != args.experiment:
            raise ValueError('Detection must initialize from classification of the same experiment.')
        if source.get('initialization_seed') != args.seed:
            raise ValueError('Use the same seed for matched classification and detection.')
        classifier = model_from_matched(source)
        classifier.transfer_to_detector(model)
        sources = {'classification_sha256': sha256_file(args.init_checkpoint)}
        signatures = list(source['training_signatures'])
    data_args = copy(args)
    if args.task == 'classification':
        data_args.clip_length = data_args.clip_stride = 1
    metadata = {} if not (source or resume) else dict(source or resume)
    train_data, val_data = build_training_data(data_args, model, metadata)
    for signature in signatures:
        guard_held_out([signature], val_data)
    own_signature = data_signature(train_data)
    if own_signature not in signatures:
        signatures.append(own_signature)
    definition = {'arguments': {k: v for k, v in saved_arguments(args).items() if k != 'init_checkpoint'},
                  'training': {k: v for k, v in own_signature.items() if k != 'video_directories'},
                  'validation': {k: v for k, v in data_signature(val_data).items() if k != 'video_directories'},
                  'num_workers': args.num_workers}
    if expanding:
        resume = expand_subset_checkpoint(resume, definition, signatures, args)
        resume['budget']['batches_per_epoch'] = min(len(train_data) // args.batch_size,
                                                   args.max_batches or len(train_data))
    if resume and definition != resume['run_definition']:
        raise ValueError('Matched training settings or data changed on resume.')
    destination = Path(args.checkpoint_out)
    best_path = destination.with_name(destination.stem + '_best' + (destination.suffix or '.pt'))
    if best_path.exists():
        stored_best = load_checkpoint(best_path)
        if not resume or stored_best['run_id'] != resume['run_id'] or checkpoint_position(stored_best) > checkpoint_position(resume):
            raise FileExistsError('Best checkpoint belongs to another run or a later training position.')
        if checkpoint_position(stored_best) == checkpoint_position(resume) and (
                stored_best['model'].keys() != resume['model'].keys() or
                any(not torch.equal(v, stored_best['model'][k]) for k, v in resume['model'].items())):
            raise FileExistsError('Best checkpoint contains different model weights.')
    if destination.exists():
        stored = load_checkpoint(destination)
        if not resume or stored['run_id'] != resume['run_id'] or checkpoint_position(stored) != checkpoint_position(resume):
            raise FileExistsError('Output belongs to another run/epoch; use a fresh destination.')
        if stored['model'].keys() != resume['model'].keys() or any(not torch.equal(v, stored['model'][k]) for k, v in resume['model'].items()):
            raise FileExistsError('Output contains different model weights.')
    if not resume and list(destination.parent.glob(destination.stem + '_epoch_*')):
        raise FileExistsError('Numbered checkpoints already exist at this destination.')
    batches = len(train_data) // args.batch_size
    if args.max_batches and args.max_batches > batches:
        raise ValueError(f'Budget requests {args.max_batches} batches but dataset provides only {batches}.')
    if not resume and args.task == 'classification':
        for name in ('big_backbone', 'small_backbone'):
            backbone = getattr(model, name)
            if hasattr(backbone, 'load_pretrained'):
                sources[name] = backbone.load_pretrained()
    model.to(args.device)
    lr = args.learning_rate or (.001 if args.task == 'classification' else .0001)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=.9, momentum=.9, eps=1.)
    if resume:
        restore_optimizer(optimizer, resume['optimizer'], args.learning_rate)
        restore_rng(resume['rng_state'])
        sources, signatures = resume['sources'], resume['training_signatures']
    if args.task == 'classification':
        train_data = ClassificationDataset(train_data, model.model_cfg.backbone, True)
        val_data = ClassificationDataset(val_data, model.model_cfg.backbone, False)

    if expanding:
        # Keep the source best as the baseline even if every new validation is worse.
        atomic_save_checkpoint(resume, best_path, overwrite=False)
        atomic_save_checkpoint(resume, destination, overwrite=False)
    return trainer.train(model, train_data, val_data, optimizer, args, resume=resume,
                         metadata=metadata, sources=sources, signatures=signatures,
                         definition=definition, early_stopping=early_stopping)


if __name__ == '__main__':
    args = parse_args()
    with run_lock(args.checkpoint_out):
        train(args)
