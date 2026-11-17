"""Reusable attention-memory training and evaluation routines."""

import torch
from PIL import Image
from torch import nn

from datasets.dataloader import budget_loader, collate_pairs
from datasets.dataset_imagenetvid import TrainingPairs
from datasets.transforms import resize_detection_views
from engine.evaluator import decode, export_evaluation
from scripts.compare_vid import score_predictions
from train_lstm_models import save_matched_checkpoints
from utils.checkpoints import capture_rng
from utils.runtime import tqdm, early_stopping_reached, ATTENTION_MEMORY_NAMES
from vid_categories import vid_to_dataset_labels

def train_batches(student, teacher, loader, optimizer, scheduler, criterion, *, device, epoch):
    """Train one prepared slice of batches; keep batch-normalization statistics frozen."""
    student.train()
    for module in student.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    loss_sum, updates = 0., 0
    for keys, current, targets in tqdm(loader, desc=f'{ATTENTION_MEMORY_NAMES[student.memory]} epoch {epoch}'):
        cache = teacher.keyframe(keys)[0] if student.memory == 'p4' else None
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.
        for position in range(current.shape[1]):
            frame_targets = [sample[position] for sample in targets]
            loss = criterion(student(current[:, position].to(device), cache), frame_targets)['total_loss']
            (loss / current.shape[1]).backward()
            batch_loss += loss.item() / current.shape[1]
        nn.utils.clip_grad_norm_(student.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        updates += 1
        loss_sum += batch_loss
    return loss_sum, updates


def train(student, teacher, pairs, selection, val, optimizer, scheduler, criterion, *,
          loader_options, state, latest, early_stopping=False):
    """Run a prepared budget, validate periodically, and save compatible latest/best checkpoints."""
    step, history = state['updates'], state['history']
    batches, validate_every = state['settings']['batches'], state['settings']['validate_every']
    budget, device = scheduler.T_max, loader_options.device
    while step < budget:
        epoch, offset = divmod(step, batches)
        count = min(batches-offset, validate_every-step % validate_every)
        loader = budget_loader(pairs, loader_options, epoch+1, offset, count, collate_fn=collate_pairs)
        loss_sum, updates = train_batches(student, teacher, loader, optimizer, scheduler, criterion,
                                          device=device, epoch=epoch+1)
        step += updates
        validate = step % validate_every == 0 or step == budget
        if validate:
            metrics = evaluate(student, teacher, selection, val, device)
            history.append(dict(epoch=epoch+1, update=step, train_loss=loss_sum/count,
                                validation={'student_COCO_AP50': metrics['student']['coco_AP50'], **metrics}))
        payload = {**state, 'model': student.state_dict(), 'optimizer': optimizer.state_dict(),
                   'scheduler': scheduler.state_dict(), 'updates': step, 'epoch': epoch+1, 'rng': capture_rng()}
        if early_stopping:
            payload['early_stopping'] = True
        save_matched_checkpoints(payload, latest, select_best=validate)
        print(f'{step}/{budget} updates; loss={loss_sum/count:.4f}; saved {latest}', flush=True)
        if validate:
            print(f"Student COCO AP50={metrics['student']['coco_AP50']:.4f}; "
                  f"whole sequence={metrics['sequence']['coco_AP50']:.4f}", flush=True)
            if early_stopping_reached(payload, early_stopping):
                break

@torch.no_grad()
def evaluate(student, teacher, selection, dataset, device, on_prediction=None):
    student.eval()
    mapping = vid_to_dataset_labels(dataset)
    key_rows, small_rows, small_frames = [], [], []
    previous = None
    for frame in tqdm(selection['frames'], desc='Interval 10', unit='frame'):
        if previous != frame['video_id']:
            step, cache = 0, None
        with Image.open(dataset.frames_root / frame['file_name']) as source:
            image = source.convert('RGB')
        width, height = image.size
        if [width, height] != frame['original_size']:
            raise ValueError('Frame dimensions differ from validation annotations.')
        rows = []
        if step % 10 == 0:
            cache, detections = teacher.keyframe([image], detect=True)
            info = dict(id=frame['image_id'], width=width, height=height)
            for row in teacher.records(detections[0], info, 0.):
                row['category_id'] = mapping[row['category_id']]
                rows.append(row)
        else:
            small_frames.append(frame)
            view = resize_detection_views(image, 160, 'bicubic')[None].to(device)
            pred = decode(student(view, cache), student, 0.)
            for box, label, score in zip(pred['boxes'].tolist(), pred['labels'].tolist(), pred['scores'].tolist()):
                x, y, right, bottom = box
                rows.append(dict(image_id=frame['image_id'], category_id=label, score=score,
                                       bbox=[x*width, y*height, (right-x)*width, (bottom-y)*height]))
        (key_rows if step % 10 == 0 else small_rows).extend(rows)
        if on_prediction is not None:
            on_prediction(frame, rows, step)
        previous, step = frame['video_id'], step + 1
    if not small_frames:
        raise ValueError('Validation needs at least one intervening student frame.')
    return {'student': score_predictions({**selection, 'frames': small_frames}, small_rows),
            'sequence': score_predictions(selection, key_rows + small_rows)}
