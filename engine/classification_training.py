"""Reusable engine for multi-label classification training and validation, shared by standalone and joint models."""
import torch
from torch.utils.data import DataLoader

from datasets.classification import classification_collate
from engine.evaluator import classification_metrics
from utils.runtime import tqdm


def classification_epoch(model, dataset, args, optimizer=None, loader=None):
    training = optimizer is not None
    model.train(training)
    if loader is None:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=training, drop_last=training,
                            num_workers=args.num_workers, collate_fn=classification_collate)
    total, samples = 0., 0
    scores, labels = [], []
    with torch.set_grad_enabled(training):
        for index, (frames, target) in enumerate(tqdm(loader, disable=args.no_progress, desc='Classification train' if training else 'Classification val'), 1):
            frames, target = frames.to(args.device), target.to(args.device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(frames, steps=args.steps)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target[:, None].expand_as(logits))
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite classification loss')
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
                optimizer.step()
            total += float(loss.detach()) * len(target)
            samples += len(target)
            if not training:
                scores.append(logits[:, -1].detach().cpu())
                labels.append(target.cpu())
            if args.max_batches and index >= args.max_batches:
                break
    if not samples:
        raise ValueError('No classification batches; check dataset size and batch-size')
    metrics = classification_metrics(torch.cat(scores), torch.cat(labels)) if not training else {}
    return {'bce_loss': total / samples, 'samples': samples, **metrics}
