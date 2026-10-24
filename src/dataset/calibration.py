"""Baseline next-token behavioral calibration from actual model logits."""
from collections import defaultdict
import torch
from collections.abc import Callable
from tqdm import tqdm
from .generate import length_batches


@torch.inference_mode()
def calibrate(model, records: list[dict], batch_size: int, chance_tolerance: float = 0.1, existing: list[dict] | None = None, checkpoint: Callable[[list[dict]], None] | None = None) -> tuple[list[dict], list[dict]]:
    """Measure forced-choice accuracy and absolute correct-name probabilities."""
    behavioral = [r for r in records if r['correct_token_id'] is not None]
    rows = list(existing or [])
    completed = {row["id"] for row in rows}
    expected = {r["id"] for r in behavioral}
    if len(completed) != len(rows) or not completed <= expected:
        raise ValueError("Invalid calibration checkpoint IDs")
    batches = list(length_batches([r for r in behavioral if r["id"] not in completed], batch_size))
    for batch in tqdm(batches, desc='Calibration'):
        ids = torch.tensor([r['token_ids'] for r in batch], device=next(model.parameters()).device)
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[:, -1].float()
        probs = logits.softmax(-1)
        for i, record in enumerate(batch):
            correct, distractor = record['correct_token_id'], record['distractor_token_id']
            difference = (logits[i, correct] - logits[i, distractor]).item()
            rows.append({'id': record['id'], 'split': record['split'], 'task': record['task'],
                'template_id': record['template_id'], 'num_distractors': record['num_distractors'],
                'accuracy': 1.0 if difference > 0 else (0.5 if difference == 0 else 0.0),
                'logit_difference': difference, 'correct_probability': probs[i, correct].item(),
                'distractor_probability': probs[i, distractor].item()})
        if checkpoint is not None:
            checkpoint(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['split'], row['task'], row['template_id'])].append(row)
        groups[(row['split'], row['task'], '__task__')].append(row)
    report = []
    for (split, task, template), group in sorted(groups.items()):
        accuracy = sum(r['accuracy'] for r in group) / len(group)
        report.append({'split': split, 'task': task, 'template_id': template, 'n': len(group),
            'accuracy': accuracy, 'near_chance': abs(accuracy - 0.5) <= chance_tolerance,
            'saturated': accuracy >= 0.95,
            'mean_correct_probability': sum(r['correct_probability'] for r in group)/len(group),
            'mean_logit_difference': sum(r['logit_difference'] for r in group)/len(group)})
    return rows, report
