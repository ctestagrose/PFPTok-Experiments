import os
from collections import Counter, defaultdict
from typing import Any, Sequence, Union

import numpy as np
import torch
from transformers import Trainer

from .Loss_Functions import CombinedFocalLabelSmoothingLoss, CombinedFocalLabelSmoothingLossMultiCat
from .Metric_Calculator import MetricsCalculator

metrics_calculator = MetricsCalculator()


def compute_metrics(eval_pred, target_format='binary'):
    logits, labels = eval_pred.predictions, eval_pred.label_ids

    probabilities = torch.sigmoid(torch.tensor(logits)).numpy()

    if target_format == 'multi-cat':
        best_thresholds, _ = metrics_calculator.find_best_threshold(labels, probabilities, target_format)
        predictions = (probabilities >= best_thresholds).astype(int)
        best_threshold_to_report = best_thresholds
    else:
        if probabilities.ndim > 1:
            probabilities = probabilities.squeeze(-1)
        best_threshold, _ = metrics_calculator.find_best_threshold(labels, probabilities, target_format)
        predictions = (probabilities >= best_threshold).astype(int)
        best_threshold_to_report = best_threshold

    f1, accuracy, hamming, jaccard, precision, recall, roc_auc, _, _ = \
        metrics_calculator.calculate_metrics(labels, predictions, probabilities, target_format)

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': roc_auc if roc_auc is not None else 0.0,
        'hamming_loss': hamming,
        'jaccard_score': jaccard,
        'best_threshold': best_threshold_to_report.tolist() if isinstance(best_threshold_to_report, np.ndarray) else best_threshold_to_report
    }


class CustomTrainer(Trainer):
    def __init__(self, *args, target_format='binary', train_labels=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_format = target_format

        device = self.args.device
        epsilon = 1e-7

        if self.target_format == 'multi-cat':
            labels_tensor = torch.tensor(train_labels).float()
            positive_counts = torch.sum(labels_tensor, dim=0)
            total_counts = labels_tensor.size(0)
            positive_freq = positive_counts / (total_counts + epsilon)
            negative_freq = 1.0 - positive_freq

            alpha = torch.clamp(1.0 - positive_freq, min=0.01, max=0.99)
            imbalance_ratio = negative_freq / (positive_freq + epsilon)
            gamma = torch.clamp_min(1.0 + torch.log10(imbalance_ratio + epsilon), min=0.0)

            print(f"Multi-Cat Loss Params - Alpha: {alpha}, Gamma: {gamma}")
            self.loss_fct = CombinedFocalLabelSmoothingLossMultiCat(
                alpha=alpha.to(device),
                gamma=gamma.to(device),
                smoothing=0.1
            )
        else:
            label_counts = Counter(train_labels)
            neg_counts = label_counts.get(0, 0)
            pos_counts = label_counts.get(1, 0)
            total = neg_counts + pos_counts

            alpha_val = 1.0 - (pos_counts / (total + epsilon))
            imbalance_ratio = neg_counts / (pos_counts + epsilon) if pos_counts > 0 else 1.0
            gamma_val = max(0.0, 1.0 + np.log10(imbalance_ratio))

            print(f"Binary Loss Params - Alpha: {alpha_val}, Gamma: {gamma_val}")
            self.loss_fct = CombinedFocalLabelSmoothingLoss(
                alpha=alpha_val,
                gamma=gamma_val,
                smoothing=0.1
            )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        inputs.pop("gene_ids")
        outputs = model(**inputs, output_attentions=False)
        logits = outputs.logits

        if self.target_format == 'multi-cat':
            loss = self.loss_fct(logits, labels.float())
        else:
            loss = self.loss_fct(logits.squeeze(-1), labels.float())

        return (loss, outputs) if return_outputs else loss


def _as_seq_list(x):
    if x is None:
        return []
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, list):
        if len(x) and isinstance(x[0], torch.Tensor):
            return [t.detach().cpu().tolist() for t in x]
        return x
    raise TypeError(f"Unsupported type for gene_ids: {type(x)}")


def attention_token_importance(
    model,
    batch,
    gene_pad_id: Union[int, str] = -100,
    normalize: bool = True,
    use_cls: bool = True
):
    model.eval()
    device = next(model.parameters()).device

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device).float()
    gene_ids_any = batch.get("gene_ids", None)
    gene_ids_list = _as_seq_list(gene_ids_any) if gene_ids_any is not None else None

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )

        layers = [a.float() for a in out.attentions]
        A = torch.stack(layers, dim=0)

        if A.dim() == 5:
            A = A.mean(dim=(0, 2))
        elif A.dim() == 4:
            A = A.mean(dim=0)
        else:
            raise ValueError(f"Unexpected stacked attention shape: {tuple(A.shape)}")

        if use_cls:
            tok_imp = A[:, 0, :]
        else:
            tok_imp = A.mean(dim=1)

        tok_imp = tok_imp * attention_mask
        if normalize:
            denom = tok_imp.sum(dim=1, keepdim=True).clamp_min(1e-12)
            tok_imp = tok_imp / denom

        B, L = tok_imp.shape
        token_scores, gene_scores = [], []
        for b in range(B):
            valid = int(attention_mask[b].sum().item())
            ts = tok_imp[b, :valid].cpu()
            token_scores.append(ts)

            gdict = defaultdict(float)
            if gene_ids_list is not None:
                gids_row: Sequence[Any] = gene_ids_list[b]
                gids = list(gids_row[:valid])
                if len(gids) != ts.numel() and os.environ.get("RANK", "0") == "0":
                    print(f"WARNING: sample {b} gids({len(gids)}) != tokens({ts.numel()}); clipping to min length")
                for s, g in zip(ts.tolist(), gids):
                    if g != gene_pad_id:
                        gdict[g] += float(s)
            gene_scores.append(dict(gdict))

    return token_scores, gene_scores