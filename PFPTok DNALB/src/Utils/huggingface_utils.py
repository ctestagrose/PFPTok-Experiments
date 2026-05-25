from collections import Counter
import os
import numpy as np
from transformers import Trainer, get_cosine_with_hard_restarts_schedule_with_warmup

from .Loss_Functions import CombinedFocalLabelSmoothingLoss
from .Metric_Calculator import MetricsCalculator
import torch
from collections import defaultdict
from typing import Any, List, Sequence, Union, Optional


metrics_calculator = MetricsCalculator()

def compute_metrics(eval_pred, target_format='binary'):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    
    probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
    
    if probabilities.ndim > 1:
        probabilities = probabilities.squeeze(-1)
    best_threshold, _ = metrics_calculator.find_best_threshold(labels, probabilities, target_format)
    predictions = (probabilities >= best_threshold).astype(int)
    best_threshold_to_report = best_threshold

    f1, accuracy, hamming, jaccard, precision, recall, roc_auc, confusion, _, aucpr = \
        metrics_calculator.calculate_metrics(
            labels,
            predictions,
            probabilities,
            target_format
        )
        
    print(confusion)
        
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': roc_auc if roc_auc is not None else 0.0,
        'aucpr': aucpr if aucpr is not None else 0.0,
        'hamming_loss': hamming,
        'jaccard_score': jaccard,
        'best_threshold': best_threshold_to_report.tolist() if isinstance(best_threshold_to_report, np.ndarray) else best_threshold_to_report
    }

class CustomTrainer(Trainer):
    def __init__(self, *args, target_format='binary', train_labels=None, **kwargs):
         super().__init__(*args, **kwargs)
         self.target_format = target_format

         device = self.args.device

         label_counts = Counter(train_labels)
         neg_counts = label_counts.get(0, 0)
         pos_counts = label_counts.get(1, 0)
         total = neg_counts + pos_counts
         epsilon = 1e-7

         alpha_val = pos_counts / (total + epsilon) 
         alpha_val = 1.0 - alpha_val 

         imbalance_ratio = neg_counts / (pos_counts + epsilon) if pos_counts > 0 else 1.0
         gamma_val = max(0.0, 1.0 + np.log10(imbalance_ratio))

         print(f"Binary Loss Params - Alpha: {alpha_val}, Gamma: {gamma_val}") 
         self.loss_fct = CombinedFocalLabelSmoothingLoss(
             alpha=alpha_val,
             gamma=gamma_val,
             smoothing=0.1 # Adjust smoothing if needed
         )


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
    
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            input_ids_alt=inputs.get("input_ids_alt"),
            attention_mask_alt=inputs.get("attention_mask_alt"),
        )
        logits = outputs.logits

        loss = self.loss_fct(logits.squeeze(-1), labels.float())

        return (loss, outputs) if return_outputs else loss
        


def _as_seq_list(x):
    if x is None: return []
    if isinstance(x, torch.Tensor): return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):   return x.tolist()
    if isinstance(x, list):
        if len(x) and isinstance(x[0], torch.Tensor):
            return [t.detach().cpu().tolist() for t in x]
        return x
    raise TypeError(f"Unsupported type for gene_ids: {type(x)}")

    

