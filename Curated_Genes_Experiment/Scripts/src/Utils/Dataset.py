import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import random


class SeqDataset(Dataset):
    def __init__(self, sequences, labels, seq_ids, gene_ids, classification_type='multi'):
        self.sequences = sequences
        self.labels = labels
        self.seq_ids = seq_ids
        self.gene_ids = gene_ids
        self.classification_type = classification_type

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        seq_id = self.seq_ids[idx]
        gene_id = self.gene_ids[idx]

        if isinstance(label, (int, float)):
            label_tensor = torch.tensor(label, dtype=torch.float)
        elif isinstance(label, list):
            label_tensor = torch.tensor(label, dtype=torch.float)
        else:
            label_tensor = label

        return {
            'input_ids': torch.tensor(sequence, dtype=torch.long),
            'labels': label_tensor,
            'seq_id': seq_id,
            'gene_ids': gene_id
        }


def dynamic_masking(inputs, mask_prob=0.15, mask_token_id=0, vocab_size=None, max_span_length=5, max_attempts=100):
    batch_size, seq_len = inputs.size()
    num_masked_tokens = int(mask_prob * seq_len)
    masked_inputs = inputs.clone()

    for i in range(batch_size):
        total_masked = 0
        attempts = 0
        spans = []
        while total_masked < num_masked_tokens and attempts < max_attempts:
            span_length = random.randint(1, max_span_length)
            start_idx = random.randint(0, seq_len - span_length)
            end_idx = min(start_idx + span_length, seq_len)
            if torch.any(masked_inputs[i, start_idx:end_idx] == 0):
                attempts += 1
                continue
            if total_masked + (end_idx - start_idx) > num_masked_tokens:
                end_idx = start_idx + (num_masked_tokens - total_masked)
            if start_idx >= end_idx:
                attempts += 1
                continue
            spans.append((start_idx, end_idx))
            total_masked += (end_idx - start_idx)
            attempts += 1

        for start_idx, end_idx in spans:
            mask_choice = torch.rand(end_idx - start_idx, device=inputs.device)
            for j in range(start_idx, end_idx):
                if mask_choice[j - start_idx] < 0.8:
                    masked_inputs[i, j] = mask_token_id
                elif mask_choice[j - start_idx] < 0.9 and vocab_size is not None:
                    masked_inputs[i, j] = torch.randint(1, vocab_size, (1,), device=inputs.device)

    mask = (masked_inputs != inputs)
    return masked_inputs, mask


def collate_fn(batch, classification_type='multi', mask_prob=0.20, MASK_TOKEN=1, PAD_TOKEN=0, VOCAB_SIZE=None):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    gene_ids = [item['gene_ids'] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=PAD_TOKEN)

    max_len = input_ids_padded.size(1)
    gene_ids_padded = []
    for gene_sequence in gene_ids:
        if len(gene_sequence) < max_len:
            padded_genes = gene_sequence[0] + ['PAD'] * (max_len - len(gene_sequence))
        else:
            padded_genes = gene_sequence[:max_len]
        gene_ids_padded.append(padded_genes)

    attention_mask = torch.ones_like(input_ids_padded, dtype=torch.long)
    attention_mask[input_ids_padded == PAD_TOKEN] = 0

    masked_inputs, dynamic_mask_indices = dynamic_masking(
        input_ids_padded,
        mask_prob=mask_prob,
        mask_token_id=MASK_TOKEN,
        vocab_size=VOCAB_SIZE
    )

    if classification_type == 'binary':
        labels_tensor = torch.stack(labels).squeeze(-1) if labels[0].ndim > 0 else torch.stack(labels)
    else:
        labels_tensor = torch.stack(labels)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask,
        'labels': labels_tensor,
        'gene_ids': gene_ids_padded
    }
    
    