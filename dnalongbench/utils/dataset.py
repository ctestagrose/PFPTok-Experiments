import torch
from torch.utils.data import Dataset, DataLoader
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

class SeqDatasetEQTL(Dataset):
    def __init__(self, ref_seqs, alt_seqs, labels, seq_ids, gene_ids, classification_type='binary'):
        self.ref_seqs = ref_seqs
        self.alt_seqs = alt_seqs
        self.labels = labels
        self.seq_ids = seq_ids
        self.gene_ids = gene_ids
        self.classification_type = classification_type

    def __len__(self):
        return len(self.ref_seqs)

    def __getitem__(self, idx):
        label = self.labels[idx]
        label_tensor = torch.tensor(label, dtype=torch.float)
        return {
            'input_ids':      torch.tensor(self.ref_seqs[idx], dtype=torch.long),
            'input_ids_alt':  torch.tensor(self.alt_seqs[idx], dtype=torch.long),
            'labels':         label_tensor,
            'seq_id':         self.seq_ids[idx],
            'gene_ids':       self.gene_ids[idx],
        }


def dynamic_masking(inputs, mask_prob=0.15, mask_token_id=0, vocab_size=None, max_span_length=5, max_attempts=100):
    batch_size, seq_len = inputs.size()
    num_masked_tokens = int(mask_prob * seq_len)
    masked_inputs = inputs.clone()
    for i in range(batch_size):
        spans = []
        total_masked = 0
        attempts = 0
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
                    masked_inputs[i, j] = torch.randint(1, vocab_size, (1,),
                                                        device=inputs.device)  
    mask = (masked_inputs != inputs)
    return masked_inputs, mask


def collate_fn(batch, classification_type='multi', mask_prob=0.20, MASK_TOKEN=1, PAD_TOKEN=0, VOCAB_SIZE=None):
    paired = 'input_ids_alt' in batch[0]

    def pad_and_mask(sequences):
        padded = pad_sequence(sequences, batch_first=True, padding_value=PAD_TOKEN)
        mask = torch.ones_like(padded, dtype=torch.long)
        mask[padded == PAD_TOKEN] = 0
        return padded, mask

    labels = [item['labels'] for item in batch]

    if paired:
        ref_ids,  ref_mask  = pad_and_mask([item['input_ids']     for item in batch])
        alt_ids,  alt_mask  = pad_and_mask([item['input_ids_alt'] for item in batch])

        max_len  = ref_ids.size(1)
        gene_ids = [item['gene_ids'][0] for item in batch]
        gene_ids_padded = [
            g + [-100] * (max_len - len(g)) if len(g) < max_len else g[:max_len]
            for g in gene_ids
        ]

        if classification_type == 'binary':
            labels_tensor = torch.stack(labels).squeeze(-1) if labels[0].ndim > 0 else torch.stack(labels)
        else:
            labels_tensor = torch.stack(labels)

        return {
            'input_ids':          ref_ids,
            'attention_mask':     ref_mask,
            'input_ids_alt':      alt_ids,
            'attention_mask_alt': alt_mask,
            'labels':             labels_tensor,
            'gene_ids':           gene_ids_padded,
        }

    else:
        input_ids = [item['input_ids'] for item in batch]
        gene_ids  = [item['gene_ids']  for item in batch]

        input_ids_padded, attention_mask = pad_and_mask(input_ids)
        max_len = input_ids_padded.size(1)

        gene_ids_padded = []
        for gene_sequence in gene_ids:
            g = gene_sequence[0] if isinstance(gene_sequence[0], list) else gene_sequence
            if len(g) < max_len:
                gene_ids_padded.append(g + ['PAD'] * (max_len - len(g)))
            else:
                gene_ids_padded.append(g[:max_len])

        masked_inputs, _ = dynamic_masking(
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
            'input_ids':      input_ids_padded,
            'attention_mask': attention_mask,
            'labels':         labels_tensor,
            'gene_ids':       gene_ids_padded,
        }
    
    