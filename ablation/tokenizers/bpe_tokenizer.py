import json
import os
import random
from collections import Counter
from typing import Dict
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, Regex

DNA_ALPHABET = list("ACGTN")


def _dna_pretok():
    return pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r"[^ACGTN]+"), behavior="removed"),
    ])


def _iter_training_genes(sequences,
                         max_total_chars=20_000_000,
                         shuffle=True,
                         seed=123,
                         window=1000,  # chunk size
                         stride=1000):  # overlap
    rng = random.Random(seed)
    genes = [g for seq in sequences for g in seq if isinstance(g, str)]
    if shuffle:
        rng.shuffle(genes)

    used = 0
    for g in genes:
        if not g:
            continue
        s = ''.join(ch for ch in g.upper() if ch in "ACGTN")
        if not s:
            continue

        for i in range(0, len(s), stride):
            if used >= max_total_chars:
                return
            chunk = s[i:i + window]
            if len(chunk) == 0:
                continue
            yield chunk
            used += len(chunk)
            if i + window >= len(s):
                break


class TokenizerManagerBPE:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "[MASK]": 4,
            "[INTB]": 5,
            "[INTA]": 6,
            "[GENE]": 7
        }
        self.tokenizer = None
        self.wrapped_tokenizer = None

        # For frequency tracking (optional)
        self.token_freq: Dict[str, int] = {}
        self.uncommon_tokens: set = set()
        self.min_count_uncommon = 2
        self.rare_quantile = 0.20

    def setup_tokenizer(self, sequences, vocab_size=None,
                        min_frequency=1, max_total_chars=20_000_000,
                        window=1000, stride=1000):
        if vocab_size:
            self.vocab_size = vocab_size

        print(f"Setting up DNA-BPE vocab={self.vocab_size}, "
              f"min_freq={min_frequency}, max_chars={max_total_chars}, "
              f"window={window}, stride={stride}")

        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = _dna_pretok()

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(self.special_tokens.keys()),
            show_progress=True,
            initial_alphabet=list("ACGTN"),
            limit_alphabet=5,
        )

        iterator = _iter_training_genes(
            sequences,
            max_total_chars=max_total_chars,
            window=window,
            stride=stride
        )

        tokenizer.train_from_iterator(iterator, trainer=trainer)

        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", tokenizer.token_to_id("[CLS]")),
                ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ],
        )

        self.tokenizer = tokenizer
        self.wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            additional_special_tokens=["[INTB]", "[INTA]", "[GENE]"],
        )

        sample_iter = _iter_training_genes(sequences, max_total_chars=2_000_000, shuffle=False)
        self._collect_token_frequencies(sample_iter)

        print(f"Done. Final vocab size: {self.wrapped_tokenizer.vocab_size}")
        return self.wrapped_tokenizer

    def _collect_token_frequencies(self, sequences):
        if not self.wrapped_tokenizer:
            return

        freq = Counter()

        for seq in tqdm(sequences, desc="Collecting token frequencies (sample)"):
            tokens = self.wrapped_tokenizer.tokenize(seq)
            freq.update(tokens)

        self.token_freq = dict(freq)

        if freq:
            counts = sorted(freq.values())
            cutoff_idx = max(0, int(len(counts) * self.rare_quantile) - 1)
            cutoff = max(self.min_count_uncommon, counts[cutoff_idx] if counts else 0)
            self.uncommon_tokens = {tok for tok, c in freq.items() if c <= cutoff}

            print(f"Token stats: {len(self.token_freq)} unique tokens sampled")
            print(f"Uncommon tokens: {len(self.uncommon_tokens)} (threshold: {cutoff})")

    def encode_sequences(
            self,
            sequences,
            tokenizer,
            genes_in_this_isolate,
            split: str = "train",
            dropout_prob: float = 0.0,
            seed: int = None,
    ):
        if seed is not None:
            random.seed(seed)

        encoded_sequences = []
        gene_mapping = []

        unk_count = 0
        non_unk_count = 0

        unk_id = tokenizer.unk_token_id

        for index, entry in enumerate(sequences):
            gene = entry[0] if isinstance(entry, list) else entry

            if gene in self.special_tokens:
                ids = tokenizer.encode(gene, add_special_tokens=False)
                encoded_sequences.extend(ids)
                gene_mapping.extend([genes_in_this_isolate[index]] * len(ids))
                non_unk_count += len(ids)
            else:
                ids = tokenizer.encode(gene, add_special_tokens=False)

                if split == "train" and dropout_prob > 0:
                    processed_ids = []
                    for token_id in ids:
                        if random.random() < dropout_prob:
                            continue
                        processed_ids.append(token_id)

                    if not processed_ids:
                        processed_ids = [unk_id]

                    ids = processed_ids

                unk_count += sum(1 for t in ids if t == unk_id)
                non_unk_count += sum(1 for t in ids if t != unk_id)

                encoded_sequences.extend(ids)
                gene_mapping.extend([genes_in_this_isolate[index]] * len(ids))

        return encoded_sequences, gene_mapping, (unk_count, non_unk_count)

    def save_tokenizer(self, tokenizer, save_path):
        tokenizer.save_pretrained(os.path.dirname(save_path))

        metadata = {
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'token_freq_sample': dict(list(self.token_freq.items())[:100]),  # Save sample
            'uncommon_tokens_count': len(self.uncommon_tokens),
            'min_count_uncommon': self.min_count_uncommon,
            'rare_quantile': self.rare_quantile,
        }

        metadata_path = save_path.replace('.json', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved tokenizer to {os.path.dirname(save_path)}")

    def load_tokenizer(self, load_path):
        load_dir = os.path.dirname(load_path)
        self.wrapped_tokenizer = PreTrainedTokenizerFast.from_pretrained(load_dir)

        metadata_path = load_path.replace('.json', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.vocab_size = metadata.get('vocab_size', self.vocab_size)
                self.special_tokens = metadata.get('special_tokens', self.special_tokens)
                self.min_count_uncommon = metadata.get('min_count_uncommon', 2)
                self.rare_quantile = metadata.get('rare_quantile', 0.20)
            print(f"Loaded tokenizer with vocab size: {self.wrapped_tokenizer.vocab_size}")

        return self.wrapped_tokenizer