from __future__ import annotations
import os
from typing import List, Tuple, Optional
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers import Regex
import re


HYENA_VOCAB = {
    "[PAD]": 0,
    "[MASK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "[UNK]": 4,
    "N": 5,
    "A": 6,
    "C": 7,
    "G": 8,
    "T": 9,
    "a": 6,
    "c": 7,
    "g": 8,
    "t": 9,
    "n": 5,
}

SPECIAL_TOKENS = ["[PAD]", "[MASK]", "[CLS]", "[SEP]", "[UNK]"]


def build_hyena_tokenizer(
    model_max_length: int = 450_000,
) -> PreTrainedTokenizerFast:
    unique_vocab = {k: v for k, v in HYENA_VOCAB.items() if k == k.upper() or k in SPECIAL_TOKENS}

    hf_tokenizer = Tokenizer(WordLevel(vocab=unique_vocab, unk_token="[UNK]"))

    hf_tokenizer.pre_tokenizer = Split(
        pattern=Regex("."),
        behavior="isolated",
    )

    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=hf_tokenizer,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        model_max_length=model_max_length,
    )
    return wrapped


def save_hyena_tokenizer(tokenizer: PreTrainedTokenizerFast, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_pretrained(save_dir)
    print(f"HyenaDNA tokenizer saved to {save_dir}")


def load_hyena_tokenizer(save_dir: str) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast.from_pretrained(save_dir)


def encode_single_sequence(
    sequence: str,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 450_000,
    truncation: bool = True,
) -> List[int]:
    seq_upper = sequence.upper()
    seq_clean = re.sub(r"[^ACGTN]", "N", seq_upper)

    encoding = tokenizer(
        seq_clean,
        add_special_tokens=True,
        truncation=truncation,
        max_length=max_length,
        return_token_type_ids=False,
        return_attention_mask=False,
    )
    return encoding["input_ids"]


def tokenize_sequences_hyena(
    prepped_zipped: List[Tuple],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 450_000,
    truncation: bool = True,
) -> Tuple[List[Tuple], Tuple[int, int]]:
    encoded = []
    unk_id  = tokenizer.unk_token_id
    unk     = 0
    non_unk = 0

    for entry in tqdm(prepped_zipped, desc="Tokenizing (HyenaDNA char-level)"):
        if len(entry) == 3:
            seq, label, seq_id = entry
        else:
            seq, label, seq_id = entry[0], entry[1], entry[2]

        ids = encode_single_sequence(seq, tokenizer, max_length, truncation)

        for t in ids:
            if t == unk_id:
                unk += 1
            else:
                non_unk += 1

        gene_map = [-100] * len(ids)

        encoded.append(([ids], label, seq_id, [gene_map]))

    return encoded, (unk, non_unk)

