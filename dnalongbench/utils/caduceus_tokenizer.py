import os
from typing import List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

DEFAULT_CADUCEUS_MODEL = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"


def build_caduceus_tokenizer(
    model_name: str = DEFAULT_CADUCEUS_MODEL, model_max_length: int = 450_000
):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.model_max_length = model_max_length
    return tok


def save_caduceus_tokenizer(tokenizer, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    tokenizer.save_pretrained(out_dir)


def load_caduceus_tokenizer(tok_dir: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)


def _build_lut(tokenizer) -> Tuple[np.ndarray, int]:
    vocab = tokenizer.get_vocab()
    unk_id = vocab.get("[UNK]", 6)
    lut = np.full(256, unk_id, dtype=np.int64)
    for tok, idx in vocab.items():
        if len(tok) == 1:
            lut[ord(tok.upper())] = idx
            lut[ord(tok.lower())] = idx
    return lut, unk_id


def _crop(ids: np.ndarray, max_length: int, mode: str) -> np.ndarray:
    if max_length is None or len(ids) <= max_length:
        return ids
    if mode == "ends":
        half = max_length // 2
        return np.concatenate([ids[:half], ids[-(max_length - half):]])
    if mode == "center":
        start = (len(ids) - max_length) // 2
        return ids[start : start + max_length]
    if mode == "right":
        return ids[-max_length:]
    return ids[:max_length]


def tokenize_sequences_caduceus(
    prepped_zipped: Sequence,
    tokenizer,
    max_length: int = 450_000,
    crop: str = "ends",
) -> Tuple[List, Tuple[int, int]]:
    lut, unk_id = _build_lut(tokenizer)

    encoded = []
    unk = 0
    non_unk = 0

    for seq, label, seq_id in tqdm(prepped_zipped, desc="Tokenizing (Caduceus)", disable=True):
        arr = np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)
        ids = lut[arr]
        ids = _crop(ids, max_length, crop)

        n_unk = int((ids == unk_id).sum())
        unk += n_unk
        non_unk += int(ids.size) - n_unk

        ids = ids.tolist()
        encoded.append(([ids], label, seq_id, [[-100] * len(ids)]))

    return encoded, (unk, non_unk)


def tokenize_sequences_caduceus_eqtl(
    prepped_zipped,
    tokenizer,
    max_length: int = 450_000,
    crop: str = "ends",
):
    lut, unk_id = _build_lut(tokenizer)

    encoded = []
    unk = 0
    non_unk = 0

    def enc(seq):
        nonlocal unk, non_unk
        arr = np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)
        ids = _crop(lut[arr], max_length, crop)
        n_unk = int((ids == unk_id).sum())
        unk += n_unk
        non_unk += int(ids.size) - n_unk
        return ids.tolist()

    for seq_ref, seq_alt, label, seq_id in tqdm(
        prepped_zipped, desc="Tokenizing eQTL pairs (Caduceus)", disable=True
    ):
        ref_ids = enc(seq_ref)
        alt_ids = enc(seq_alt)
        encoded.append(
            ([ref_ids], [alt_ids], label, seq_id, [[-100] * len(ref_ids)])
        )

    return encoded, (unk, non_unk)
