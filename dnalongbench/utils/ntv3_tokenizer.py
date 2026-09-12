import os
from typing import List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

DEFAULT_NTV3_MODEL = "InstaDeepAI/NTv3_100M_pre"
DOWNSAMPLE_MULTIPLE = 128


def round_length(n: int, mode: str = "pad", multiple: int = DOWNSAMPLE_MULTIPLE) -> int:
    if n % multiple == 0:
        return n
    if mode == "pad":
        return ((n // multiple) + 1) * multiple
    if mode == "truncate":
        return (n // multiple) * multiple
    raise ValueError(f"mode must be 'pad' or 'truncate', got {mode!r}")


def build_ntv3_tokenizer(model_name: str = DEFAULT_NTV3_MODEL):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def save_ntv3_tokenizer(tokenizer, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    tokenizer.save_pretrained(out_dir)


def load_ntv3_tokenizer(tok_dir: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)


def _build_lut(tokenizer) -> Tuple[np.ndarray, int, int]:
    vocab = tokenizer.get_vocab()
    unk_id = vocab.get("<unk>", vocab.get("[UNK]", 0))
    n_id = vocab.get("N")
    if n_id is None:
        raise RuntimeError(f"No 'N' token in NTv3 vocab: {sorted(vocab)}")
    lut = np.full(256, unk_id, dtype=np.int64)
    for tok, idx in vocab.items():
        if len(tok) == 1:
            lut[ord(tok.upper())] = idx
            lut[ord(tok.lower())] = idx
    return lut, unk_id, n_id


def _fit_length(ids: np.ndarray, target: int, n_id: int, mode: str) -> np.ndarray:
    if len(ids) == target:
        return ids
    if len(ids) < target:
        return np.concatenate([ids, np.full(target - len(ids), n_id, dtype=ids.dtype)])
    if mode == "truncate":
        return ids[:target]
    return ids[:target]


def tokenize_sequences_ntv3(
    prepped_zipped: Sequence,
    tokenizer,
    max_length: int = 450_000,
    length_mode: str = "pad",
) -> Tuple[List, Tuple[int, int]]:
    lut, unk_id, n_id = _build_lut(tokenizer)
    target = round_length(max_length, length_mode)

    encoded, unk, non_unk = [], 0, 0
    for seq, label, seq_id in tqdm(prepped_zipped, desc="Tokenizing (NTv3)", disable=True):
        arr = np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)
        ids = _fit_length(lut[arr], target, n_id, length_mode)

        n_unk = int((ids == unk_id).sum())
        unk += n_unk
        non_unk += int(ids.size) - n_unk

        ids = ids.tolist()
        encoded.append(([ids], label, seq_id, [[-100] * len(ids)]))

    return encoded, (unk, non_unk)


def tokenize_sequences_ntv3_eqtl(
    prepped_zipped: Sequence,
    tokenizer,
    max_length: int = 450_000,
    length_mode: str = "pad",
) -> Tuple[List, Tuple[int, int]]:
    lut, unk_id, n_id = _build_lut(tokenizer)
    target = round_length(max_length, length_mode)

    encoded, unk, non_unk = [], 0, 0

    def enc(seq):
        nonlocal unk, non_unk
        arr = np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)
        ids = _fit_length(lut[arr], target, n_id, length_mode)
        n_unk = int((ids == unk_id).sum())
        unk += n_unk
        non_unk += int(ids.size) - n_unk
        return ids.tolist()

    for seq_ref, seq_alt, label, seq_id in tqdm(
        prepped_zipped, desc="Tokenizing eQTL pairs (NTv3)", disable=True
    ):
        ref_ids = enc(seq_ref)
        alt_ids = enc(seq_alt)
        encoded.append(([ref_ids], [alt_ids], label, seq_id, [[-100] * len(ref_ids)]))

    return encoded, (unk, non_unk)
