import hashlib
import random
from collections import Counter
from itertools import product
from typing import Literal, List

from tokenizers import Tokenizer, models, pre_tokenizers
from tqdm import tqdm

_CHAR_MAP = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 5}
_BASE = 5
_MOD = (1 << 61) - 1

def karp_rabin_hash(window: str) -> int:
    h = 0
    for ch in window:
        h = (h * _BASE + _CHAR_MAP.get(ch, 0)) % _MOD
    return h


def prefix_free_parse(sequence: str, w: int = 10, d: int = 127) -> List[str]:
    n = len(sequence)
    triggers = []

    h = karp_rabin_hash(sequence[:w])
    if h % d == 0:
        triggers.append(0)

    power = pow(_BASE, w - 1, _MOD)
    for i in range(1, n - w + 1):
        left_val = _CHAR_MAP.get(sequence[i - 1], 0)
        right_val = _CHAR_MAP.get(sequence[i + w - 1], 0)
        h = (h - left_val * power) % _MOD
        h = (h * _BASE + right_val) % _MOD
        if h % d == 0:
            triggers.append(i)

    # Ensure start and end are included
    if not triggers or triggers[0] != 0:
        triggers.insert(0, 0)
    if triggers[-1] != n - w:
        triggers.append(n - w)

    # Build overlapping phrases, then strip overlap
    phrases = []
    for i in range(len(triggers) - 1):
        start = triggers[i]
        end = triggers[i + 1] + w
        phrases.append(sequence[start:min(end, n)])

    if phrases:
        non_overlapping = [phrases[0]]
        for ph in phrases[1:]:
            non_overlapping.append(ph[w:] if len(ph) > w else ph)
        return non_overlapping

    return phrases


class TokenizerManager:

    def __init__(self, vocab_size=None, w=3, d=117):
        self.vocab_size = vocab_size
        self.w = w
        self.d = d
        self.k = 6  # k-mer size for fallback
        self.special_tokens = []
        self.tokenizer = None

    def _build_vocab(self, sorted_phrases, special_tokens):
        """Build vocabulary from phrases, k-mers, and special tokens."""
        vocab = {}
        next_id = 0

        for ph in sorted_phrases:
            if ph not in vocab:
                vocab[ph] = next_id
                next_id += 1

        for tok in special_tokens:
            if tok not in vocab:
                vocab[tok] = next_id
                next_id += 1

        return vocab

    def setup_tokenizer(self, sequences, w, d, *,
                        min_count_uncommon: int = 2,
                        rare_quantile: float = 0.20):
        self.w = w
        self.d = d
        self.min_count_uncommon = min_count_uncommon
        self.rare_quantile = rare_quantile
        print(f"PFP parameters — W: {w}, D: {d}")

        self.special_tokens = [
            "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]",
            "[INTB]", "[INTA]", "[GENE]"
        ]

        freq = Counter()
        all_phrases = set()
        for seq in tqdm(sequences, desc="Building PFP vocabulary"):
            for gene in seq:
                if gene not in self.special_tokens:
                    phrases = prefix_free_parse(gene, self.w, self.d)
                    freq.update(phrases)
                    all_phrases.update(phrases)

        self.phrase_freq = dict(freq)
        self.total_phrase_count = sum(freq.values())

        counts = sorted(freq.values())
        if counts:
            quantile_cutoff = counts[max(0, int(len(counts) * self.rare_quantile) - 1)]
        else:
            quantile_cutoff = 0
        cutoff = max(self.min_count_uncommon, quantile_cutoff)
        self.uncommon_phrases = {ph for ph, c in freq.items() if c <= cutoff}

        print(f"Unique phrases: {len(all_phrases)}")
        print(f"Uncommon phrases (count <= {cutoff}): {len(self.uncommon_phrases)}")

        # Build vocab and tokenizer
        sorted_phrases = sorted(all_phrases)
        vocab = self._build_vocab(sorted_phrases, self.special_tokens)
        print(f"Vocabulary size: {len(vocab)}")

        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        # Sanity check
        for token in self.special_tokens:
            if tokenizer.encode(token) is None:
                raise ValueError(f"{token} not properly initialized in tokenizer.")

        self.tokenizer = tokenizer
        return tokenizer

    def save_tokenizer(self, tokenizer, save_path):
        tokenizer.save(save_path)

    def load_tokenizer(self, load_path):
        tokenizer = Tokenizer.from_file(load_path)
        if tokenizer.encode("[UNK]") is None:
            raise ValueError("Loaded tokenizer does not recognize [UNK].")
        self.tokenizer = tokenizer
        return tokenizer

    def encode_sequences_genes(self, sequences, tokenizer,
                               genes_in_this_isolate, seed=None):
        if seed is not None:
            random.seed(seed)

        encoded_sequences = []
        unk_count = 0
        non_unk_count = 0
        gene_mapping = []
        excluded = set(self.special_tokens)

        unk_enc = tokenizer.encode("[UNK]")
        unk_id = unk_enc.ids[0] if hasattr(unk_enc, "ids") else unk_enc[0]

        def _ids(enc):
            return enc.ids if hasattr(enc, "ids") else enc

        for index, entry in enumerate(sequences):
            gene = entry[0]
            encoded = []

            if gene not in excluded:
                phrases = prefix_free_parse(gene, self.w, self.d)

                for phrase in phrases:
                    if not phrase:
                        continue
                    if isinstance(phrase, list):
                        phrase = "".join(phrase)

                    ids = _ids(tokenizer.encode(phrase))
                    is_known = ids and ids[0] != unk_id

                    if not is_known:
                        encoded.append(unk_id)
                        unk_count += 1
                    else:
                        encoded.append(ids[0])
                        gene_mapping.extend([genes_in_this_isolate[index]] * len(ids))
                        non_unk_count += 1
            else:
                ids = _ids(tokenizer.encode(gene))
                tok = ids[0] if ids else unk_id
                encoded.append(tok)
                if tok == unk_id:
                    unk_count += 1
                else:
                    non_unk_count += 1

            encoded_sequences.append(encoded)

        combined = [tok for sublist in encoded_sequences for tok in sublist]
        return combined, gene_mapping, (unk_count, non_unk_count)
