import hashlib
from tokenizers import Tokenizer, models, pre_tokenizers
from tqdm import tqdm
from collections import Counter
import numpy as np

_CHAR_MAP = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 5}
_BASE = 5
_MOD = (1 << 61) - 1

def md5_hash(window: str) -> int:
    h = hashlib.md5(window.encode('utf-8')).hexdigest()
    return int(h, 16)

def karp_rabin_hash(window: str) -> int:
    h = 0
    for ch in window:
        h = (h * _BASE + _CHAR_MAP.get(ch, 0)) % _MOD
    return h

    
def prefix_free_parse(sequence: str, w: int = 10, d: int = 127, use_simple_hash: bool = True):
    n = len(sequence)
    triggers = []

    if use_simple_hash:
        h = karp_rabin_hash(sequence[:w])
        if h % d == 0:
            triggers.append(0)

        power = pow(_BASE, w - 1, _MOD)

        for i in range(1, n - w + 1):
            left_val  = _CHAR_MAP.get(sequence[i - 1], 0)
            right_val = _CHAR_MAP.get(sequence[i + w - 1], 0)

            h = (h - left_val * power) % _MOD
            h = (h * _BASE + right_val) % _MOD

            if h % d == 0:
                triggers.append(i)

    else:
        for i in range(n - w + 1):
            window = sequence[i : i + w]
            if md5_hash(window) % d == 0:
                triggers.append(i)

    if not triggers or triggers[0] != 0:
        triggers.insert(0, 0)
    if triggers[-1] != n - w:
        triggers.append(n - w)

    phrases = []
    for i in range(len(triggers) - 1):
        start = triggers[i]
        end = triggers[i + 1] + w
        if end <= n:
            phrases.append(sequence[start:end])
        else:
            phrases.append(sequence[start:])

    if phrases:
        non_overlapping = [phrases[0]]
        for ph in phrases[1:]:
            non_overlapping.append(ph[w:] if len(ph) > w else ph)
        return non_overlapping

    return phrases  


class PFPTok:
    def __init__(self, vocab_size=None, w=3, d=117):
        self.vocab_size = vocab_size
        self.w = w
        self.d = d
        self.k = 6
        self.special_tokens = []

    def _build_vocab(self, sorted_phrases, k, special_tokens):
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


    def setup_tokenizer(
        self,
        sequences,
        w,
        d
    ):
        self.w = w
        self.d = d
        print(f"W: {w}, P: {d}")
        self.special_tokens = ["[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]", "[INTB]", "[INTA]", "[GENE]"]

        freq = Counter()
        all_phrases = set()
        for i, seq in tqdm(enumerate(sequences), total=len(sequences),
                           desc=f"Setting up tokenizer with {len(sequences)} sequences"):
            if seq in self.special_tokens:
                continue
            phrases = prefix_free_parse(seq, self.w, self.d)
            freq.update(phrases)
            all_phrases.update(phrases)

        sorted_phrases = sorted(list(all_phrases))
        vocab = self._build_vocab(sorted_phrases, self.k, self.special_tokens)

        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        for token in self.special_tokens:
            token_id = tokenizer.encode(token)
            if token_id is None:
                raise ValueError(f"{token} token was not properly initialized in the tokenizer.")

        return tokenizer


    def save_tokenizer(self, tokenizer, save_path):
        tokenizer.save(save_path)

    def load_tokenizer(self, load_path):
        tokenizer = Tokenizer.from_file(load_path)
        if tokenizer.encode("[UNK]") is None:
            raise ValueError("Loaded tokenizer does not recognize the [UNK] token.")
        return tokenizer
        
    def encode_sequences(
        self,
        sequences,
        tokenizer,
    ):
        unk_encoding = tokenizer.encode("[UNK]")
        unk_id = unk_encoding.ids[0] if hasattr(unk_encoding, "ids") else unk_encoding[0]
        
        gene = sequences[0] if isinstance(sequences, (list, tuple)) else sequences
        
        if gene in self.special_tokens:
            enc_ids = tokenizer.encode(gene)
            tok_id = (enc_ids.ids[0] if hasattr(enc_ids, "ids") else enc_ids[0]) if enc_ids else unk_id
            return [tok_id]
        
        phrases = prefix_free_parse(gene, self.w, self.d)
        if not phrases:
            return []

        phrase_encodings = tokenizer.backend_tokenizer.encode_batch(phrases)
        phrase_ids = [
            (enc.ids[0] if hasattr(enc, "ids") else enc[0]) if enc else unk_id
            for enc in phrase_encodings
        ]
        
        encoded = []
        for phrase, phrase_id in zip(phrases, phrase_ids):
            if not phrase:
                continue
                
            if phrase_id == unk_id:
                # Phrase not in vocab
                encoded.append(unk_id)
            else:
                # Phrase in vocab
                encoded.append(phrase_id)

        return encoded

