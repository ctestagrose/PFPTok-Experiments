import hashlib
from tokenizers import Tokenizer, models, pre_tokenizers
from collections import Counter
from tqdm import tqdm

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


def generate_all_kmers(k, alphabet='ATCGN'):
    from itertools import product

    all_kmers = []

    for length in range(1, k + 1):
        for combination in product(alphabet, repeat=length):
            kmer = ''.join(combination)
            all_kmers.append(kmer)

    return all_kmers


def prefix_free_parse(sequence: str, w: int = 10, d: int = 127, use_simple_hash: bool = True):
    n = len(sequence)
    triggers = []

    if use_simple_hash:
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

    else:
        for i in range(n - w + 1):
            window = sequence[i: i + w]
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


class TokenizerManager:
    def __init__(self, vocab_size=None, w=3, d=117):
        self.vocab_size = vocab_size
        self.w = w
        self.d = d
        self.k = 6
        self.special_tokens = []
        self.phrase_freq: dict[str, int] = {}
        self.total_phrase_count: int = 0
        self.uncommon_phrases: set[str] = set()
        self.min_count_uncommon = 2
        self.rare_quantile = 0.20
        self.tokenizer = None

    def _build_vocab(self, sorted_phrases, k, special_tokens):
        vocab = {}
        next_id = 0
        kmers = generate_all_kmers(k)

        for ph in sorted_phrases:
            if ph not in vocab:
                vocab[ph] = next_id
                next_id += 1

        for kmer in kmers:
            if kmer not in vocab:
                vocab[kmer] = next_id
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

        for seq in tqdm(sequences, desc=f"Setting up tokenizer with {len(sequences)} sequences"):
            for gene in seq:
                if gene not in self.special_tokens:
                    phrases = prefix_free_parse(gene, self.w, self.d)
                    freq.update(phrases)
                    all_phrases.update(phrases)

        self.phrase_freq = dict(freq)
        self.total_phrase_count = sum(freq.values())

        sorted_phrases = sorted(list(all_phrases))
        vocab = self._build_vocab(sorted_phrases, self.k, self.special_tokens)

        print(f"Final vocabulary size: {len(vocab)}")

        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        for token in self.special_tokens:
            token_id = tokenizer.encode(token)
            if token_id is None:
                raise ValueError(f"{token} token was not properly initialized in the tokenizer.")

        self.tokenizer = tokenizer
        return tokenizer

    def save_tokenizer(self, tokenizer, save_path):
        tokenizer.save(save_path)

    def load_tokenizer(self, load_path):
        tokenizer = Tokenizer.from_file(load_path)
        if tokenizer.encode("[UNK]") is None:
            raise ValueError("Loaded tokenizer does not recognize the [UNK] token.")
        self.tokenizer = tokenizer
        return tokenizer

    def encode_sequences(
            self,
            sequences,
            seed: int = None,
    ):
        import random
        if seed is not None:
            random.seed(seed)

        encoded_sequences = []

        unk_id = self.tokenizer.token_to_id("[UNK]")
        excluded = set(self.special_tokens)
        unk_count = 0
        non_unk_count = 0

        def _ids_from(enc):
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

                    tok_id = self.tokenizer.token_to_id(phrase)
                    if tok_id is None:
                        encoded.append(unk_id)
                        unk_count += 1
                    else:
                        encoded.append(tok_id)
                        non_unk_count += 1
            else:
                ids = _ids_from(self.tokenizer.encode(gene))
                tok = ids[0] if ids else unk_id
                encoded.append(tok)
                if tok == unk_id:
                    unk_count += 1
                else:
                    non_unk_count += 1

            encoded_sequences.append(encoded)

        combined_list = [item for sublist in encoded_sequences for item in sublist]

        return combined_list, (unk_count, non_unk_count)
