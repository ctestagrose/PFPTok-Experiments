import json
import os
import random
import tempfile
from collections import Counter
from typing import Dict
import sentencepiece as spm
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

class TokenizerManagerUnigram:
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
        self.sp_model = None
        self.wrapped_tokenizer = None

        self.token_freq: Dict[str, int] = {}
        self.uncommon_tokens: set = set()
        self.min_count_uncommon = 2
        self.rare_quantile = 0.20

    def setup_tokenizer(self, sequences, vocab_size=None):
        model_type = 'unigram'
        if vocab_size:
            self.vocab_size = vocab_size

        print(f"Setting up SentencePiece tokenizer with vocab_size={self.vocab_size}, model_type={model_type}")

        training_data = []
        for seq in tqdm(sequences, desc="Preparing sequences for SentencePiece"):
            for gene in seq:
                if gene not in self.special_tokens:
                    print(len(gene))
                    training_data.append(gene)

        print(f"Total sequences: {len(training_data)}")

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_file = f.name
            for gene in training_data:
                f.write(gene + '\n')

        try:
            user_defined_symbols = [
                token for token in self.special_tokens.keys()
                if token not in ["[UNK]", "[PAD]"]
            ]

            with tempfile.TemporaryDirectory() as tmpdir:
                model_prefix = os.path.join(tmpdir, 'sp_model')

                def _train_sp(vocab_size):
                    spm.SentencePieceTrainer.train(
                        input=temp_file,
                        model_prefix=model_prefix,
                        vocab_size=vocab_size,
                        model_type=model_type,
                        character_coverage=1.0,
                        pad_id=self.special_tokens["[PAD]"],
                        unk_id=self.special_tokens["[UNK]"],
                        bos_id=-1,
                        eos_id=-1,
                        pad_piece="[PAD]",
                        unk_piece="[UNK]",
                        user_defined_symbols=user_defined_symbols,
                        num_threads=os.cpu_count(),
                        train_extremely_large_corpus=True,
                        max_sentencepiece_length=64,
                        split_by_whitespace=False,
                        split_by_unicode_script=False,
                    )

                import re
                try:
                    _train_sp(self.vocab_size)
                except RuntimeError as e:
                    match = re.search(r'<= (\d+)', str(e))
                    if match:
                        max_vocab = int(match.group(1))
                        print(f"Requested vocab_size {self.vocab_size} exceeds data limit, "
                              f"falling back to {max_vocab}")
                        self.vocab_size = max_vocab
                        _train_sp(self.vocab_size)
                    else:
                        raise

                self.sp_model = spm.SentencePieceProcessor()
                self.sp_model.load(f'{model_prefix}.model')
                self._temp_model_path = f'{model_prefix}.model'

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        for token, expected_id in self.special_tokens.items():
            actual_id = self.sp_model.piece_to_id(token)
            if actual_id != expected_id:
                print(f"Warning: {token} has id {actual_id}, expected {expected_id}")

        self._create_hf_wrapper()
        self._collect_token_frequencies(training_data)

        print(f"BPE tokenizer trained with final vocab size: {self.wrapped_tokenizer.vocab_size}")

        return self.wrapped_tokenizer

    def _create_hf_wrapper(self):
        class SPMWrapper:
            def __init__(self, sp_model, special_tokens):
                self.sp_model = sp_model
                self.special_tokens = special_tokens
                self.unk_token_id = special_tokens["[UNK]"]
                self.pad_token_id = special_tokens["[PAD]"]
                self.cls_token_id = special_tokens["[CLS]"]
                self.sep_token_id = special_tokens["[SEP]"]
                self.mask_token_id = special_tokens["[MASK]"]
                self.vocab_size = sp_model.get_piece_size()

            def encode(self, text, add_special_tokens=False):
                if isinstance(text, str):
                    ids = self.sp_model.encode(text)
                    if add_special_tokens:
                        ids = [self.cls_token_id] + ids + [self.sep_token_id]
                    return ids
                return []

            def tokenize(self, text):
                return self.sp_model.encode_as_pieces(text)

            def decode(self, ids, skip_special_tokens=False):
                if skip_special_tokens:
                    special_token_ids = set(self.special_tokens.values())
                    ids = [id for id in ids if id not in special_token_ids]
                return self.sp_model.decode(ids)

            def save_pretrained(self, save_directory):
                pass

        self.wrapped_tokenizer = SPMWrapper(self.sp_model, self.special_tokens)

    def _collect_token_frequencies(self, sequences):
        if not self.wrapped_tokenizer:
            return

        freq = Counter()

        sample_size = min(len(sequences), 10000)
        sampled = random.sample(sequences, sample_size)

        for seq in tqdm(sampled, desc="Collecting token frequencies (sample)"):
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

    def encode_sequences_genes(
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

        if not self.sp_model:
            raise ValueError("Tokenizer not initialized. Call setup_tokenizer first.")

        encoded_sequences = []
        gene_mapping = []

        unk_count = 0
        non_unk_count = 0

        unk_id = tokenizer.unk_token_id

        for index, entry in enumerate(sequences):
            gene = entry[0] if isinstance(entry, list) else entry

            if gene in self.special_tokens:
                token_id = self.sp_model.piece_to_id(gene)
                encoded_sequences.append(token_id)
                gene_mapping.append(genes_in_this_isolate[index])
                non_unk_count += 1
            else:
                if split == "train" and dropout_prob > 0:
                    ids = self.sp_model.encode(gene, enable_sampling=True, alpha=0.1, nbest_size=-1)
                else:
                    ids = self.sp_model.encode(gene)

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

    def save_tokenizer(self, save_path):
        save_dir = os.path.dirname(save_path) or '.'
        os.makedirs(save_dir, exist_ok=True)

        if hasattr(self, '_temp_model_path') and os.path.exists(self._temp_model_path):
            model_save_path = save_path.replace('.json', '.model')
            with open(self._temp_model_path, 'rb') as f_in:
                with open(model_save_path, 'wb') as f_out:
                    f_out.write(f_in.read())
        else:
            model_save_path = save_path.replace('.json', '.model')
            print("Warning: Cannot save model without original .model file")

        metadata = {
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'token_freq_sample': dict(list(self.token_freq.items())[:100]),  # Save sample
            'uncommon_tokens_count': len(self.uncommon_tokens),
            'min_count_uncommon': self.min_count_uncommon,
            'rare_quantile': self.rare_quantile,
            'model_path': model_save_path,
        }

        metadata_path = save_path.replace('.json', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved tokenizer to {os.path.dirname(save_path)}")

    def load_tokenizer(self, load_path):
        model_path = load_path.replace('.json', '.model')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)

        self._create_hf_wrapper()

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

