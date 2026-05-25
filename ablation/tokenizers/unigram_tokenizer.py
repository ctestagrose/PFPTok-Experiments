###CHUNKED BELOW###

import json
import os
import random
import tempfile
from collections import Counter
from typing import List, Dict, Tuple, Optional
import sentencepiece as spm
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

class TokenizerManagerUnigram:
    def __init__(self, vocab_size=10000, chunk_size=50000):
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
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

    def chunk_sequence(self, sequence: str, chunk_size: Optional[int] = None, overlap: int = 0) -> List[str]:
        if chunk_size is None:
            chunk_size = self.chunk_size

        if len(sequence) <= chunk_size:
            return [sequence]

        chunks = []
        step = chunk_size - overlap

        for i in range(0, len(sequence), step):
            chunk = sequence[i:i + chunk_size]
            if len(chunk) > 0:
                chunks.append(chunk)

        return chunks

    def setup_tokenizer(self, sequences, vocab_size=None, min_frequency=5,
                      max_sentence_length=100_000, max_sentencepiece_length=64,
                      model_type="bpe", max_training_sequences=10000,
                      use_chunking=True):
        if vocab_size:
            self.vocab_size = vocab_size

        print(f"Setting up SentencePiece tokenizer with vocab_size={self.vocab_size}, model_type={model_type}")
        print(f"use_chunking={use_chunking}, chunk_size={self.chunk_size} bp")
        print(f"Number of input sequences: {len(sequences)}")

        training_data = []
        long_sequences_found = 0
        total_bp = 0
        total_chunks = 0

        for idx, seq in enumerate(tqdm(sequences, desc="Preparing sequences for SentencePiece")):
            if isinstance(seq, str):
                seq_len = len(seq)
                total_bp += seq_len

                if seq not in self.special_tokens:
                    if len(seq) > self.chunk_size:
                        long_sequences_found += 1
                        if use_chunking:
                            chunks = self.chunk_sequence(seq, chunk_size=self.chunk_size)
                            training_data.extend(chunks)
                            total_chunks += len(chunks)
                        else:
                            print(f"  WARNING: Sequence {idx} is {seq_len:,} bp (>{self.chunk_size}), adding as-is")
                            training_data.append(seq)
                    else:
                        training_data.append(seq)
            elif isinstance(seq, list):
                print(f"  Sequence {idx}: List of {len(seq)} genes (legacy format)")
                for gene in seq:
                    if gene not in self.special_tokens:
                        training_data.append(gene)
            else:
                raise ValueError(f"Sequence {idx} has unexpected type: {type(seq)}")

        print(f"\n📊 Data preparation summary:")
        print(f"  Input sequences: {len(sequences)}")
        print(f"  Total base pairs: {total_bp:,}")
        print(f"  Long sequences (>{self.chunk_size} bp): {long_sequences_found}")
        print(f"  Total chunks created: {total_chunks}")
        print(f"  Output training items: {len(training_data)}")

        if use_chunking and long_sequences_found > 0:
            expected_ratio = total_chunks / long_sequences_found
            print(f"  Avg chunks per long sequence: {expected_ratio:.1f}")
            if expected_ratio < 2:
                print(f"WARNING: Expected more chunks! Check chunking logic.")

        if len(training_data) == 0:
            raise ValueError("No training data! All sequences were filtered out.")

        if len(training_data) > max_training_sequences:
            print(f"Sampling {max_training_sequences} sequences from {len(training_data)} for faster training...")
            training_data = random.sample(training_data, max_training_sequences)

        print(f"Training on {len(training_data)} sequences/chunks...")

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_file = f.name
            for item in training_data:
                f.write(item + '\n')

        try:
            user_defined_symbols = [
                token for token in self.special_tokens.keys()
                if token not in ["[UNK]", "[PAD]"]
            ]

            with tempfile.TemporaryDirectory() as tmpdir:
                model_prefix = os.path.join(tmpdir, 'sp_model')

                num_training_samples = len(training_data)
                if num_training_samples <= 100:
                    input_sentence_size_param = 0  # Use all
                    print(f"Using all {num_training_samples} sequences (input_sentence_size=0)")
                else:
                    input_sentence_size_param = min(num_training_samples, max_training_sequences)
                    print(f"Using input_sentence_size={input_sentence_size_param}")

                # Train SentencePiece
                print(f"Training SentencePiece (this may take 2-4 minutes)...")
                spm.SentencePieceTrainer.train(
                    input=temp_file,
                    model_prefix=model_prefix,
                    vocab_size=self.vocab_size,
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
                    max_sentencepiece_length=max_sentencepiece_length,
                    max_sentence_length=max_sentence_length,
                    input_sentence_size=input_sentence_size_param,
                    shuffle_input_sentence=True,
                    split_by_whitespace=False,
                    split_by_unicode_script=False,
                    minloglevel=1,
                )

                # Load the trained model
                self.sp_model = spm.SentencePieceProcessor()
                self.sp_model.load(f'{model_prefix}.model')

                # Save model bytes
                with open(f'{model_prefix}.model', 'rb') as f:
                    self._temp_model_bytes = f.read()

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        for token, expected_id in self.special_tokens.items():
            actual_id = self.sp_model.piece_to_id(token)
            if actual_id != expected_id:
                print(f"Warning: {token} has id {actual_id}, expected {expected_id}")

        self._create_hf_wrapper()

        sample_for_stats = training_data[:10000] if len(training_data) > 10000 else training_data
        self._collect_token_frequencies(sample_for_stats)

        print(f"✓ Tokenizer trained with vocab size: {self.wrapped_tokenizer.vocab_size}")

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
        sampled = random.sample(sequences, sample_size) if len(sequences) > sample_size else sequences

        for seq in tqdm(sampled, desc="Collecting token frequencies"):
            tokens = self.wrapped_tokenizer.tokenize(seq)
            freq.update(tokens)

        self.token_freq = dict(freq)

        if freq:
            counts = sorted(freq.values())
            cutoff_idx = max(0, int(len(counts) * self.rare_quantile) - 1)
            cutoff = max(self.min_count_uncommon, counts[cutoff_idx] if counts else 0)
            self.uncommon_tokens = {tok for tok, c in freq.items() if c <= cutoff}

            print(f"Token stats: {len(self.token_freq)} unique tokens")
            print(f"Uncommon tokens: {len(self.uncommon_tokens)} (threshold: {cutoff})")

    def encode_sequences(
        self,
        sequences: List,
        tokenizer,
        genes_in_this_isolate: List,
        split: str = "train",
        dropout_prob: float = 0.0,
        seed: Optional[int] = None,
    ) -> Tuple[List[int], List, Tuple[int, int]]:
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

    def encode_whole_genome(
        self,
        genome_sequence: str,
        tokenizer=None,
        split: str = "test",
        dropout_prob: float = 0.0,
        use_chunking: bool = True,
        seed: Optional[int] = None,
    ) -> Tuple[List[int], int, int]:
        if seed is not None:
            random.seed(seed)

        if tokenizer is None:
            tokenizer = self.wrapped_tokenizer

        if not self.sp_model:
            raise ValueError("Tokenizer not initialized. Call setup_tokenizer first.")

        unk_id = tokenizer.unk_token_id

        if genome_sequence in self.special_tokens:
            return [self.special_tokens[genome_sequence]], 0, 1

        if use_chunking and len(genome_sequence) > self.chunk_size:
            chunks = self.chunk_sequence(genome_sequence, chunk_size=self.chunk_size)
            all_token_ids = []

            for chunk in chunks:
                if split == "train" and dropout_prob > 0:
                    token_ids = self.sp_model.encode(chunk, enable_sampling=True, alpha=0.1, nbest_size=-1)
                else:
                    token_ids = self.sp_model.encode(chunk)

                all_token_ids.extend(token_ids)

            token_ids = all_token_ids
        else:
            if split == "train" and dropout_prob > 0:
                token_ids = self.sp_model.encode(genome_sequence, enable_sampling=True, alpha=0.1, nbest_size=-1)
            else:
                token_ids = self.sp_model.encode(genome_sequence)

        if split == "train" and dropout_prob > 0:
            processed_ids = []
            for token_id in token_ids:
                if random.random() < dropout_prob:
                    continue
                processed_ids.append(token_id)

            if not processed_ids:
                processed_ids = [unk_id]

            token_ids = processed_ids

        unk_count = sum(1 for t in token_ids if t == unk_id)
        total_tokens = len(token_ids)

        return token_ids, unk_count, total_tokens

    def encode_sequence_chunked(
        self,
        sequence: str,
        tokenizer=None,
        split: str = "train",
        dropout_prob: float = 0.0,
        seed: Optional[int] = None,
        use_chunking: bool = True,
    ) -> Tuple[List[int], None, None]:

        token_ids, unk_count, total_tokens = self.encode_whole_genome(
            genome_sequence=sequence,
            tokenizer=tokenizer,
            split=split,
            dropout_prob=dropout_prob,
            use_chunking=use_chunking,
            seed=seed
        )

        return (token_ids, None, None)

    def save_tokenizer(self, tokenizer, save_path):
        save_dir = os.path.dirname(save_path) or '.'
        os.makedirs(save_dir, exist_ok=True)

        model_save_path = save_path.replace('.json', '.model')
        if hasattr(self, '_temp_model_bytes'):
            with open(model_save_path, 'wb') as f:
                f.write(self._temp_model_bytes)
            print(f"Saved SentencePiece model to {model_save_path}")

        metadata = {
            'vocab_size': self.vocab_size,
            'chunk_size': self.chunk_size,
            'special_tokens': self.special_tokens,
            'token_freq_sample': dict(list(self.token_freq.items())[:100]),
            'uncommon_tokens_count': len(self.uncommon_tokens),
            'min_count_uncommon': self.min_count_uncommon,
            'rare_quantile': self.rare_quantile,
            'model_path': model_save_path,
        }

        metadata_path = save_path.replace('.json', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved tokenizer metadata to {metadata_path}")

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
                self.chunk_size = metadata.get('chunk_size', self.chunk_size)
                self.special_tokens = metadata.get('special_tokens', self.special_tokens)
                self.min_count_uncommon = metadata.get('min_count_uncommon', 2)
                self.rare_quantile = metadata.get('rare_quantile', 0.20)
            print(f"Loaded tokenizer with vocab size: {self.wrapped_tokenizer.vocab_size}")
            print(f"Chunk size: {self.chunk_size} bp")

        return self.wrapped_tokenizer