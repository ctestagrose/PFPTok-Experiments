import json
from pathlib import Path

import pytest

SAMPLE_DIR = Path(__file__).parent.parent / "sample_data"
CURATED_TRAIN = SAMPLE_DIR / "curated_genes_sample_data" / "train"
CURATED_TEST = SAMPLE_DIR / "curated_genes_sample_data" / "test"
TARGETS_FILE = SAMPLE_DIR / "cryptic_targets_all.json"


# Helpers
def _read_fasta(path: Path) -> list[str]:
    sequences, current = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                current = []
            else:
                current.append(line.upper())
    if current:
        sequences.append("".join(current))
    return sequences


@pytest.fixture(scope="module")
def train_sequences():
    seqs = []
    for fasta in sorted(CURATED_TRAIN.glob("*.fasta"))[:5]:
        seqs.append(_read_fasta(fasta))
    return seqs


# Smoke tests: sample data layout
def test_sample_data_directories_exist():
    assert CURATED_TRAIN.exists()
    assert CURATED_TEST.exists()


def test_sample_fasta_files_present():
    train_files = list(CURATED_TRAIN.glob("*.fasta"))
    test_files = list(CURATED_TEST.glob("*.fasta"))
    assert len(train_files) >= 10, "Expected at least 10 training isolates"
    assert len(test_files) >= 10, "Expected at least 10 test isolates"


def test_targets_file_loadable():
    assert TARGETS_FILE.exists()
    with open(TARGETS_FILE) as fh:
        targets = json.load(fh)
    assert isinstance(targets, dict)
    assert len(targets) > 0


def test_fasta_sequences_are_dna():
    valid_chars = set("ACGTN")
    for fasta in sorted(CURATED_TRAIN.glob("*.fasta"))[:3]:
        for seq in _read_fasta(fasta):
            assert set(seq).issubset(valid_chars), f"Unexpected chars in {fasta.name}"


# PFP tokenizer end-to-end
def test_pfp_tokenizer_trains_on_sample_data(train_sequences):
    from curated_genes.test_tokenizers.pfp_tokenizer import TokenizerManager

    tm = TokenizerManager()
    tok = tm.setup_tokenizer(train_sequences, w=6, p=31)
    assert tok is not None
    assert tok.get_vocab_size() > 8  # at least the 8 special tokens


def test_pfp_vocab_contains_special_tokens(train_sequences):
    from curated_genes.test_tokenizers.pfp_tokenizer import TokenizerManager

    tm = TokenizerManager()
    tok = tm.setup_tokenizer(train_sequences, w=6, p=31)
    for token in ["[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"]:
        assert tok.token_to_id(token) is not None


def test_pfp_encode_on_sample_sequences(train_sequences):
    from curated_genes.test_tokenizers.pfp_tokenizer import TokenizerManager

    tm = TokenizerManager()
    tok = tm.setup_tokenizer(train_sequences, w=6, p=31)

    # Encode the first isolate's sequences
    flat = [[s] for s in train_sequences[0]]
    genes = [f"gene_{i}" for i in range(len(flat))]
    encoded, _, (unk_c, non_unk_c) = tm.encode_sequences_genes(flat, tok, genes)
    assert len(encoded) > 0
    total = unk_c + non_unk_c
    assert total > 0
    unk_pct = unk_c / total
    assert unk_pct < 1.0, "All tokens are UNK — tokenizer likely broken"


# Sequence processor end-to-end
def test_sequence_processor_on_sample_data(train_sequences):
    from curated_genes.utils.sequence_processor import SequenceProcessor

    sp = SequenceProcessor(kmer_size=6, stride=1)
    labels = [0] * len(train_sequences)
    unique_mers, prepped, prepped_labels = sp.extract_and_prep_genes(train_sequences, labels)

    assert len(unique_mers) > 0
    assert all(len(m) <= 6 for m in unique_mers)
    assert len(prepped) == len(train_sequences)
    assert prepped_labels == labels


# Ablation tokenizer end-to-end
def test_ablation_pfp_tokenizer_on_sample_data(train_sequences):
    from ablation.test_tokenizers.pfp_tokenizer import TokenizerManager as AblationTM

    tm = AblationTM()
    tok = tm.setup_tokenizer(train_sequences, w=6, p=31)
    assert tok is not None
    assert tok.get_vocab_size() > 8
