import pytest
from curated_genes.test_tokenizers.pfp_tokenizer import (
    karp_rabin_hash,
    prefix_free_parse,
    TokenizerManager,
)


class TestKarpRabinHash:
    def test_deterministic(self):
        assert karp_rabin_hash("ACGT") == karp_rabin_hash("ACGT")

    def test_different_sequences_differ(self):
        assert karp_rabin_hash("ACGT") != karp_rabin_hash("TGCA")

    def test_empty_string(self):
        assert karp_rabin_hash("") == 0

    def test_single_char(self):
        h = karp_rabin_hash("A")
        assert isinstance(h, int)
        assert h >= 0


class TestPrefixFreeParse:
    def test_returns_list(self):
        result = prefix_free_parse("ACGTACGTACGT", w=4, p=7)
        assert isinstance(result, list)

    def test_reconstructs_sequence(self):
        seq = "ACGTACGTACGTACGT"
        phrases = prefix_free_parse(seq, w=4, p=7)
        assert "".join(phrases) == seq

    def test_repeated_base_reconstructs(self):
        seq = "A" * 50
        phrases = prefix_free_parse(seq, w=5, p=7)
        assert "".join(phrases) == seq

    def test_n_bases_preserved(self):
        seq = "ACGTNNNACGT"
        phrases = prefix_free_parse(seq, w=4, p=7)
        assert "".join(phrases) == seq

    def test_sequence_shorter_than_window(self):
        phrases = prefix_free_parse("ACG", w=10, p=7)
        assert isinstance(phrases, list)

    def test_all_phrases_nonempty(self):
        seq = "ACGTACGTACGTACGTACGT"
        phrases = prefix_free_parse(seq, w=6, p=31)
        assert all(len(p) > 0 for p in phrases)

    @pytest.mark.parametrize("w,p", [(3, 7), (6, 31), (10, 127)])
    def test_various_params_reconstruct(self, w, p):
        seq = "ACGTACGT" * 20
        phrases = prefix_free_parse(seq, w=w, p=p)
        assert "".join(phrases) == seq


class TestTokenizerManager:
    @pytest.fixture
    def simple_seqs(self):
        return [
            ["ACGTACGTACGT", "TGCATGCATGCA"],
            ["AAAACCCCGGGG", "TTTTAAAACCCC"],
        ]

    def test_setup_returns_tokenizer(self, simple_seqs):
        tm = TokenizerManager()
        tok = tm.setup_tokenizer(simple_seqs, w=4, p=7)
        assert tok is not None

    def test_special_tokens_in_vocab(self, simple_seqs):
        tm = TokenizerManager()
        tok = tm.setup_tokenizer(simple_seqs, w=4, p=7)
        for token in ["[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"]:
            assert tok.token_to_id(token) is not None, f"{token} missing from vocab"

    def test_vocab_size_positive(self, simple_seqs):
        tm = TokenizerManager()
        tok = tm.setup_tokenizer(simple_seqs, w=4, p=7)
        assert tok.get_vocab_size() > len(tm.special_tokens)

    def test_phrase_freq_populated(self, simple_seqs):
        tm = TokenizerManager()
        tm.setup_tokenizer(simple_seqs, w=4, p=7)
        assert len(tm.phrase_freq) > 0

    def test_encode_sequences_genes_returns_ids(self, simple_seqs):
        tm = TokenizerManager()
        tok = tm.setup_tokenizer(simple_seqs, w=4, p=7)
        encoded, gene_mapping, (unk_c, non_unk_c) = tm.encode_sequences_genes(
            [["ACGTACGT"]], tok, ["rpoB"]
        )
        assert len(encoded) > 0
        assert unk_c + non_unk_c > 0

    def test_special_token_sequences_handled(self, simple_seqs):
        tm = TokenizerManager()
        tok = tm.setup_tokenizer(simple_seqs, w=4, p=7)
        encoded, _, _ = tm.encode_sequences_genes(
            [["[CLS]"]], tok, ["special"]
        )
        assert len(encoded) == 1

    def test_longer_sequence(self):
        import random
        random.seed(42)
        bases = "ACGT"
        long_seq = "".join(random.choice(bases) for _ in range(500))
        seqs = [[long_seq]]
        tm = TokenizerManager()
        tok = tm.setup_tokenizer(seqs, w=10, p=31)
        encoded, _, _ = tm.encode_sequences_genes(seqs, tok, ["gene1"])
        assert len(encoded) > 0
