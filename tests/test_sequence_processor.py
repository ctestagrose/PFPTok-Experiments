import pytest
from curated_genes.utils.sequence_processor import SequenceProcessor


@pytest.fixture
def sp():
    return SequenceProcessor(kmer_size=6, stride=1)


class TestHandleNs:
    def test_nothing_strategy_is_identity(self, sp):
        assert sp.handle_ns("ACNGT", strategy="nothing") == "ACNGT"

    def test_trim_removes_ns(self, sp):
        result = sp.handle_ns("ACNGT", strategy="trim")
        assert "N" not in result
        assert result == "ACGT"

    def test_mask_replaces_with_x(self, sp):
        assert sp.handle_ns("ACNGT", strategy="mask") == "ACXGT"

    def test_substitute_replaces_ns(self, sp):
        result = sp.handle_ns("ANNG", strategy="substitute")
        assert "N" not in result
        assert len(result) == 4

    def test_filter_passes_low_n_content(self, sp):
        seq = "ACN" + "ACGT" * 50
        assert sp.handle_ns(seq, strategy="filter") != "XXXXXXXXXXXXXXXXXX"

    def test_filter_rejects_high_n_content(self, sp):
        assert sp.handle_ns("N" * 100, strategy="filter") == "XXXXXXXXXXXXXXXXXX"

    def test_invalid_strategy_raises(self, sp):
        with pytest.raises(ValueError):
            sp.handle_ns("ACGT", strategy="bogus")


class TestExtractAndPrepGenes:
    def test_basic_extraction_shape(self, sp):
        sequences = [["ACGTACGTACGT"]]
        labels = [0]
        unique_mers, prepped, prepped_labels = sp.extract_and_prep_genes(sequences, labels)
        assert len(prepped) == 1
        assert prepped_labels == [0]
        assert len(unique_mers) > 0

    def test_kmer_length_bounded(self, sp):
        sequences = [["ACGTACGTACGTACGT"]]
        labels = [0]
        unique_mers, _, _ = sp.extract_and_prep_genes(sequences, labels)
        for mer in unique_mers:
            assert len(mer) <= sp.kmer_size

    def test_multiple_genes_per_isolate(self, sp):
        sequences = [["ACGTACGT", "TGCATGCA", "GGGGCCCC"]]
        labels = [1]
        _, prepped, _ = sp.extract_and_prep_genes(sequences, labels)
        assert len(prepped[0]) == 3

    def test_multiple_isolates(self, sp):
        sequences = [["ACGTACGT"], ["TGCATGCA"]]
        labels = [0, 1]
        _, prepped, prepped_labels = sp.extract_and_prep_genes(sequences, labels)
        assert len(prepped) == 2
        assert prepped_labels == [0, 1]

    def test_stride_affects_kmer_count(self):
        sp_stride1 = SequenceProcessor(kmer_size=6, stride=1)
        sp_stride3 = SequenceProcessor(kmer_size=6, stride=3)
        seq = [["ACGTACGTACGTACGT"]]
        labels = [0]
        _, prepped1, _ = sp_stride1.extract_and_prep_genes(seq, labels)
        _, prepped3, _ = sp_stride3.extract_and_prep_genes(seq, labels)
        assert len(prepped3[0][0]) <= len(prepped1[0][0])

    def test_sequence_with_ns(self, sp):
        sequences = [["ACGNACGT"]]
        labels = [0]
        unique_mers, prepped, _ = sp.extract_and_prep_genes(sequences, labels)
        assert len(prepped) == 1


class TestGetFullSet:
    def test_returns_set(self, sp):
        seqs = [["ACGTACGT"], ["TGCATGCA"]]
        result = sp.get_full_set(seqs)
        assert isinstance(result, set)

    def test_kmers_bounded(self, sp):
        seqs = [["ACGTACGTACGTACGT"]]
        result = sp.get_full_set(seqs)
        assert all(len(m) <= sp.kmer_size for m in result)
