import pytest
from curated_genes.utils.metric_calculator import MetricsCalculator


@pytest.fixture
def mc():
    return MetricsCalculator()


class TestFindBestThreshold:
    def test_returns_threshold_and_score(self, mc):
        labels = [0, 1, 0, 1, 0, 1]
        probs = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]
        threshold, score = mc.find_best_threshold(labels, probs)
        assert 0.05 <= threshold <= 0.95
        assert 0.0 <= score <= 1.0

    def test_perfect_separation_gives_f1_1(self, mc):
        labels = [0, 0, 1, 1]
        probs = [0.1, 0.2, 0.8, 0.9]
        _, score = mc.find_best_threshold(labels, probs)
        assert score == pytest.approx(1.0)

    def test_threshold_above_chance(self, mc):
        labels = [0, 1, 0, 1, 0, 1, 0, 1]
        probs = [0.1, 0.9, 0.2, 0.8, 0.15, 0.85, 0.05, 0.95]
        threshold, score = mc.find_best_threshold(labels, probs)
        assert score > 0.5


class TestCalculateMetrics:
    def test_perfect_predictions(self, mc):
        labels = [0, 1, 0, 1]
        preds = [0, 1, 0, 1]
        f1, acc, *_ = mc.calculate_metrics(labels, preds)
        assert acc == pytest.approx(1.0)
        assert f1 == pytest.approx(1.0)

    def test_all_wrong(self, mc):
        labels = [0, 1, 0, 1]
        preds = [1, 0, 1, 0]
        _, acc, *_ = mc.calculate_metrics(labels, preds)
        assert acc == pytest.approx(0.0)

    def test_with_probabilities_returns_auc(self, mc):
        labels = [0, 1, 0, 1]
        preds = [0, 1, 0, 1]
        probs = [0.1, 0.9, 0.2, 0.8]
        f1, acc, hamming, jaccard, precision, recall, auc, confusion, report = \
            mc.calculate_metrics(labels, preds, final_probabilities=probs)
        assert auc is not None
        assert 0.0 <= auc <= 1.0

    def test_no_probabilities_auc_is_none(self, mc):
        labels = [0, 1, 0, 1]
        preds = [0, 1, 0, 1]
        *_, auc, _cm, _report = mc.calculate_metrics(labels, preds)
        assert auc is None

    def test_all_return_values_present(self, mc):
        labels = [0, 1, 1, 0, 1]
        preds = [0, 1, 0, 0, 1]
        result = mc.calculate_metrics(labels, preds)
        assert len(result) == 9


class TestCalculateMetricsThreshold:
    def test_auto_threshold_from_probs(self, mc):
        labels = [0, 1, 0, 1, 0, 1]
        probs = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]
        result = mc.calculate_metrics_threshold(labels, final_probabilities=probs)
        f1, acc = result[0], result[1]
        assert 0.0 <= f1 <= 1.0
        assert 0.0 <= acc <= 1.0

    def test_explicit_predictions_used(self, mc):
        labels = [0, 1, 0, 1]
        preds = [0, 1, 0, 1]
        probs = [0.1, 0.9, 0.2, 0.8]
        result = mc.calculate_metrics_threshold(labels, final_predictions=preds,
                                                final_probabilities=probs)
        assert result[1] == pytest.approx(1.0)
