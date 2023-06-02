import unittest

from rexmex.dataset import DatasetReader
from rexmex.metricset import ClassificationMetricSet, CoverageMetricSet, RatingMetricSet
from rexmex.scorecard import CoverageScoreCard, ScoreCard


class TestMetricAggregation(unittest.TestCase):
    def setUp(self):
        reader = DatasetReader()
        self.scores = reader.read_dataset()

    def test_classification(self):
        metric_set = ClassificationMetricSet()
        score_card = ScoreCard(metric_set)

        performance_metrics = score_card.generate_report(self.scores)
        assert performance_metrics.shape == (1, len(metric_set))

        performance_metrics = score_card.generate_report(self.scores, grouping=["source_group"])
        assert performance_metrics.shape == (5, len(metric_set))

        performance_metrics = score_card.generate_report(self.scores, grouping=["source_group", "target_group"])
        assert performance_metrics.shape == (20, len(metric_set))

    def test_regression(self):
        metric_set = RatingMetricSet()
        metric_set.normalize_metrics()
        score_card = ScoreCard(metric_set)

        performance_metrics = score_card.generate_report(self.scores)
        assert performance_metrics.shape == (1, len(metric_set))

        performance_metrics = score_card.generate_report(self.scores, grouping=["source_group"])
        assert performance_metrics.shape == (5, len(metric_set))

        performance_metrics = score_card.generate_report(self.scores, grouping=["source_group", "target_group"])
        assert performance_metrics.shape == (20, len(metric_set))

    def test_addition(self):
        metric_set = RatingMetricSet() + ClassificationMetricSet()
        score_card = ScoreCard(metric_set)

        performance_metrics = score_card.generate_report(self.scores)
        assert performance_metrics.shape == (1, len(metric_set))

        performance_metrics = score_card.generate_report(self.scores, grouping=["source_group"])
        assert performance_metrics.shape == (5, len(metric_set))

        performance_metrics = score_card.generate_report(self.scores, grouping=["source_group", "target_group"])
        assert performance_metrics.shape == (20, len(metric_set))

    def test_coverage(self):
        # change example dataset columns to fit requirements of coverage metrics
        cov_scores = self.scores.copy()
        cov_scores.rename({"source_id": "user", "target_id": "item"}, axis=1, inplace=True)

        # define positive predictions for y_score > 0.75
        recommendations = cov_scores.loc[cov_scores["y_score"] > 0.75, :]
        all_users = cov_scores["user"].unique()
        all_items = cov_scores["item"].unique()

        metric_set = CoverageMetricSet()
        score_card = CoverageScoreCard(metric_set, all_users, all_items)

        performance_metrics = score_card.generate_report(recommendations)
        assert performance_metrics.shape == (1, len(metric_set))

        performance_metrics = score_card.generate_report(recommendations, grouping=["source_group"])
        assert performance_metrics.shape == (5, len(metric_set))

        performance_metrics = score_card.generate_report(recommendations, grouping=["source_group", "target_group"])
        assert performance_metrics.shape == (20, len(metric_set))
