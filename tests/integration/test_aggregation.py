import pytest
import unittest

from rexmex.scorecard import ScoreCard
from rexmex.dataset import DatasetReader
from rexmex.metricset import ClassificationMetricSet, RatingMetricSet

class TestMetricAggregation(unittest.TestCase):

    def test_classification(self):
        reader = DatasetReader()
        scores = reader.read_dataset()
        metric_set = ClassificationMetricSet()
        score_card = ScoreCard(metric_set)

        performance_metrics = score_card.generate_report(scores, groupping=["source_group"])
        assert performance_metrics.shape == (5, 11)

        performance_metrics = score_card.generate_report(scores, groupping=["source_group", "target_group"])
        assert performance_metrics.shape == (20, 11)

    def test_regression(self):
        reader = DatasetReader()
        scores = reader.read_dataset()
        metric_set = RatingMetricSet()
        metric_set.normalize_metrics()
        score_card = ScoreCard(metric_set)

        performance_metrics = score_card.generate_report(scores, groupping=["source_group"])
        assert performance_metrics.shape == (5, 7)

        performance_metrics = score_card.generate_report(scores, groupping=["source_group", "target_group"])
        assert performance_metrics.shape == (20, 7)

    def test_addition(self):
        reader = DatasetReader()
        scores = reader.read_dataset()
        metric_set = RatingMetricSet() + ClassificationMetricSet()
        score_card = ScoreCard(metric_set)

        performance_metrics = score_card.generate_report(scores, groupping=["source_group"])
        assert performance_metrics.shape == (5, 18)

        performance_metrics = score_card.generate_report(scores, groupping=["source_group", "target_group"])
        assert performance_metrics.shape == (20, 18)
