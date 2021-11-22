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
        assert performance_metrics.shape == (10, 8)

    def test_regression(self):
        reader = DatasetReader()
        scores = reader.read_dataset()
        metric_set = RatingMetricSet()
        score_card = ScoreCard(metric_set)
        performance_metrics = score_card.generate_report(scores, groupping=["source_group"])
        assert performance_metrics.shape == (10, 7)