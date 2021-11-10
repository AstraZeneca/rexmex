import pytest
import unittest

from rexmex.scorecard import ScoreCard
from rexmex.dataset import DatasetReader
from rexmex.metrics import ClassificationMetricSet

class TestMetricAggregation(unittest.TestCase):

    def test_classification(self):
        reader = DatasetReader()
        scores = reader.read_dataset()
        metric_set = ClassificationMetricSet()
        metric_set.setup_basic_metrics()
        score_card = ScoreCard(metric_set)
        performance_metrics = score_card.calculate(scores, groupping=["source_group"])
        print(performance_metrics)
        assert performance_metrics.shape == (10, 6)