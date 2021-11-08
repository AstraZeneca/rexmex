import pytest
import unittest

from rexmex.dataset import DatasetReader
from rexmex.metrics import ClassificationMetricSet

class TestScoreCard(unittest.TestCase):

    def test_classification(self):

        reader = DatasetReader()
        dataset = reader.read_dataset()
        metric_set = ClassificationMetricSet()
        score_card = ScoreCard(metric_set)
        performance_metrics = score_card.calculate(scores, groupping=["source_group", "target_group"], filter=["roc_auc"])
        print(performance_metrics)