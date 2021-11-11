import unittest
import pytest
import pandas as pd

from rexmex.metrics import ClassificationMetricSet

class TestMetrics(unittest.TestCase):

    def test_classification_metric_set(self):
        scores = pd.DataFrame([[1, 0.1], [0,0.5], [1,0.99]], columns = ["y_true", "y_score"])
        metric_set = ClassificationMetricSet()
        performance_metrics = metric_set.get_performance_metrics(scores.y_true, scores.y_score)
        assert performance_metrics.shape == (1, 6)

