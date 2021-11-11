import unittest
import pytest
import pandas as pd

from rexmex.metricset import ClassificationMetricSet

class TestMetrics(unittest.TestCase):

    def test_classification_metric_set(self):
        metric_set = ClassificationMetricSet()
        assert "auc_score" in metric_set

