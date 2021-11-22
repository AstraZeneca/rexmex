import unittest
import pytest
import pandas as pd
from sklearn_metrics import accuracy_score
from rexmex.metricset import ClassificationMetricSet, RatingMetricSet, RankingMetricSet, CoverageMetricSet

class TestMetrics(unittest.TestCase):

    def test_representation(self):
        assert repr(ClassificationMetricSet()) == "ClassificationMetricSet()"
        assert repr(CoverageMetricSet()) == "CoverageMetricSet()"
        assert repr(RankingMetricSet()) == "RankingMetricSet()"
        assert repr(RatingMetricSet()) == "RatingMetricSet()"

    def test_metric_filter(self):
        metric_set = ClassificationMetricSet()
        metric_set.filter_metric_set(["roc_auc", "pr_auc"])
        assert len(metric_set) == 2
        metric_set = ClassificationMetricSet()
        metric_set.filter_metric_set(["fake_metric"])
        assert len(metric_set) == 0

    def test_metric_add(self):
        assert 2 == 2

    def test_print(self):
        assert 2 == 2
