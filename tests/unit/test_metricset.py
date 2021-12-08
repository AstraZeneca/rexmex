import sys
import unittest
from io import StringIO

from sklearn.metrics import accuracy_score, balanced_accuracy_score

from rexmex.metricset import (
    ClassificationMetricSet,
    CoverageMetricSet,
    MetricSet,
    RankingMetricSet,
    RatingMetricSet,
)


class TestMetricSet(unittest.TestCase):
    def test_representation(self):
        assert repr(ClassificationMetricSet()) == "ClassificationMetricSet()"
        assert repr(CoverageMetricSet()) == "CoverageMetricSet()"
        assert repr(RankingMetricSet()) == "RankingMetricSet()"
        assert repr(RatingMetricSet()) == "RatingMetricSet()"

    def test_metric_filter(self):
        metric_set = ClassificationMetricSet()
        metric_set.filter_metrics(["roc_auc", "pr_auc"])
        assert len(metric_set) == 2
        metric_set = ClassificationMetricSet()
        metric_set.filter_metrics(["fake_metric"])
        assert len(metric_set) == 0

    def test_metric_add(self):
        metric_set = MetricSet()
        metric_set.add_metrics([("accuracy", accuracy_score)])
        assert len(metric_set) == 1
        metric_set.add_metrics([("accuracy", accuracy_score)])
        assert len(metric_set) == 1
        metric_set.add_metrics([("balanced_accuracy", balanced_accuracy_score)])
        assert len(metric_set) == 2

    def test_print(self):
        metric_set = MetricSet()
        metric_set.add_metrics([("accuracy", accuracy_score)])
        metric_set.add_metrics([("balanced_accuracy", balanced_accuracy_score)])
        captured = StringIO()
        sys.stdout = captured
        metric_set.print_metrics()
        sys.stdout = sys.__stdout__
        out = captured.getvalue().strip("\n")
        assert out == str({"accuracy", "balanced_accuracy"})
