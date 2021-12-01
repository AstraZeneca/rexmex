import sys
import pytest
import unittest
from io import StringIO

from rexmex.metricset import ClassificationMetricSet
from rexmex.scorecard import ScoreCard


class TestMetricSet(unittest.TestCase):
    def test_representation(self):
        metric_set = ClassificationMetricSet()
        score_card = ScoreCard(metric_set)
        assert repr(score_card) == "ScoreCard()"

    def test_printing(self):
        metric_set = ClassificationMetricSet()
        metric_set.filter_metrics(["roc_auc", "pr_auc"])
        score_card = ScoreCard(metric_set)
        captured = StringIO()
        sys.stdout = captured
        score_card.print_metrics()
        sys.stdout = sys.__stdout__
        out = captured.getvalue().strip("\n")
        assert out == str({"roc_auc", "pr_auc"})
