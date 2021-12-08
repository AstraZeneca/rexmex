import sys
import unittest
from io import StringIO

import pytest

from rexmex.metricset import ClassificationMetricSet
from rexmex.scorecard import ScoreCard


class TestMetricSet(unittest.TestCase):
    def setUp(self):
        self.metric_set = ClassificationMetricSet()
        self.score_card = ScoreCard(self.metric_set)

    def test_representation(self):
        assert repr(self.score_card) == "ScoreCard()"

    def test_printing(self):
        self.score_card.metric_set.filter_metrics(["roc_auc", "pr_auc"])
        captured = StringIO()
        sys.stdout = captured
        self.score_card.print_metrics()
        sys.stdout = sys.__stdout__
        out = captured.getvalue().strip("\n")
        assert out == str({"roc_auc", "pr_auc"})
