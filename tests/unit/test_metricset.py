import unittest
import pytest
import pandas as pd

from rexmex.metricset import ClassificationMetricSet, RatingMetricSet, RankingMetricSet, CoverageMetricSet

class TestMetrics(unittest.TestCase):

    def test_representation(self):
        assert repr(ClassificationMetricSet()) == "ClassificationMetricSet()"
        assert repr(CoverageMetricSet()) == "CoverageMetricSet()"
        assert repr(RankingMetricSet()) == "RankingMetricSet()"
        assert repr(RatingMetricSet()) == "RatingMetricSet()"
