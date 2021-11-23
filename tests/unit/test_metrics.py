import pytest
import unittest
import numpy as np

from sklearn.metrics import precision_score, recall_score

from rexmex.metrics import condition_positive, condition_negative
from rexmex.metrics import true_positive, true_negative, false_positive, false_negative
from rexmex.metrics import specificity, selectivity, true_negative_rate

from rexmex.metrics import sensitivity, hit_rate, true_positive_rate

class TestClassificationMetrics(unittest.TestCase):
    """
    Newly defined classification metric tests.
    """
    def setUp(self):
        self.y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
        self.y_scores = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0])

    def test_conditions(self):
        assert condition_positive(self.y_true) == 6
        assert condition_negative(self.y_true) == 8

    def test_false_and_true(self):
        assert false_positive(self.y_true, self.y_scores) == 5
        assert false_negative(self.y_true, self.y_scores) == 2
        assert true_positive(self.y_true, self.y_scores) == 4
        assert true_negative(self.y_true, self.y_scores) == 3

    def test_specificity(self):
        assert specificity(self.y_true, self.y_scores) == 0.375
        assert selectivity(self.y_true, self.y_scores) == 0.375 
        assert true_negative_rate(self.y_true, self.y_scores) == 0.375

    def test_sensitivity(self):
        assert sensitivity(self.y_true, self.y_scores) == 4/6
        assert hit_rate(self.y_true, self.y_scores) == 4/6
        assert true_positive_rate(self.y_true, self.y_scores) == 4/6

        assert sensitivity(self.y_true, self.y_scores) == recall_score(self.y_true, self.y_scores)
        assert hit_rate(self.y_true, self.y_scores) == recall_score(self.y_true, self.y_scores)
        assert true_positive_rate(self.y_true, self.y_scores) == recall_score(self.y_true, self.y_scores)


class TestRatingMetrics(unittest.TestCase):
    """
    Newly defined rating metric tests.
    """
    def setUp(self):
        pass
    
    def test_metrics(self):
        pass