import pytest
import unittest
import numpy as np

from sklearn.metrics import precision_score, recall_score
from rexmex.metrics import condition_positive, condition_negative

class TestClassificationMetrics(unittest.TestCase):
    """
    Newly defined classification metric tests.
    """
    def setUp(self):
        self.y_true = np.array([0, 1, 1, 1, 1, 0, 0, 0, 0])
        self.y_scores = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0])

    def test_conditions(self):
        assert condition_positive(self.y_true) == 4
        assert condition_negative(self.y_true) == 5

    def test_false_and_true(self):
        assert binarize(false_positive(self.y_true, self.y_scores)) == 2
        assert binarize(false_negative(self.y_true, self.y_scores)) == 2 
        assert binarize(true_positive(self.y_true, self.y_scores)) == 2 
        assert binarize(true_negative(self.y_true, self.y_scores)) == 3

class TestRatingMetrics(unittest.TestCase):

    def setUp(self):
        pass
    
    def test_metrics(self)
        pass