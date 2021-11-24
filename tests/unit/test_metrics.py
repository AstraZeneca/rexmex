import pytest
import unittest
import numpy as np

from sklearn.metrics import precision_score, recall_score

from rexmex.metrics import condition_positive, condition_negative
from rexmex.metrics import true_positive, true_negative, false_positive, false_negative
from rexmex.metrics import specificity, selectivity, true_negative_rate

from rexmex.metrics import sensitivity, hit_rate, true_positive_rate

from rexmex.metrics import positive_predictive_value, negative_predictive_value
from rexmex.metrics import miss_rate, false_negative_rate
from rexmex.metrics import fall_out, false_positive_rate
from rexmex.metrics import false_discovery_rate, false_omission_rate

from rexmex.metrics import positive_likelihood_ratio, negative_likelihood_ratio
from rexmex.metrics import prevalence_threshold, threat_score, critical_success_index

from rexmex.metrics import fowlkes_mallows_index, informedness, markedness, diagnostic_odds_ratio

from rexmex.metrics import root_mean_squared_error, symmetric_mean_absolute_percentage_error

class TestClassificationMetrics(unittest.TestCase):
    """
    Newly defined classification metric behaviour tests.
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
        assert specificity(self.y_true, self.y_scores) == 3/8
        assert selectivity(self.y_true, self.y_scores) == 3/8
        assert true_negative_rate(self.y_true, self.y_scores) == 3/8

    def test_sensitivity(self):
        assert sensitivity(self.y_true, self.y_scores) == 4/6
        assert hit_rate(self.y_true, self.y_scores) == 4/6
        assert true_positive_rate(self.y_true, self.y_scores) == 4/6

        assert sensitivity(self.y_true, self.y_scores) == recall_score(self.y_true, self.y_scores)
        assert hit_rate(self.y_true, self.y_scores) == recall_score(self.y_true, self.y_scores)
        assert true_positive_rate(self.y_true, self.y_scores) == recall_score(self.y_true, self.y_scores)

    def test_positivie_predictive_value(self):
        assert positive_predictive_value(self.y_true, self.y_scores) == precision_score(self.y_true, self.y_scores)
        assert positive_predictive_value(self.y_true, self.y_scores) == 4/9

    def test_negative_predictive_value(self):
        assert negative_predictive_value(self.y_true, self.y_scores) == 3/5

    def test_false_negative_rate(self):
        assert miss_rate(self.y_true, self.y_scores) == 1/3
        assert false_negative_rate(self.y_true, self.y_scores) == 1/3

    def test_false_positive_rate(self):
        assert fall_out(self.y_true, self.y_scores) == 5/8
        assert false_positive_rate(self.y_true, self.y_scores) == 5/8

    def test_discovery_omission(self):
        assert false_omission_rate(self.y_true, self.y_scores) == (1 - negative_predictive_value(self.y_true, self.y_scores))
        assert false_discovery_rate(self.y_true, self.y_scores) == (1 - positive_predictive_value(self.y_true, self.y_scores))

    def test_likelihood_ratios(self):
        assert positive_likelihood_ratio(self.y_true, self.y_scores) == 16/15
        assert negative_likelihood_ratio(self.y_true, self.y_scores) == 8/9

    def test_prevalence_threshold(self):
        assert prevalence_threshold(self.y_true, self.y_scores) == (5/8)**0.5/((4/6)**0.5 + (5/8)**0.5)
        assert threat_score(self.y_true, self.y_scores) == 4/11
        assert critical_success_index(self.y_true, self.y_scores) == 4/11

    def test_informedness_markedness(self):
        assert fowlkes_mallows_index(self.y_true, self.y_scores) == (8/27)**0.5
        assert round(informedness(self.y_true, self.y_scores), 5) == round(1/24, 5)
        assert round(markedness(self.y_true, self.y_scores), 5) == round(2/45, 5)
        assert diagnostic_odds_ratio(self.y_true, self.y_scores) == 6/5

class TestRatingMetrics(unittest.TestCase):
    """
    Newly defined rating metric behaviour tests.
    """
    def setUp(self):
        self.y_true = np.array([0, 2, 3])
        self.y_scores = np.array([4, 7, 8])
    
    def test_metrics(self):
        assert root_mean_squared_error(self.y_true, self.y_scores) == 22**0.5
        assert round(symmetric_mean_absolute_percentage_error(self.y_true, self.y_scores)) == 134