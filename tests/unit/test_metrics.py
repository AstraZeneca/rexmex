import pytest
import unittest
import numpy as np

from rexmex.metrics.classification import precision_score, recall_score

from rexmex.metrics.classification import condition_positive, condition_negative
from rexmex.metrics.classification import (
    true_positive,
    true_negative,
    false_positive,
    false_negative,
)
from rexmex.metrics.classification import specificity, selectivity, true_negative_rate

from rexmex.metrics.classification import sensitivity, hit_rate, true_positive_rate

from rexmex.metrics.classification import (
    positive_predictive_value,
    negative_predictive_value,
)
from rexmex.metrics.classification import miss_rate, false_negative_rate
from rexmex.metrics.classification import fall_out, false_positive_rate
from rexmex.metrics.classification import false_discovery_rate, false_omission_rate

from rexmex.metrics.classification import (
    positive_likelihood_ratio,
    negative_likelihood_ratio,
)
from rexmex.metrics.classification import (
    prevalence_threshold,
    threat_score,
    critical_success_index,
)

from rexmex.metrics.classification import (
    fowlkes_mallows_index,
    informedness,
    markedness,
    diagnostic_odds_ratio,
)
from rexmex.metrics.ranking import (
    average_percision_at_k,
    hits_at_k,
    intra_list_similarity,
    kendall_tau,
    mean_average_percision_at_k,
    mean_reciprocal_rank,
    personalization,
    reciprocal_rank,
    spearmanns_rho,
)

from rexmex.metrics.rating import (
    root_mean_squared_error,
    symmetric_mean_absolute_percentage_error,
)


class TestClassificationMetrics(unittest.TestCase):
    """
    Newly defined classification metric behaviour tests.
    """

    def setUp(self):
        self.y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
        self.y_score = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0])

    def test_conditions(self):
        assert condition_positive(self.y_true) == 6
        assert condition_negative(self.y_true) == 8

    def test_false_and_true(self):
        assert false_positive(self.y_true, self.y_score) == 5
        assert false_negative(self.y_true, self.y_score) == 2
        assert true_positive(self.y_true, self.y_score) == 4
        assert true_negative(self.y_true, self.y_score) == 3

    def test_specificity(self):
        assert specificity(self.y_true, self.y_score) == 3 / 8
        assert selectivity(self.y_true, self.y_score) == 3 / 8
        assert true_negative_rate(self.y_true, self.y_score) == 3 / 8

    def test_sensitivity(self):
        assert sensitivity(self.y_true, self.y_score) == 4 / 6
        assert hit_rate(self.y_true, self.y_score) == 4 / 6
        assert true_positive_rate(self.y_true, self.y_score) == 4 / 6

        assert sensitivity(self.y_true, self.y_score) == recall_score(self.y_true, self.y_score)
        assert hit_rate(self.y_true, self.y_score) == recall_score(self.y_true, self.y_score)
        assert true_positive_rate(self.y_true, self.y_score) == recall_score(self.y_true, self.y_score)

    def test_positivie_predictive_value(self):
        assert positive_predictive_value(self.y_true, self.y_score) == precision_score(self.y_true, self.y_score)
        assert positive_predictive_value(self.y_true, self.y_score) == 4 / 9

    def test_negative_predictive_value(self):
        assert negative_predictive_value(self.y_true, self.y_score) == 3 / 5

    def test_false_negative_rate(self):
        assert miss_rate(self.y_true, self.y_score) == 1 / 3
        assert false_negative_rate(self.y_true, self.y_score) == 1 / 3

    def test_false_positive_rate(self):
        assert fall_out(self.y_true, self.y_score) == 5 / 8
        assert false_positive_rate(self.y_true, self.y_score) == 5 / 8

    def test_discovery_omission(self):
        assert false_omission_rate(self.y_true, self.y_score) == (
            1 - negative_predictive_value(self.y_true, self.y_score)
        )
        assert false_discovery_rate(self.y_true, self.y_score) == (
            1 - positive_predictive_value(self.y_true, self.y_score)
        )

    def test_likelihood_ratios(self):
        assert positive_likelihood_ratio(self.y_true, self.y_score) == 16 / 15
        assert negative_likelihood_ratio(self.y_true, self.y_score) == 8 / 9

    def test_prevalence_threshold(self):
        assert prevalence_threshold(self.y_true, self.y_score) == (5 / 8) ** 0.5 / ((4 / 6) ** 0.5 + (5 / 8) ** 0.5)
        assert threat_score(self.y_true, self.y_score) == 4 / 11
        assert critical_success_index(self.y_true, self.y_score) == 4 / 11

    def test_informedness_markedness(self):
        assert fowlkes_mallows_index(self.y_true, self.y_score) == (8 / 27) ** 0.5
        assert round(informedness(self.y_true, self.y_score), 5) == round(1 / 24, 5)
        assert round(markedness(self.y_true, self.y_score), 5) == round(2 / 45, 5)
        assert diagnostic_odds_ratio(self.y_true, self.y_score) == 6 / 5


class TestRatingMetrics(unittest.TestCase):
    """
    Newly defined rating metric behaviour tests.
    """

    def setUp(self):
        self.y_true = np.array([0, 2, 3])
        self.y_score = np.array([4, 7, 8])

    def test_metrics(self):
        assert root_mean_squared_error(self.y_true, self.y_score) == 22 ** 0.5
        assert round(symmetric_mean_absolute_percentage_error(self.y_true, self.y_score)) == 134


class TestRankingMetrics(unittest.TestCase):
    """
    Newly defined ranking metric behaviour tests.
    """

    def setUp(self):
        self.relevant_items = np.array([1, 2, 3, 4, 5, 6])
        self.ranking = np.array([1, 7, 3, 4, 5, 2, 10, 8, 9, 6])

    def test_reciprocal_rank(self):
        assert reciprocal_rank(1, self.ranking) == 1
        assert reciprocal_rank(2, self.ranking) == 1 / 6
        assert reciprocal_rank(3, self.ranking) == 1 / 3

    def test_average_percision_at_k(self):
        ap_10 = average_percision_at_k(self.relevant_items, self.ranking, k=10)
        self.assertAlmostEqual(ap_10, 0.775, 2)

        ap_4 = average_percision_at_k(self.relevant_items, self.ranking, k=4)
        self.assertAlmostEqual(ap_4, 0.805, 2)

    def test_mean_average_percision_at_k(self):
        relevant_items_1 = np.array([1, 2, 3, 4, 5])
        relevant_items_2 = np.array([1, 2, 3])

        ranking_1 = np.array([1, 10, 3, 9, 6, 5, 7, 8, 4, 2])
        ranking_2 = np.array([7, 2, 5, 4, 3, 6, 1, 8, 9, 10])

        map_10 = mean_average_percision_at_k(
            relevant_items=[relevant_items_1, relevant_items_2],
            rankings=[ranking_1, ranking_2],
            k=10,
        )

        self.assertAlmostEqual(map_10, 0.53, 2)

    def test_mean_reciprocal_rank(self):
        self.assertAlmostEqual(mean_reciprocal_rank(self.relevant_items, self.ranking), 0.3416, 3)

    def test_hits_at_k(self):
        hits_at_1 = hits_at_k(self.relevant_items, self.ranking, k=1)
        hits_at_5 = hits_at_k(self.relevant_items, self.ranking, k=5)
        hits_at_10 = hits_at_k(self.relevant_items, self.ranking, k=10)

        self.assertAlmostEqual(hits_at_1, 1.0, 2)
        self.assertAlmostEqual(hits_at_5, 0.8, 2)
        self.assertAlmostEqual(hits_at_10, 0.6, 2)

    def test_spearmanns_rho(self):
        corr, p_value = spearmanns_rho(range(1, 5), range(1, 5))

        assert corr == 1.0
        assert p_value == 0.0

        corr, p_value = spearmanns_rho(range(1, 5), range(4, 0, -1))
        assert corr == -1.0
        assert p_value == 0.0

    def test_kendall_tau(self):
        corr, p_value = kendall_tau(range(1, 5), range(1, 5))

        assert corr == 1.0
        self.assertAlmostEqual(p_value, 0.0833, 2)

    def test_intra_list_similarity(self):
        features = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        recommendation_1 = [0, 0, 0]
        recommendation_2 = [0, 1, 2]

        # same similarities
        sim = intra_list_similarity([recommendation_1], features)
        assert sim == 1.0

        # no similarities
        sim = intra_list_similarity([recommendation_2], features)
        assert sim == 0.0

        # multiple recommendations
        sim = intra_list_similarity([recommendation_1, recommendation_2], features)
        assert sim == 0.5

    def test_personalization(self):

        rec1 = [0, 0, 0]
        rec2 = [1, 1, 1]
        recommendations = [rec1, rec2]
        assert personalization(recommendations) == 1.0

        rec1 = [0, 0, 0]
        rec2 = [0, 0, 0]
        recommendations = [rec1, rec2]
        assert personalization(recommendations) == 0.0
