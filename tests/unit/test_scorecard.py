import sys
import unittest
from io import StringIO

from sklearn.model_selection import train_test_split

from rexmex.dataset import DatasetReader
from rexmex.metricset import ClassificationMetricSet
from rexmex.scorecard import ScoreCard


class TestMetricSet(unittest.TestCase):
    def setUp(self):
        reader = DatasetReader()
        self.scores = reader.read_dataset()
        self.metric_set = ClassificationMetricSet()
        self.score_card = ScoreCard(self.metric_set)

    def test_representation(self):
        assert repr(self.score_card) == "ScoreCard(metric_set=ClassificationMetricSet())"

    def test_printing(self):
        self.score_card.metric_set.filter_metrics(["roc_auc", "pr_auc"])
        captured = StringIO()
        sys.stdout = captured
        self.score_card.print_metrics()
        sys.stdout = sys.__stdout__
        out = captured.getvalue().strip("\n")
        assert out == str({"roc_auc", "pr_auc"})

    def test_score_filtering(self):
        train, remainder = train_test_split(self.scores[["source_id", "target_id"]], test_size=0.2)
        train, test = train_test_split(train, test_size=0.5)
        test, validation = train_test_split(test, test_size=0.5)

        new_scores = self.score_card.filter_scores(
            scores=self.scores,
            training_set=train,
            testing_set=test,
            validation_set=validation,
            columns=["source_id", "target_id"],
        )

        assert remainder.shape[0] == new_scores.shape[0]
