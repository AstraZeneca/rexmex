from typing import Collection, List, Mapping, Tuple

import numpy as np
import pandas as pd

from rexmex.utils import Metric

__all__ = [
    "ScoreCard",
    "CoverageScoreCard",
]


class ScoreCard:
    """
    A scorecard can be used to aggregate metrics, plot those, and generate performance reports.
    """

    metric_set: Mapping[str, Metric]

    def __init__(self, metric_set: Mapping[str, Metric]):
        self.metric_set = metric_set

    def get_performance_metrics(self, y_true: np.array, y_score: np.array) -> pd.DataFrame:
        """
        A method to get the performance metrics for a pair of vectors.

        Args:
            y_true (np.array): A vector of ground truth values.
            y_score (np.array): A vector of model predictions.
        Returns:
            performance_metrics (pd.DataFrame): The performance metrics calculated from the vectors.
        """
        performance_metrics = {name: [metric(y_true, y_score)] for name, metric in self.metric_set.items()}
        performance_metrics_df = pd.DataFrame.from_dict(performance_metrics)
        return performance_metrics_df

    def generate_report(self, scores_to_evaluate: pd.DataFrame, grouping: List[str] = None) -> pd.DataFrame:
        """
        A method to calculate (aggregated) performance metrics based
        on a dataframe of ground truth and predictions. It assumes that the dataframe has the `y_true`
        and `y_score` keys in the dataframe.

        Args:
            scores_to_evaluate (pd.DataFrame): A dataframe with the scores and ground-truth - it has the `y_true`
            and `y_score` keys.
            grouping (list): A list of performance grouping variable names.
        Returns:
            report (pd.DataFrame): The performance report.
        """
        if grouping is not None:
            scores_to_evaluate = scores_to_evaluate.groupby(grouping)
            report = scores_to_evaluate.apply(lambda group: self.get_performance_metrics(group.y_true, group.y_score))
        else:
            report = self.get_performance_metrics(scores_to_evaluate.y_true, scores_to_evaluate.y_score)
        return report

    def filter_scores(
        self,
        scores: pd.DataFrame,
        training_set: pd.DataFrame,
        testing_set: pd.DataFrame,
        validation_set: pd.DataFrame,
        columns: List[str],
    ) -> pd.DataFrame:
        """
        A method to filter out those entries which also appear in either the training, testing or validation sets.
        `The original is here: <https://papers.nips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf>.`
        Args:
            scores (pd.DataFrame): A dataframe with the scores.
            training_set (pd.DataFrame): A dataframe of training data points.
            testing_set (pd.DataFrame): A dataframe of testing data points.
            validation_set (pd.DataFrame): A dataframe of validation data points.
            columns (list): A list of column names used for cross-referencing.
        Returns:
            scores (pd.DataFrame): The scores for data points which are not in the reference sets.
        """
        scores_columns = list(scores.columns.tolist())
        in_sample_examples = pd.concat([training_set, testing_set, validation_set])
        scores = scores.merge(in_sample_examples.drop_duplicates(), on=columns, how="left", indicator=True)
        scores = scores[scores["_merge"] == "left_only"].reset_index()[scores_columns]
        return scores

    def __repr__(self):
        """
        A representation of the ScoreCard object.
        """
        return f"ScoreCard(metric_set={self.metric_set!r})"

    def print_metrics(self):
        """
        Printing the name of metrics.
        """
        print({k for k in self.metric_set.keys()})


class CoverageScoreCard(ScoreCard):
    """
    Coverage scorecard can be used to aggregate coverage-related metrics, plot those, and generate performance reports.
    """

    def __init__(self, metric_set: Mapping[str, Metric], all_users: Collection[str], all_items: Collection[str]):
        """
        all_users and all_items define the relevant user and item space.
        """

        super().__init__(metric_set)
        self.all_users = all_users
        self.all_items = all_items

    def get_coverage_metrics(self, recommendations: List[Tuple]) -> pd.DataFrame:
        """
        Gets all coverage (performance) values using the defined metric_set. It expects a list of tuples of user/item
        combinations, e.g., [(user_1, item_1), (user_2, item1),]. The space of possible users and items to recommend
        is defined during initalisation of this class.

        Args:
            recommendations List[Tuple]: recommendations of items to users, made by the evaluated system.
            The user has to decide which score or confidence levels to use prior to calling this ScoreCard.

        Returns:
            performance_metrics (pd.DataFrame): The coverage (performance) metrics calculated from the recommendations.

        """
        performance_metrics = {
            name: [metric((self.all_users, self.all_items), recommendations)]
            for name, metric in self.metric_set.items()
        }
        performance_metrics_df = pd.DataFrame.from_dict(performance_metrics)
        return performance_metrics_df

    def generate_report(self, recs_to_evaluate: pd.DataFrame, grouping: List[str] = None) -> pd.DataFrame:
        """
        A method to calculate (aggregated) coverage/performance metrics based on a dataframe of predictions.
        It assumes that the dataframe has the `user` and `item` keys in the dataframe.

        Args:
            recs_to_evaluate (pd.DataFrame): A dataframe holding the recommendations (users, items). Contains
            columns `user` and `item`.
            grouping (list): A list of performance grouping variable names (e.g., different recommender settings).

        Returns:
            report (pd.DataFrame): The performance report.
        """
        if "user" not in recs_to_evaluate.columns or "item" not in recs_to_evaluate.columns:
            raise ValueError("recs_to_evaluate has to have user and item columns!")

        if grouping is not None:
            recs_to_evaluate = recs_to_evaluate.groupby(grouping)
            report = recs_to_evaluate.apply(
                lambda group: self.get_coverage_metrics(
                    list(group[["user", "item"]].itertuples(index=False, name=None))
                )
            )
        else:
            report = self.get_coverage_metrics(
                list(recs_to_evaluate[["user", "item"]].itertuples(index=False, name=None))
            )
        return report
