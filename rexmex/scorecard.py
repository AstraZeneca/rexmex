import numpy as np
import pandas as pd
import rexmex.metricset
from typing import List

class ScoreCard(object):
    """
    A score card can be used to aggregate metrics, plot those, and generate performance reports.
    """
    def __init__(self, metric_set: rexmex.metricset.MetricSet):   
        self._metric_set = metric_set

    def _get_performance_metrics(self, y_true: np.array, y_score: np.array) -> pd.DataFrame:
        """
        A method to get the performance metrics for a pair of vectors.

        Args:
            y_true (np.array): A vector of ground truth values.
            y_score (np.array): A vector of model predictions.
        Returns:
            performance_metrics (pd.DataFrame): The performance metrics calculated from the vectors.
        """
        performance_metrics = {name: [metric(y_true, y_score)] for name, metric in self._metric_set.items()}
        performance_metrics = pd.DataFrame.from_dict(performance_metrics)
        return performance_metrics

    def generate_report(self, scores: pd.DataFrame, groupping: List[str]=None) -> pd.DataFrame:
        """
        A method to calculate (aggregated) performance metrics based
        on a dataframe of ground truth and predictions. It assumes that the dataframe has the `y_true`
        and `y_score` keys in the dataframe.

        Args:
            scores (pd.DataFrame): A dataframe with the scores and ground-truth - it has the `y_true`
            and `y_score` keys.
            groupping (list): A list of performance groupping variable names.
        Returns:
            report (pd.DataFrame): The performance report.
        """
        if groupping is not None:
             scores = scores.groupby(groupping)
        report = scores.apply(lambda group: self._get_performance_metrics(group.y_true, group.y_score))
        return report

    def __repr__(self):
        """
        A representation of the ScoreCard object.
        """
        return "ScoreCard()"

    def print_metrics(self):
        """
        Printing the name of metrics.
        """
        print({k for k in self._metric_set.keys()})