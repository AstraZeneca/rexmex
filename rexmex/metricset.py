from typing import List, Tuple

from rexmex.utils import binarize, normalize

from rexmex.metrics.classification import roc_auc_score
from rexmex.metrics.classification import accuracy_score, balanced_accuracy_score
from rexmex.metrics.classification import (
    pr_auc_score,
    specificity,
    fowlkes_mallows_index,
)
from rexmex.metrics.classification import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    matthews_correlation_coefficient,
)

from rexmex.metrics.rating import pearson_correlation_coefficient
from rexmex.metrics.rating import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)
from rexmex.metrics.rating import (
    symmetric_mean_absolute_percentage_error,
    root_mean_squared_error,
)


class MetricSet(dict):
    """
    A metric set is a special dictionary that contains metric
     name keys and evaluation metric function values.
    """

    def filter_metrics(self, filter: List[str] = None):
        """
        A method to keep a list of metrics.

        Args:
            filter (str): A list of metric names to keep.
        Returns:
            self: The metric set after the metrics were filtered out.
        """
        for name in list(self.keys()):
            if name not in filter:
                del self[name]
        return self

    def add_metrics(self, metrics: List[Tuple]):
        """
        A method to add metric functions from a list of function names and functions.

        Args:
            metrics (List[Tuple]): A list of metric name and metric function tuples.
        Returns:
            self: The metric set after the metrics were added.
        """
        for metric in metrics:
            metric_name, metric_function = metric
            self[metric_name] = metric_function
        return self

    def __repr__(self):
        """
        A representation of the MetricSet object.
        """
        return "MetricSet()"

    def print_metrics(self):
        """
        Printing the name of metrics.
        """
        print({k for k in self.keys()})

    def __add__(self, other_metric_set):
        """
        Adding two metric sets together with the addition syntactic sugar operator.

        Args:
            other_metric_set (rexmex.metricset.MetricSet): Metric set added from the right.
        Returns:
            new_metric_set (rexmex.metricset.MetricSet): The combined metric set.
        """
        new_metric_set = self
        for name, metric in other_metric_set.items():
            new_metric_set[name] = metric
        return new_metric_set


class ClassificationMetricSet(MetricSet):
    """
    A set of classification metrics with the following metrics included:

    | **Area Under the Receiver Operating Characteristic Curve**
    | **Area Under the Precision Recall Curve**
    | **Average Precision**
    | **F-1 Score**
    | **Matthew's Correlation Coefficient**
    | **Fowlkes-Mallows Index**
    | **Precision**
    | **Recall**
    | **Specificity**
    | **Accuracy**
    | **Balanced Accuracy**
    """

    def __init__(self):
        self["roc_auc"] = roc_auc_score
        self["pr_auc"] = pr_auc_score
        self["average_precision"] = average_precision_score
        self["f1_score"] = binarize(f1_score)
        self["matthews_correlation_coefficent"] = binarize(matthews_correlation_coefficient)
        self["fowlkes_mallows_index"] = binarize(fowlkes_mallows_index)
        self["precision"] = binarize(precision_score)
        self["recall"] = binarize(recall_score)
        self["specificity"] = binarize(specificity)
        self["accuracy"] = binarize(accuracy_score)
        self["balanced_accuracy"] = binarize(balanced_accuracy_score)

    def __repr__(self):
        """
        A representation of the ClassificationMetricSet object.
        """
        return "ClassificationMetricSet()"


class RatingMetricSet(MetricSet):
    """
    A set of rating metrics with the following metrics included:

    | **Mean Absolute Error**
    | **Mean Squared Error**
    | **Root Mean Squared Error**
    | **Mean Absolute Percentage Error**
    | **Symmetric Mean Absolute Percentage Error**
    | **Coefficient of Determination**
    | **Pearson Correlation Coefficient**
    """

    def __init__(self):
        self["mae"] = mean_absolute_error
        self["mse"] = mean_squared_error
        self["rmse"] = root_mean_squared_error
        self["mape"] = mean_absolute_percentage_error
        self["smape"] = symmetric_mean_absolute_percentage_error
        self["r_squared"] = r2_score
        self["pearson_correlation"] = pearson_correlation_coefficient

    def normalize_metrics(self):
        """
        A method to normalize a set of metrics.

        Returns:
            self: The metric set after the metrics were normalized.
        """
        for name, metric in self.items():
            self[name] = normalize(metric)
        return self

    def __repr__(self):
        """
        A representation of the RatingMetricSet object.
        """
        return "RatingMetricSet()"


class CoverageMetricSet(MetricSet):
    """
    A set of coverage metrics with the following metrics included:

    """

    def __repr__(self):
        """
        A representation of the CoverageMetricSet object.
        """
        return "CoverageMetricSet()"


class RankingMetricSet(MetricSet):
    """
    A set of ranking metrics with the following metrics included:

    """

    def __repr__(self):
        """
        A representation of the RankingMetricSet object.
        """
        return "RankingMetricSet()"
