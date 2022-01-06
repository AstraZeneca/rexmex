from typing import Collection, List, Tuple

from rexmex.metrics.classification import classifications
from rexmex.metrics.coverage import item_coverage, user_coverage
from rexmex.metrics.rating import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    pearson_correlation_coefficient,
    r2_score,
    root_mean_squared_error,
    symmetric_mean_absolute_percentage_error,
)
from rexmex.utils import binarize, normalize


class MetricSet(dict):
    """
    A metric set is a special dictionary that contains metric
     name keys and evaluation metric function values.
    """

    def filter_metrics(self, filter: Collection[str]):
        """
        A method to keep a list of metrics.

        Args:
            filter: A list of metric names to keep.
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
        super().__init__()
        for func in classifications:
            name = func.__name__
            if name.endswith("_score"):
                name = name[: -len("_score")]
            if func.binarize:
                func = binarize(func)
            self[name] = func

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
    | **Item Coverage**
    | **User Coverage**
    """

    def __init__(self):
        self["item_coverage"] = item_coverage
        self["user_coverage"] = user_coverage

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
