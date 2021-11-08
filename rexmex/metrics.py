import pandas as pd
from typing import List, Tuple
from sklearn.metrics  import roc_auc_score, f1_score

class MetricSet(object):
    """
    """
    def _get_metrics(self, filter: List[Tuple[str]]=None) -> List[Tuple[str]]:
        if filter is None:
            selected_metrics = self._metrics
        else:
            selected_metrics = [(name, metric) in self._metrics]
        return selected_metrics

    def get_performance_metrics(self, y_true, y_score, filter: List[Tuple[str]]=None) -> pd.DataFrame:
        selected_metrics = self._get_metrics(filter)
        performance_metrics = {name: [metric(y_true, y_score)] for name, metric in selected_metrics}
        performance_metrics = pd.DataFrame.from_dict(performance_metrics)
        return performance_metrics

class ClassificationMetricSet(MetricSet):
    """
    """
    def __init__(self):
        self._metrics = []
        self._metrics.append(("roc_auc", roc_auc_score))
        self._metrics.append(("f1_score", roc_auc_score))


class RankingMetricSet(MetricSet):
    pass

class RatingMetrics(MetricSet):
    pass

class CoverageMetrics(MetricSet):
    pass