import pandas as pd
import rexmex.metricset
from typing import List

class ScoreCard(object):
    def __init__(self, metric_set: rexmex.metricset.MetricSet):
        
        self.metric_set = metric_set

    def get_performance_metrics(self, y_true, y_score) -> pd.DataFrame:
        """
        """
        performance_metrics = {name: [metric(y_true, y_score)] for name, metric in self.metric_set.items()}
        performance_metrics = pd.DataFrame.from_dict(performance_metrics)
        return performance_metrics

    def group_calculate(self, scores: pd.DataFrame, groupping: List[str]=None) -> pd.DataFrame:
        if groupping is not None:
             scores = scores.groupby(groupping)
        performance_metrics = scores.apply(lambda group: self.get_performance_metrics(group.y_true, group.y_score))
        return performance_metrics