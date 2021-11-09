import pandas as pd
import rexmex.metrics
from typing import List

class ScoreCard(object):
    def __init__(self, metric_set: rexmex.metrics.MetricSet):
        
        self.metric_set = metric_set

    def calculate(self, scores: pd.DataFrame, groupping: List[str]=None, filter: List[str]=None) -> pd.DataFrame:
        if groupping is not None:
             scores = scores.groupby(groupping)
        performance_metrics = scores.apply(lambda group: self.metric_set.get_performance_metrics(group.y_true, group.y_score, filter))
        return performance_metrics