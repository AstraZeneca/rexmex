import numpy
from sklearn.metrics precision_recall_curve, auc
from sklearn.metrics import mean_squared_error

def pr_auc_score(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    return pr_auc

def root_mean_squared_error(y_true, y_scores):
    rmse = mean_squared_error(y_true, y_scores)**0.5
    return rmse

def symmetric_mean_absolute_percentage_error(y_true, y_scores):
    smape = np.mean(np.abs(y_scores-y_true)/((np.abs(y_scores)+np.abs(y_true))/2))
    return smape 

def normalize(y_true, y_scores):
    y_mean = np.mean(y_true)
    y_std = np.mean(y_true)
    y_true = , (y_true - y_mean)/y_std
    y_scores = (y_scores - y_mean)/y_std