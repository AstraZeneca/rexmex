import numpy as np
import sklearn.metrics
import scipy.stats.stats

def mean_squared_error(y_true: np.array, y_score: np.array) -> float:
    mse =  sklearn.metrics.mean_squared_error(y_true, y_score)
    return mse

def mean_absolute_error(y_true: np.array, y_score: np.array) -> float:
    mae =  sklearn.metrics.mean_absolute_error(y_true, y_score)
    return mae

def mean_absolute_percentage_error(y_true: np.array, y_score: np.array) -> float:
    mape =  sklearn.metrics.mean_absolute_percentage_error(y_true, y_score)
    return mape

def r2_score(y_true: np.array, y_score: np.array) -> float:
    r2 =  sklearn.metrics.r2_score(y_true, y_score)
    return r2

def pearson_correlation_coefficient(y_true: np.array, y_score: np.array) -> float:
    rho =  scipy.stats.stats.pearsonr(y_true, y_score)
    return rho

def root_mean_squared_error(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the root mean squared error (RMSE) for a ground-truth prediction vector pair.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        rmse (float): The value of the root mean squared error.
    """
    rmse = mean_squared_error(y_true, y_score)**0.5
    return rmse

def symmetric_mean_absolute_percentage_error(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the symmetric mean absolute percentage error (SMAPE) for a ground-truth prediction vector pair.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        smape (float): The value of the symmetric mean absolute percentage error .
    """
    smape = 100*np.mean(np.abs(y_score - y_true)/((np.abs(y_score)+np.abs(y_true))/2))
    return smape