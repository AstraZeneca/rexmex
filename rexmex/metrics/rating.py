import numpy as np
import sklearn.metrics
import scipy.stats.stats


def mean_squared_error(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the mean squared error (MSE) for a ground-truth prediction vector pair.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        mse (float): The mean squared error value.
    """
    mse = sklearn.metrics.mean_squared_error(y_true, y_score)
    return mse


def mean_absolute_error(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the mean absolute error (MAE) for a ground-truth prediction vector pair.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        mae (float): The mean absolute error value.
    """
    mae = sklearn.metrics.mean_absolute_error(y_true, y_score)
    return mae


def mean_absolute_percentage_error(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the mean absolute percentage error (MAPE) for a ground-truth prediction vector pair.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        mape (float): The mean absolute percentage error value.
    """
    mape = sklearn.metrics.mean_absolute_percentage_error(y_true, y_score)
    return mape


def r2_score(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the coefficient of determination (R^2) for a ground-truth prediction vector pair.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        r2 (float): The coefficient of determination value.
    """
    r2 = sklearn.metrics.r2_score(y_true, y_score)
    return r2


def pearson_correlation_coefficient(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the Pearson correlation coefficient for a ground-truth prediction vector pair.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        rho (float): The value of the correlation coefficient.
    """
    rho = scipy.stats.stats.pearsonr(y_true, y_score)
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
    rmse = mean_squared_error(y_true, y_score) ** 0.5
    return rmse


def symmetric_mean_absolute_percentage_error(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the symmetric mean absolute percentage error (SMAPE) for a ground-truth prediction vector pair.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        smape (float): The value of the symmetric mean absolute percentage error.
    """
    smape = 100 * np.mean(np.abs(y_score - y_true) / ((np.abs(y_score) + np.abs(y_true)) / 2))
    return smape
