import numpy as np
import scipy.stats.stats
import sklearn.metrics

from rexmex.utils import Annotator

ratings = Annotator()


@ratings.annotate(
    key="mse",
    lower=0.0,
    upper=float("inf"),
    higher_is_better=False,
    link="https://en.wikipedia.org/wiki/Mean_squared_error",
    description="...",
)
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


@ratings.annotate(
    key="mae",
    lower=0.0,
    upper=float("inf"),
    higher_is_better=False,
    link="https://en.wikipedia.org/wiki/Mean_absolute_error",
    description="...",
)
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


@ratings.annotate(
    key="mape",
    lower=0.0,
    upper=100.00,
    higher_is_better=False,
    link="https://en.wikipedia.org/wiki/Mean_absolute_percentage_error",
    description=r"\frac{100}{n} \sum_{t=1}^n \mid \frac{A_t - F_t}{A_t} \mid",
)
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


@ratings.annotate(
    key="r_squared",
    lower=0.0,
    upper=1.0,
    higher_is_better=True,
    link="https://en.wikipedia.org/wiki/Coefficient_of_determination",
    description=r"1 - \frac{\sum_i (y_i - f_i)^2}{\sum_i (y_i - \bar{y})^2}",
)
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


@ratings.annotate(
    key="pearson_correlation",
    lower=-1.0,
    upper=1.0,
    higher_is_better=True,
    link="https://en.wikipedia.org/wiki/Pearson_correlation_coefficient",
    description="...",
)
def pearson_correlation_coefficient(y_true: np.array, y_score: np.array) -> float:
    """
    Calculate the Pearson correlation coefficient for a ground-truth prediction vector pair.

    Args:
        y_true (array-like): An N x 1 array of ground truth values.
        y_score (array-like):  An N x 1 array of predicted values.
    Returns:
        rho (float): The value of the correlation coefficient.
    """
    rho, _prob = scipy.stats.stats.pearsonr(y_true, y_score)
    return rho


@ratings.annotate(
    key="rmse",
    lower=0.0,
    upper=float("inf"),
    higher_is_better=False,
    link="https://en.wikipedia.org/wiki/Root-mean-square_deviation",
    description="...",
)
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


@ratings.annotate(
    key="smape",
    lower=0.0,
    upper=100.0,
    higher_is_better=False,
    link="https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error",
    description="...",
)
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
