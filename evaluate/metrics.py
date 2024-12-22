import numpy.typing as npt
import numpy as np


def similarity(predict: npt.ArrayLike, actual: npt.ArrayLike) -> float:
    '''
    Calculate the similarity between the predicted and actual values.
    Range [0, 1], where 1 is the best similarity.
    '''
    _denominator = np.max(predict) - np.min(predict)
    if _denominator == 0:
        _denominator = np.finfo(_denominator).eps
    _sim = 1 / (1 + np.abs(predict - actual) / _denominator)
    return np.sum(_sim) / predict.shape[0]


def nmae(predict: npt.ArrayLike, actual: npt.ArrayLike) -> float:
    '''
    Calculate the Normalized Mean Absolute Error.
    Range [0, 1], where 0 is the best NMAE.
    '''
    _denominator = np.max(actual) - np.min(actual)
    if _denominator == 0:
        _denominator = np.finfo(_denominator).eps
    return np.sum(np.abs(predict - actual) / _denominator) / predict.shape[0]


def r2(predict: npt.ArrayLike, actual: npt.ArrayLike) -> float:
    '''
    Calculate the R^2 score.
    Range [-inf, 1], where 1 is the best R^2 score.
    '''
    _mean_predict = np.mean(predict)
    _mean_actual = np.mean(actual)
    _numerator = np.sum((actual - _mean_actual) * (predict - _mean_predict))
    _denominator = np.sqrt(np.sum((actual - _mean_actual) ** 2)
                           * np.sum((predict - _mean_predict) ** 2))
    if _denominator == 0:
        _denominator = np.finfo(_denominator).eps
    return _numerator / _denominator


def rmse(predict: npt.ArrayLike, actual: npt.ArrayLike) -> float:
    '''
    Calculate the Root Mean Squared Error.
    Range [0, inf], where 0 is the best RMSE.
    '''
    return np.sqrt(np.mean((predict - actual) ** 2))


def fsd(predict: npt.ArrayLike, actual: npt.ArrayLike) -> float:
    '''
    Calculate the Fraction of Standard Deviation.
    Range [0, 1], where 0 is the best FSD.
    '''
    _std_actual = np.std(actual)
    _std_predict = np.std(predict)
    _demominator = _std_actual + _std_predict
    if _demominator == 0:
        _demominator = np.finfo(_demominator).eps
    return 2 * np.abs(_std_actual - _std_predict) / _demominator


def fb(predict: npt.ArrayLike, actual: npt.ArrayLike) -> float:
    '''
    Calculate the Fraction of Bias.
    Range [-inf, inf], where 0 is the best FB.
    '''
    _mean_actual = np.mean(actual)
    _mean_predict = np.mean(predict)
    _demominator = np.abs(_mean_actual) + np.abs(_mean_predict)
    if _demominator == 0:
        _demominator = np.finfo(_demominator).eps
    return 2 * _mean_actual - _mean_predict / _demominator


def fa2(predict: npt.ArrayLike, actual: npt.ArrayLike, upper_bound: float = 2.0, lower_bound: float = 0.5) -> float:
    '''
    Calculate the Fraction of Absolute Error.
    Range [0, 1], where 1 is the best FA2.
    '''
    y = predict / actual
    return np.where((y >= lower_bound) & (y <= upper_bound))[0].shape[0] / actual.shape[0]
