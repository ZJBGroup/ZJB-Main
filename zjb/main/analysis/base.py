import numpy as np
from scipy.signal import butter, correlate, detrend, filtfilt, hilbert, resample
from traits.api import Enum, HasTraits

from zjb.main.data.series import TimeSeries, TimeUnit


class AnalyzerBase(HasTraits):
    @staticmethod
    def zero_mean(arr: np.ndarray, axis: int = 0):
        arr_mean = arr.mean(axis=axis, keepdims=True)
        arr_zero_mean = arr - arr_mean
        return arr_zero_mean

    @staticmethod
    def compute_norms(arr: np.ndarray, centered=False, axis: int = 0):
        if centered:
            arr = arr - arr.mean(axis=~axis, keepdims=True)
        norms = np.sqrt(np.sum(arr**2, axis=axis))
        return norms

    @staticmethod
    def extract_dt_unit(arr: TimeSeries) -> tuple:
        unit_coefficient = {
            TimeUnit.MILLISECOND: 0.001,
            TimeUnit.SECOND: 1.0,
            TimeUnit.UNKNOWN: np.nan,
        }
        if arr.sample_unit not in unit_coefficient.keys():
            raise ValueError("Missing unit coefficient")
        unit = unit_coefficient[arr.sample_unit]
        if arr.sample_period <= 0 or np.isnan(unit).any():
            dt = np.nan
        else:
            dt = arr.sample_period * unit
        return dt, unit

    @staticmethod
    def resample_series(arr: np.ndarray, fs_old: float, fs_new: float, axis: int = 0):
        if fs_new == fs_old:
            pass
        else:
            dt_old = 1 / fs_old
            dt_new = 1 / fs_new
            resample_length = int(np.ceil(arr.shape[0] * dt_old / dt_new))
            arr = resample(arr, resample_length, axis=axis)

        return arr

    @classmethod
    def hilbert_trans(cls, arr: np.ndarray, axis: int = 0):
        arr_zero_mean = cls.zero_mean(arr, axis)
        arr_norm = arr_zero_mean / np.std(arr, axis=axis, keepdims=True)
        analytic_signal = hilbert(arr_norm)
        phases = np.angle(analytic_signal)
        return phases
