import numpy as np
from scipy.signal import butter, correlate, detrend, filtfilt, hilbert, resample
from traits.api import Enum, HasTraits

from zjb.main.data.series import TimeSeries, TimeUnit


class AnalyzerBase(HasTraits):
    """
    用于对不同数据类型执行各种分析操作的基类。

    此类包含用于执行零均值归一化、范数计算、时间单位提取、重采样和希尔伯特变换等操作的静态方法。
    """

    @staticmethod
    def zero_mean(arr: np.ndarray, axis: int = 0):
        """
        沿指定轴计算数组的零均值归一化.

        Parameters
        ----------
        arr : np.ndarray
            输入数组。
        axis : int, 可选
            计算均值的轴。默认为0。

        Returns
        -------
        arr_zero_mean : np.ndarray
            沿指定轴减去均值后的数组。
        """
        arr_mean = arr.mean(axis=axis, keepdims=True)
        arr_zero_mean = arr - arr_mean
        return arr_zero_mean

    @staticmethod
    def compute_norms(arr: np.ndarray, centered=False, axis: int = 0):
        """
        计算数组的范数（norms）

        Parameters
        ----------
        arr : np.ndarray
            输入数组。
        centered : bool, 可选
            如果为True，则在计算范数之前将数组均值置零。默认为False。
        axis : int, 可选
            计算范数的轴。默认为0。

        Returns
        -------
        norms : np.ndarray
            数组的范数。
        """
        if centered:
            arr = arr - arr.mean(axis=~axis, keepdims=True)
        norms = np.sqrt(np.sum(arr**2, axis=axis))
        return norms

    @staticmethod
    def extract_dt_unit(arr: TimeSeries) -> tuple:
        """
        从TimeSeries对象提取采样间隔（dt）和单位。
        Parameters
        ----------
        arr : TimeSeries
            输入的的TimeSeries对象。

        Returns
        -------
        tuple
            包含采样间隔和单位的元组。

        """
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
        """
        将数组重采样到新的采样频率。

        Parameters
        ----------
        arr : np.ndarray
            输入数组。
        fs_old : float
            旧采样频率。
        fs_new : float
            新采样频率。
        axis : int, 可选
            重采样的轴。默认为0。

        Returns
        -------
        arr : np.ndarray
            重采样后的数组。
        """
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
        """
        对数组应用希尔伯特变换，并返回相位角。

        Parameters
        ----------
        arr : np.ndarray
            输入的数组。
        axis : int, 可选
            应用希尔伯特变换的轴。默认为0。

        Returns
        -------
        phases : np.ndarray
            希尔伯特变换后数组的相位角。
        """
        arr_zero_mean = cls.zero_mean(arr, axis)
        arr_norm = arr_zero_mean / np.std(arr, axis=axis, keepdims=True)
        analytic_signal = hilbert(arr_norm)
        phases = np.angle(analytic_signal)
        return phases
